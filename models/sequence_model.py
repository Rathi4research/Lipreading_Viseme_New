"""Temporal sequence encoders over per-frame visual features.

All encoders take (B, T, input_dim) features plus the real clip lengths / padding mask and return
(B, T, hidden_dim). Selected with Config.SEQ_MODEL / main.py --seq-model:

  bilstm       2-layer bidirectional LSTM
  transformer  4-layer Transformer encoder (sinusoidal positions)
  hybrid       1-layer BiLSTM, concatenated with its input, then a 3-layer Transformer (original design)
  conformer    4 Conformer blocks (feed-forward / self-attention / depthwise convolution)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config

SEQ_MODELS = ('bilstm', 'transformer', 'hybrid', 'conformer')


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding; expects (T, B, E) input (kept for the language model)."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


def add_positions(x, pos_encoding):
    """Add sinusoidal positions to batch-first (B, T, E) features."""
    return pos_encoding(x.transpose(0, 1)).transpose(0, 1)


def _transformer(d_model, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
                                       dropout=dropout, batch_first=True)
    # enable_nested_tensor=False: the padding-mask fast path has no MPS kernel
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


def _run_lstm(lstm, x, lengths):
    """Run an LSTM over padded input, packing by the real lengths so padding never feeds the state."""
    if lengths is None:
        return lstm(x)[0]
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    return out


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=Config.DROPOUT_RATE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None, pad_mask=None):
        return self.dropout(_run_lstm(self.lstm, x, lengths))


class TransformerSeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, num_heads=4, dropout=Config.DROPOUT_RATE):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max(Config.SEQUENCE_LENGTH, 512))
        self.transformer = _transformer(hidden_dim, num_heads, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None, pad_mask=None):
        x = add_positions(self.input_projection(x), self.pos_encoding)
        return self.dropout(self.transformer(x, src_key_padding_mask=pad_mask))


class HybridEncoder(nn.Module):
    """BiLSTM for local context, concatenated with its input, then a Transformer for global context."""

    def __init__(self, input_dim, hidden_dim, dropout=Config.DROPOUT_RATE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.transformer = TransformerSeqEncoder(hidden_dim, hidden_dim, num_layers=3, num_heads=4, dropout=dropout)

    def forward(self, x, lengths=None, pad_mask=None):
        lstm_out = self.lstm_dropout(_run_lstm(self.lstm, x, lengths))
        fused = self.fusion(torch.cat([x, lstm_out], dim=-1))
        return self.transformer(fused, lengths, pad_mask)


class ConformerBlock(nn.Module):
    """Macaron feed-forward / multi-head self-attention / depthwise-convolution block (Gulati et al., 2020)."""

    def __init__(self, d_model, num_heads=4, kernel_size=15, dropout=Config.DROPOUT_RATE):
        super().__init__()

        def ffn():
            return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))

        self.ff1, self.ff2 = ffn(), ffn()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.conv_bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.conv_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.attn_dropout(h)
        h = self.conv_norm(x)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)  # padding must not leak through the convolution
        h = F.glu(self.pointwise1(h.transpose(1, 2)), dim=1)
        h = self.pointwise2(F.silu(self.conv_bn(self.depthwise(h))))
        x = x + self.conv_dropout(h.transpose(1, 2))
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, num_heads=4, dropout=Config.DROPOUT_RATE):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max(Config.SEQUENCE_LENGTH, 512))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ConformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, lengths=None, pad_mask=None):
        x = self.dropout(add_positions(self.input_projection(x), self.pos_encoding))
        for block in self.blocks:
            x = block(x, pad_mask)
        return x


def build_sequence_encoder(kind, input_dim=Config.VISUAL_FEATURE_DIM, hidden_dim=Config.HIDDEN_DIM):
    if kind == 'bilstm':
        return BiLSTMEncoder(input_dim, hidden_dim)
    if kind == 'transformer':
        return TransformerSeqEncoder(input_dim, hidden_dim)
    if kind == 'hybrid':
        return HybridEncoder(input_dim, hidden_dim)
    if kind == 'conformer':
        return ConformerEncoder(input_dim, hidden_dim)
    raise ValueError(f"seq_model must be one of {SEQ_MODELS}, got {kind!r}")
