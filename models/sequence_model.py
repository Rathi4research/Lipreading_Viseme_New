import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.config import Config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LSTMSequenceModel(nn.Module):
    """LSTM-based sequence model"""

    def __init__(self,
                 input_dim=Config.VISUAL_FEATURE_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=Config.NUM_LAYERS,
                 num_visemes=12,
                 dropout=Config.DROPOUT_RATE):
        super(LSTMSequenceModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, num_visemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features (batch_size, sequence_length, input_dim)
            lengths: Actual sequence lengths for packing (optional)
        Returns:
            output: Viseme predictions (batch_size, sequence_length, num_visemes)
            hidden: Hidden states (batch_size, sequence_length, hidden_dim * 2)
        """
        batch_size, seq_len, _ = x.shape

        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack sequences if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to viseme space
        output = self.output_projection(lstm_out)

        return output, lstm_out


class TransformerSequenceModel(nn.Module):
    """Transformer-based sequence model"""

    def __init__(self,
                 input_dim=Config.VISUAL_FEATURE_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_heads=8,
                 num_layers=6,
                 num_visemes=12,
                 dropout=Config.DROPOUT_RATE):
        super(TransformerSequenceModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=Config.SEQUENCE_LENGTH)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_visemes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features (batch_size, sequence_length, input_dim)
            mask: Attention mask (batch_size, sequence_length)
        Returns:
            output: Viseme predictions (batch_size, sequence_length, num_visemes)
            hidden: Hidden states (batch_size, sequence_length, hidden_dim)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)

        # Create padding mask if not provided
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)

        # Transformer forward
        hidden = self.transformer(x, src_key_padding_mask=mask)
        hidden = self.dropout(hidden)

        # Project to viseme space
        output = self.output_projection(hidden)

        return output, hidden


class HybridSequenceModel(nn.Module):
    """Hybrid model combining LSTM and Transformer"""

    def __init__(self,
                 input_dim=Config.VISUAL_FEATURE_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_visemes=12):
        super(HybridSequenceModel, self).__init__()

        # LSTM for local temporal modeling
        self.lstm = LSTMSequenceModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=1,
            num_visemes=hidden_dim // 2
        )

        # Transformer for global context
        self.transformer = TransformerSequenceModel(
            input_dim=hidden_dim,  # Input from LSTM + original features
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=3,
            num_visemes=num_visemes
        )

        # Fusion layer
        self.fusion = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, lengths=None, mask=None):
        """
        Args:
            x: Input features (batch_size, sequence_length, input_dim)
            lengths: Sequence lengths for LSTM
            mask: Attention mask for Transformer
        Returns:
            output: Final viseme predictions (batch_size, sequence_length, num_visemes)
        """
        # LSTM processing
        lstm_out, lstm_hidden = self.lstm(x, lengths)

        # Combine original features with LSTM hidden states
        combined = torch.cat([x, lstm_hidden], dim=-1)
        fused = self.fusion(combined)

        # Transformer processing
        output, transformer_hidden = self.transformer(fused, mask)

        return output, transformer_hidden