import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from models.sequence_model import PositionalEncoding, _run_lstm


def time_match(x, target_len: int):
    """
    Align sequence length S -> target_len.
    Expands or pools along time dimension.
    """
    B, S, E = x.shape
    if S == target_len:
        return x
    if S < target_len:
        scale = math.ceil(target_len / S)
        x = x.repeat_interleave(scale, dim=1)[:, :target_len, :]
    else:
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)
    return x


class GRIDLanguageModel(nn.Module):
    """Word-level context model over the frame sequence.

    Input is a per-frame feature sequence (in the full model: the phoneme-LM features, so this
    is the "word" level of a bottom-up hierarchy). Word indices can still be given directly for
    stand-alone use. A learned absolute frame-position embedding lets it pick up GRID's fixed
    sentence layout ([command][color][preposition][letter][digit][adverb] occur at roughly
    stable times within the 3 s clip). (Previously the input was always all-zero tokens and the
    position was frame_index % 6, which treated individual frames as GRID word slots.)
    """

    def __init__(self, vocab_size=len(Config.VOCABULARY),
                 embed_dim=Config.HIDDEN_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=2,
                 max_len=None):
        super(GRIDLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len or Config.SEQUENCE_LENGTH

        # Word embeddings (used only when word indices are passed in)
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        # Absolute frame-position embeddings
        self.position_embedding = nn.Embedding(self.max_len, embed_dim)

        # LSTM language model
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=Config.DROPOUT_RATE if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)

        # Self-attention over the sequence
        self.grammar_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        self._init_word_type_mappings()

    def _init_word_type_mappings(self):
        """Initialize mappings for GRID sentence structure"""
        self.word_to_type = {}
        self.type_to_words = {
            0: Config.COMMANDS,  # command
            1: Config.COLORS,  # color
            2: Config.PREPOSITIONS,  # preposition
            3: Config.LETTERS,  # letter
            4: Config.DIGITS,  # digit
            5: Config.ADVERBS  # adverb
        }

        for type_id, words in self.type_to_words.items():
            for word in words:
                if word in Config.VOCABULARY:
                    self.word_to_type[Config.VOCABULARY.index(word)] = type_id

    def forward(self, input_features=None, word_indices=None,
                sequence_length=Config.SEQUENCE_LENGTH, batch_size=None, lengths=None, pad_mask=None):
        """
        Args:
            input_features: per-frame input features (B, T, embed_dim), or
            word_indices: word indices (B, T); if neither is given an all-zero input is used
        Returns:
            language_features: (B, T, hidden_dim * 2)
            word_logits: (B, T, vocab_size)
        """
        device = next(self.parameters()).device
        if input_features is not None:
            embeds = input_features
        elif word_indices is not None:
            embeds = self.word_embedding(word_indices)
        else:
            embeds = torch.zeros(batch_size or 1, sequence_length, self.embed_dim, device=device)

        batch_size, seq_len, _ = embeds.shape
        positions = torch.arange(seq_len, device=embeds.device).clamp_max(self.max_len - 1)
        embeds = self.dropout(embeds + self.position_embedding(positions).unsqueeze(0))

        lstm_out = self.dropout(_run_lstm(self.lstm, embeds, lengths))

        attended_out, _ = self.grammar_attention(lstm_out, lstm_out, lstm_out, key_padding_mask=pad_mask,
                                                 need_weights=False)
        language_features = lstm_out + attended_out

        word_logits = self.output_projection(language_features)
        return language_features, word_logits

    def get_grammar_constraints(self, position):
        """Valid words for the given *word* position (0-5) in a GRID sentence"""
        grid_position = position % 6
        if grid_position in self.type_to_words:
            valid_words = self.type_to_words[grid_position]
            return [Config.VOCABULARY.index(word) for word in valid_words if word in Config.VOCABULARY]
        return list(range(self.vocab_size))

    def apply_grammar_constraints(self, word_probs, positions):
        """Apply grammar constraints to word-level (not frame-level) probabilities"""
        batch_size, seq_len, vocab_size = word_probs.shape
        constrained_probs = torch.full_like(word_probs, -float('inf'))

        for i in range(seq_len):
            valid_indices = self.get_grammar_constraints(i)
            for batch_idx in range(batch_size):
                constrained_probs[batch_idx, i, valid_indices] = word_probs[batch_idx, i, valid_indices]

        return F.softmax(constrained_probs, dim=-1)


class PhonemeLanguageModel(nn.Module):
    """Context model over a (soft) phoneme sequence"""

    def __init__(self,
                 num_phonemes=len(Config.PHONEMES),
                 embed_dim=Config.HIDDEN_DIM,
                 hidden_dim=Config.HIDDEN_DIM):
        super(PhonemeLanguageModel, self).__init__()

        self.num_phonemes = num_phonemes
        self.embed_dim = embed_dim

        self.phoneme_embedding = nn.Embedding(num_phonemes, embed_dim)

        # Transformers are order-agnostic without positional information
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max(Config.SEQUENCE_LENGTH, 512))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)

        self.output_projection = nn.Linear(embed_dim, num_phonemes)

    def forward(self, phoneme_probs=None, phoneme_indices=None,
                sequence_length=Config.SEQUENCE_LENGTH, batch_size=None, pad_mask=None):
        """
        Args:
            phoneme_probs: soft phoneme distribution per frame (B, T, num_phonemes) — in the full
                model this is the visual stream's own estimate, never the ground truth.
            phoneme_indices: hard phoneme indices (B, T), alternative to phoneme_probs.
            If neither is given an all-zero input is used.
        Returns:
            phoneme_features: (B, T, embed_dim)
            phoneme_logits: (B, T, num_phonemes)
        """
        if phoneme_probs is not None:
            # Expected embedding under the distribution
            phoneme_embeds = torch.matmul(phoneme_probs, self.phoneme_embedding.weight)
        else:
            if phoneme_indices is None:
                phoneme_indices = torch.zeros(batch_size or 1, sequence_length, dtype=torch.long,
                                              device=next(self.parameters()).device)
            phoneme_embeds = self.phoneme_embedding(phoneme_indices)

        # PositionalEncoding expects (T, B, E)
        phoneme_embeds = self.pos_encoding(phoneme_embeds.transpose(0, 1)).transpose(0, 1)

        phoneme_features = self.transformer(phoneme_embeds, src_key_padding_mask=pad_mask)
        phoneme_logits = self.output_projection(phoneme_features)
        return phoneme_features, phoneme_logits


class HierarchicalLanguageModel(nn.Module):
    """Phoneme-level context -> word-level context, combined with cross-attention"""

    def __init__(self, vocab_size, phoneme_vocab_size):
        super(HierarchicalLanguageModel, self).__init__()

        self.phoneme_lm = PhonemeLanguageModel(num_phonemes=phoneme_vocab_size)
        self.word_lm = GRIDLanguageModel(vocab_size=vocab_size)

        # Project phoneme features (HIDDEN_DIM) to the word-feature size (HIDDEN_DIM * 2)
        self.phoneme_proj = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM * 2)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.HIDDEN_DIM * 2,
            num_heads=Config.NUM_HEADS,
            batch_first=True
        )

        self.fc_out = nn.Linear(Config.HIDDEN_DIM * 2, vocab_size)

    def forward(self, phoneme_probs=None, sequence_length=Config.SEQUENCE_LENGTH, batch_size=None,
                lengths=None, pad_mask=None):
        # Bottom-up: phoneme context first, then word context over the phoneme features
        phoneme_features, _ = self.phoneme_lm(phoneme_probs=phoneme_probs, sequence_length=sequence_length,
                                              batch_size=batch_size, pad_mask=pad_mask)
        word_features, _ = self.word_lm(input_features=phoneme_features, lengths=lengths, pad_mask=pad_mask)

        phoneme_features = self.phoneme_proj(phoneme_features)
        target_len = max(word_features.size(1), phoneme_features.size(1))
        word_features = time_match(word_features, target_len)
        phoneme_features = time_match(phoneme_features, target_len)

        attended_phonemes, _ = self.cross_attention(
            phoneme_features,  # query
            word_features,  # key
            word_features,  # value
            key_padding_mask=pad_mask,
            need_weights=False
        )

        combined_features = attended_phonemes + word_features
        word_probs = F.softmax(self.fc_out(combined_features), dim=-1)
        return combined_features, word_probs


class ContextualLanguageModel(nn.Module):
    """Hierarchical LM with speaker adaptation.

    Speaker embeddings have one extra "unknown speaker" slot (Config.UNKNOWN_SPEAKER_ID). During
    training each sample's speaker id is replaced by it with probability Config.SPEAKER_DROPOUT,
    so that slot is trained and can be used for speakers not seen in training (the usual
    speaker-independent split, where val/test speakers would otherwise get untrained embeddings).
    """

    def __init__(self, num_speakers=Config.NUM_SPEAKERS, vocab_size=len(Config.VOCABULARY),
                 speaker_dropout=Config.SPEAKER_DROPOUT):
        super().__init__()
        self.base_lm = HierarchicalLanguageModel(
            vocab_size=Config.VOCAB_SIZE,
            phoneme_vocab_size=Config.PHONEME_VOCAB_SIZE
        )

        self.unknown_speaker_id = num_speakers
        self.speaker_dropout = speaker_dropout
        self.speaker_embedding = nn.Embedding(num_speakers + 1, Config.HIDDEN_DIM * 2)

        self.context_adaptation = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 4, Config.HIDDEN_DIM * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM * 2)  # 512 -> 512
        )

        self.classifier = nn.Linear(Config.HIDDEN_DIM * 2, vocab_size)

    def forward(self, phoneme_probs=None, speaker_ids=None, sequence_length=Config.SEQUENCE_LENGTH,
                lengths=None, pad_mask=None):
        batch_size = speaker_ids.size(0) if speaker_ids is not None else None
        base_features, _ = self.base_lm(phoneme_probs=phoneme_probs, sequence_length=sequence_length,
                                        batch_size=batch_size, lengths=lengths, pad_mask=pad_mask)

        if speaker_ids is not None:
            if self.training and self.speaker_dropout > 0:
                drop = torch.rand(speaker_ids.shape, device=speaker_ids.device) < self.speaker_dropout
                speaker_ids = speaker_ids.masked_fill(drop, self.unknown_speaker_id)

            speaker_embeds = self.speaker_embedding(speaker_ids)  # (B, H)
            seq_len = base_features.size(1)
            speaker_embeds = speaker_embeds.unsqueeze(1).expand(-1, seq_len, -1)

            combined = torch.cat([base_features, speaker_embeds], dim=-1)  # (B, L, 1024)
            adapted_features = self.context_adaptation(combined)  # (B, L, 512)
        else:
            adapted_features = base_features

        word_logits = self.classifier(adapted_features)  # (B, L, V)
        return adapted_features, word_logits
