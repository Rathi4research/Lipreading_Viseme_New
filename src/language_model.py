import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.config import Config

# ✅ put here
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
    """Language model specifically designed for GRID corpus structure"""

    def __init__(self, vocab_size=len(Config.VOCABULARY),
                 embed_dim=Config.HIDDEN_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=2):
        super(GRIDLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        # Position embeddings for GRID sentence structure
        # GRID: [command] [color] [preposition] [letter] [digit] [adverb]
        self.position_embedding = nn.Embedding(6, embed_dim)

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

        # Grammar-aware attention
        self.grammar_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Initialize word type embeddings for GRID structure
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

        # Create reverse mapping
        for type_id, words in self.type_to_words.items():
            for word in words:
                if word in Config.VOCABULARY:
                    word_idx = Config.VOCABULARY.index(word)
                    self.word_to_type[word_idx] = type_id

    def forward(self, word_indices=None, sequence_length=Config.SEQUENCE_LENGTH, batch_size=None):
        """
        Args:
            word_indices: Word indices (batch_size, seq_len) - optional for unconditional generation
            sequence_length: Length of sequence to generate features for
            batch_size: Batch size to use when word_indices is None (must match the caller's real
                batch size so downstream concatenation with per-sample features doesn't break)
        Returns:
            language_features: Language model features (batch_size, seq_len, hidden_dim * 2)
            word_probs: Word probabilities (batch_size, seq_len, vocab_size)
        """
        if word_indices is not None:
            batch_size = word_indices.size(0)
        elif batch_size is None:
            batch_size = 1

        if word_indices is None:
            # Generate dummy sequence for feature extraction
            word_indices = torch.zeros(batch_size, sequence_length, dtype=torch.long)
            word_indices = word_indices.to(next(self.parameters()).device)

        seq_len = word_indices.size(1)

        # Word embeddings
        word_embeds = self.word_embedding(word_indices)

        # Add positional information based on GRID structure
        positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        positions = positions.to(word_indices.device)
        # Map to GRID positions (cycle through 6 positions)
        grid_positions = positions % 6
        pos_embeds = self.position_embedding(grid_positions)

        # Combine embeddings
        combined_embeds = word_embeds + pos_embeds
        combined_embeds = self.dropout(combined_embeds)

        # LSTM forward pass
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_out = self.dropout(lstm_out)

        # Apply grammar-aware attention
        attended_out, attention_weights = self.grammar_attention(lstm_out, lstm_out, lstm_out)

        # Combine LSTM output with attended output
        language_features = lstm_out + attended_out

        # Generate word probabilities
        word_probs = self.output_projection(language_features)

        return language_features, word_probs

    def get_grammar_constraints(self, position):
        """Get valid words for given position in GRID structure"""
        grid_position = position % 6
        if grid_position in self.type_to_words:
            valid_words = self.type_to_words[grid_position]
            valid_indices = [Config.VOCABULARY.index(word) for word in valid_words
                             if word in Config.VOCABULARY]
            return valid_indices
        return list(range(self.vocab_size))

    def apply_grammar_constraints(self, word_probs, positions):
        """Apply grammar constraints to word probabilities"""
        batch_size, seq_len, vocab_size = word_probs.shape
        constrained_probs = torch.full_like(word_probs, -float('inf'))

        for i in range(seq_len):
            valid_indices = self.get_grammar_constraints(i)
            for batch_idx in range(batch_size):
                constrained_probs[batch_idx, i, valid_indices] = word_probs[batch_idx, i, valid_indices]

        return F.softmax(constrained_probs, dim=-1)


class PhonemeLanguageModel(nn.Module):
    """Language model for phoneme sequences"""

    def __init__(self,
                 num_phonemes=len(Config.PHONEMES),
                 embed_dim=Config.HIDDEN_DIM,
                 hidden_dim=Config.HIDDEN_DIM):
        super(PhonemeLanguageModel, self).__init__()

        self.num_phonemes = num_phonemes

        # Phoneme embeddings
        self.phoneme_embedding = nn.Embedding(num_phonemes, embed_dim)

        # Transformer for phoneme sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Output projection
        self.output_projection = nn.Linear(embed_dim, num_phonemes)

    def forward(self, phoneme_indices=None, sequence_length=Config.SEQUENCE_LENGTH, batch_size=None):
        """
        Args:
            phoneme_indices: Phoneme indices (batch_size, seq_len) - optional for unconditional generation.
                Must never be the ground-truth phoneme sequence being predicted elsewhere in the
                pipeline, since embedding it here would let the model trivially echo the answer.
            sequence_length: Length of sequence to generate features for, used when phoneme_indices is None
            batch_size: Batch size to use when phoneme_indices is None (must match the caller's real
                batch size so downstream concatenation with per-sample features doesn't break)
        Returns:
            phoneme_features: Features (batch_size, seq_len, embed_dim)
            phoneme_probs: Probabilities (batch_size, seq_len, num_phonemes)
        """
        if phoneme_indices is None:
            # Generate dummy sequence for feature extraction (mirrors GRIDLanguageModel's
            # unconditional path) so this branch never depends on the target it's used to predict.
            if batch_size is None:
                batch_size = 1
            phoneme_indices = torch.zeros(batch_size, sequence_length, dtype=torch.long)
            phoneme_indices = phoneme_indices.to(next(self.parameters()).device)

        # Phoneme embeddings
        phoneme_embeds = self.phoneme_embedding(phoneme_indices)

        # Transformer forward
        phoneme_features = self.transformer(phoneme_embeds)

        # Generate phoneme probabilities
        phoneme_probs = self.output_projection(phoneme_features)

        return phoneme_features, phoneme_probs


class HierarchicalLanguageModel(nn.Module):
    def __init__(self, vocab_size, phoneme_vocab_size):
        super(HierarchicalLanguageModel, self).__init__()

        # Use your existing models
        self.word_lm = GRIDLanguageModel(vocab_size=vocab_size)
        self.phoneme_lm = PhonemeLanguageModel(num_phonemes=phoneme_vocab_size)

        # Projection for matching dimensions
        self.phoneme_proj = nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM * 2)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.HIDDEN_DIM * 2,
            num_heads=Config.NUM_HEADS,
            batch_first=True
        )

        # src/language_model.py  (inside HierarchicalLanguageModel.__init__)
        self.fc_out = nn.Linear(Config.HIDDEN_DIM * 2, vocab_size)  # was Config.HIDDEN_DIM

    def forward(self, word_indices, phoneme_indices, sequence_length=Config.SEQUENCE_LENGTH, batch_size=None):
        # word_features: (B, Lw, Ew)
        # phoneme_features: (B, Lp, Ep)
        word_features, word_probs = self.word_lm(word_indices, sequence_length=sequence_length, batch_size=batch_size)  # expected Ew == Config.HIDDEN_DIM*2
        phoneme_features, phoneme_probs = self.phoneme_lm(phoneme_indices, sequence_length=sequence_length, batch_size=batch_size)  # expected Ep == Config.HIDDEN_DI

        # project phoneme features up to match word feature dim
        phoneme_features = self.phoneme_proj(phoneme_features)
        # now phoneme_features dim == Config.HIDDEN_DIM*2
        # Ensure batch sizes match by expanding if necessary
        if word_features.size(0) != phoneme_features.size(0):
            if word_features.size(0) == 1:
                word_features = word_features.expand(phoneme_features.size(0), -1, -1)
            elif phoneme_features.size(0) == 1:
                phoneme_features = phoneme_features.expand(word_features.size(0), -1, -1)
            else:
                raise RuntimeError(
                    f"Batch size mismatch: words {word_features.size()}, phonemes {phoneme_features.size()}")

        # align both time lengths to the same target
        target_len = max(word_features.size(1), phoneme_features.size(1))
        word_features = time_match(word_features, target_len)  # (B, target_len, E_word)
        phoneme_features = time_match(phoneme_features, target_len)  # (B, target_len, E_phoneme_proj)

        # make tensors contiguous and float32
        word_features = word_features.contiguous().float()
        phoneme_features = phoneme_features.contiguous().float()

        # 🔍 Debug shapes after alignment

        # Ensure attention embed_dim matches actual embedding size
        attn_embed_dim = self.cross_attention.embed_dim if hasattr(self.cross_attention, "embed_dim") else None
        actual_dim = word_features.size(-1)
        if attn_embed_dim is None:
            attn_embed_dim = getattr(self.cross_attention, "d_model", None)

        if attn_embed_dim != actual_dim:
            print(f"[HierLM] ⚠️ Attention mismatch: attn_embed_dim={attn_embed_dim}, actual_dim={actual_dim}")
            if attn_embed_dim is not None:
                proj_qk = nn.Linear(actual_dim, attn_embed_dim).to(word_features.device)
                word_features = proj_qk(word_features)
                proj_v = nn.Linear(actual_dim, attn_embed_dim).to(phoneme_features.device)
                phoneme_features = proj_v(phoneme_features)
                actual_dim = attn_embed_dim
                print(f"[HierLM] projected both features to {attn_embed_dim}")
            else:
                raise RuntimeError(f"Attention module embed dim unknown and actual_dim={actual_dim}")

        # Check divisibility by num_heads
        num_heads = self.cross_attention.num_heads if hasattr(self.cross_attention, "num_heads") else getattr(
            self.cross_attention, "nhead", None)
        assert num_heads is not None, "Cannot determine attention num_heads"
        assert actual_dim % num_heads == 0, f"embed_dim {actual_dim} not divisible by num_heads {num_heads}"

        # 🔍 Debug before attention
        print(
            f"[HierLM] feeding into cross_attention: query={phoneme_features.shape}, key={word_features.shape}, value={word_features.shape}")

        # Now run attention. MultiheadAttention expects (batch, seq, embed) when batch_first=True
        attended_phonemes, attn_weights = self.cross_attention(
            phoneme_features,  # query
            word_features,  # key
            word_features  # value
        )

        combined_features = attended_phonemes + word_features
        output_logits = self.fc_out(combined_features)

        # 🔍 Final debug
        print(f"[HierLM] combined_features: {combined_features.shape}, output_logits: {output_logits.shape}")

        word_probs = F.softmax(output_logits, dim=-1)
        return combined_features, word_probs


import torch
import torch.nn as nn

class ContextualLanguageModel(nn.Module):
    """Hierarchical LM with speaker adaptation"""

    def __init__(self, num_speakers=3, vocab_size=len(Config.VOCABULARY)):

        super().__init__()
        self.base_lm = HierarchicalLanguageModel(
            vocab_size=Config.VOCAB_SIZE,
            phoneme_vocab_size=Config.PHONEME_VOCAB_SIZE
        )

        # Speaker embeddings
        # src/language_model.py  (inside ContextualLanguageModel.__init__)
        self.speaker_embedding = nn.Embedding(num_speakers, Config.HIDDEN_DIM * 2)  # was HIDDEN_DIM

        # Context adaptation
        # src/language_model.py  (inside ContextualLanguageModel.__init__)
        self.context_adaptation = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 4, Config.HIDDEN_DIM * 2),  # 1024 -> 512
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM * 2)  # 512 -> 512
        )

        # Final classifier after adaptation
        # src/language_model.py  (inside ContextualLanguageModel.__init__)
        self.classifier = nn.Linear(Config.HIDDEN_DIM * 2, vocab_size)  # was HIDDEN_DIM

    def forward(self, word_indices=None, phoneme_indices=None,
                speaker_ids=None, sequence_length=Config.SEQUENCE_LENGTH):

        # Real batch size comes from speaker_ids when word/phoneme indices aren't provided
        # (the normal case now that this branch never sees the ground-truth phoneme target) —
        # otherwise the dummy features default to batch_size=1 and break concatenation below.
        batch_size = speaker_ids.size(0) if speaker_ids is not None else None

        # Base hierarchical LM
        base_features, _ = self.base_lm(word_indices, phoneme_indices, sequence_length=sequence_length, batch_size=batch_size)

        if speaker_ids is not None:
            # Speaker embeddings
            speaker_embeds = self.speaker_embedding(speaker_ids)  # (B, H)

            # Broadcast across sequence
            batch_size, seq_len, feat_dim = base_features.shape
            speaker_embeds = speaker_embeds.unsqueeze(1).repeat(1, seq_len, 1)

            # Concatenate and adapt
            combined = torch.cat([base_features, speaker_embeds], dim=-1)  # (B,L,1024)
            adapted_features = self.context_adaptation(combined)  # (B,L,512)


        else:
            adapted_features = base_features

        # Compute new word probabilities from adapted features
        word_probs = self.classifier(adapted_features)  # (B,L,V)

        return adapted_features, word_probs
