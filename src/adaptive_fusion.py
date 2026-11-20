import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from src.viseme_mapping import viseme_mapper


class AdaptiveFusionModule(nn.Module):
    def __init__(self,
                 visual_dim=Config.VISUAL_FEATURE_DIM,
                 language_dim=Config.HIDDEN_DIM * 2,
                 fusion_dim=Config.HIDDEN_DIM,
                 num_visemes=12):
        super(AdaptiveFusionModule, self).__init__()

        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.fusion_dim = fusion_dim
        self.num_visemes = num_visemes

        # Visual feature projection
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)

        # Language feature projection
        self.language_projection = nn.Linear(language_dim, fusion_dim)

        # Ambiguity-aware gating network
        self.ambiguity_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # ✅ Output projection (missing in your version)
        self.output_projection = nn.Linear(fusion_dim, num_visemes)

        # Attention mechanism for visual features
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=Config.DROPOUT_RATE,
            batch_first=True
        )

    def forward(self, visual_features, language_features, ambiguity_scores):
        batch_size, seq_len, _ = visual_features.shape

        # Project features
        visual_proj = self.visual_projection(visual_features)
        language_proj = self.language_projection(language_features)

        # Apply attention to visual features
        visual_attended, _ = self.visual_attention(visual_proj, visual_proj, visual_proj)

        # Adaptive fusion weights
        ambiguity_expanded = ambiguity_scores.unsqueeze(-1)  # (B, T, 1)
        fusion_weights = self.ambiguity_gate(ambiguity_expanded)  # (B, T, 2)

        visual_weights = fusion_weights[:, :, 0:1]
        language_weights = fusion_weights[:, :, 1:2]

        # Weighted sum
        weighted_visual = visual_attended * visual_weights
        weighted_language = language_proj * language_weights

        # Fuse
        concatenated = torch.cat([weighted_visual, weighted_language], dim=-1)
        fused_features = self.fusion_network(concatenated)

        # ✅ Final output (viseme logits)
        output = self.output_projection(fused_features)

        return output, fusion_weights


    def get_fusion_statistics(self, ambiguity_scores):
        """Get statistics about fusion weight distribution"""
        ambiguity_expanded = ambiguity_scores.unsqueeze(-1)
        fusion_weights = self.ambiguity_gate(ambiguity_expanded)

        visual_weights = fusion_weights[:, :, 0]
        language_weights = fusion_weights[:, :, 1]

        stats = {
            'mean_visual_weight': torch.mean(visual_weights).item(),
            'mean_language_weight': torch.mean(language_weights).item(),
            'std_visual_weight': torch.std(visual_weights).item(),
            'std_language_weight': torch.std(language_weights).item(),
        }

        return stats


class ConfusionMatrixDecoder(nn.Module):
    """Decoder that uses confusion matrix for phoneme disambiguation"""

    # Probability floor for phonemes outside a viseme's class. Keeps the prior's log finite (so
    # the learnable adjustment can still move mass there) while leaving ~95%+ of each row's mass
    # on the viseme's own phonemes at initialisation.
    PRIOR_FLOOR = 1e-3

    def __init__(self, num_visemes=12, num_phonemes=len(Config.PHONEMES)):
        super(ConfusionMatrixDecoder, self).__init__()

        # Prior P(phoneme | viseme), columns in Config.PHONEMES order (= phoneme target indices)
        confusion_matrix, phoneme_to_idx = viseme_mapper.confusion_matrix
        assert confusion_matrix.shape == (num_visemes, num_phonemes), \
            f"confusion prior {confusion_matrix.shape} != ({num_visemes}, {num_phonemes})"
        self.num_visemes = num_visemes
        self.num_phonemes = num_phonemes
        self.register_buffer('confusion_matrix', torch.FloatTensor(confusion_matrix))
        self.phoneme_to_idx = phoneme_to_idx

        # Learnable adjustment, applied in log space: softmax(log prior + adjustment).
        # (Previously softmax was applied directly to the 0..1 probabilities, which flattened the
        # prior to almost uniform — e.g. viseme 2's 12 phonemes kept ~24% of the mass vs 22% uniform.)
        self.confusion_adjustment = nn.Parameter(torch.zeros_like(self.confusion_matrix))

        # Sequence model over the phoneme distribution
        self.phoneme_lstm = nn.LSTM(
            input_size=num_phonemes,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final prediction layer: a learned correction on top of the prior (see forward).
        # Zero-initialised so that at the start of training the output is exactly the prior.
        self.phoneme_predictor = nn.Linear(Config.HIDDEN_DIM * 2, num_phonemes)
        nn.init.zeros_(self.phoneme_predictor.weight)
        nn.init.zeros_(self.phoneme_predictor.bias)

    def forward(self, viseme_logits):
        """
        Args:
            viseme_logits: Viseme logits (batch_size, seq_len, num_visemes)
        Returns:
            phoneme_logits: (batch_size, seq_len, num_phonemes)
        """
        viseme_probs = F.softmax(viseme_logits, dim=-1)

        # Viseme probabilities -> phoneme probabilities via the (adjusted) prior
        # viseme_probs: (B, T, V), confusion: (V, P) -> (B, T, P)
        phoneme_probs = torch.matmul(viseme_probs, self.get_confusion_matrix())

        # Sequence context to disambiguate phonemes within a viseme class, as a residual
        # correction to the prior's log-probabilities (so the prior is actually used for the
        # prediction rather than only shaping the LSTM input).
        lstm_out, _ = self.phoneme_lstm(phoneme_probs)
        return torch.log(phoneme_probs + 1e-8) + self.phoneme_predictor(lstm_out)

    def get_confusion_matrix(self):
        """Current P(phoneme | viseme) including the learned adjustment, rows sum to 1"""
        log_prior = torch.log(self.confusion_matrix.clamp_min(self.PRIOR_FLOOR))
        return F.softmax(log_prior + self.confusion_adjustment, dim=-1)


class MultiScaleAdaptiveFusion(nn.Module):
    """Multi-scale adaptive fusion with temporal modeling"""

    def __init__(self,
                 visual_dim=Config.VISUAL_FEATURE_DIM,
                 language_dim=Config.HIDDEN_DIM,
                 num_visemes=12):
        super(MultiScaleAdaptiveFusion, self).__init__()

        # Different time scales
        self.scales = [1, 3, 5, 7]  # Different temporal window sizes

        # Fusion modules for each scale
        self.fusion_modules = nn.ModuleList([
            AdaptiveFusionModule(visual_dim, Config.HIDDEN_DIM * 2, Config.HIDDEN_DIM, num_visemes)
            for _ in self.scales
        ])

        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(len(self.scales) * num_visemes, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, len(self.scales)),
            nn.Softmax(dim=-1)
        )




    def temporal_pooling(self, features, scale):
        """Apply temporal pooling with given scale"""
        if scale == 1:
            return features

        # Apply average pooling across time dimension
        kernel_size = min(scale, features.size(1))
        pooled = F.avg_pool1d(
            features.permute(0, 2, 1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        return pooled.permute(0, 2, 1)

    def forward(self, visual_features, language_features, ambiguity_scores):
        """
        Multi-scale fusion forward pass
        """
        scale_outputs = []
        scale_weights_list = []

        # Process each scale
        for i, scale in enumerate(self.scales):
            # Apply temporal pooling
            visual_scaled = self.temporal_pooling(visual_features, scale)
            language_scaled = self.temporal_pooling(language_features, scale)
            ambiguity_scaled = self.temporal_pooling(ambiguity_scores.unsqueeze(-1), scale).squeeze(-1)

            # Apply fusion
            output, weights = self.fusion_modules[i](visual_scaled, language_scaled, ambiguity_scaled)
            scale_outputs.append(output)
            scale_weights_list.append(weights)

        # Scale attention over the concatenated per-scale outputs
        concat_scales = torch.cat(scale_outputs, dim=-1)  # (B, T, num_visemes * num_scales)
        scale_attention_weights = self.scale_attention(concat_scales)  # (B, T, num_scales)
        scale_attention_weights = scale_attention_weights.unsqueeze(-1)  # (B, T, num_scales, 1)

        final_output = torch.zeros_like(scale_outputs[0])
        for i, output in enumerate(scale_outputs):
            weight = scale_attention_weights[:, :, i]  # (B, T, 1)
            final_output = final_output + output * weight

        return final_output, {
            'scale_outputs': scale_outputs,
            'scale_attention_weights': scale_attention_weights,
            'fusion_weights': scale_weights_list
        }
