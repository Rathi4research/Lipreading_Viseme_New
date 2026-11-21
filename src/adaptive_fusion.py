"""Fusion of visual and language features, and the prior-guided phoneme head.

Fusion gates (Config.GATE / main.py --gate) - how much each frame relies on the language model:

  ambiguity            learned function of the frame's intrinsic viseme ambiguity (expected number of
                       phonemes sharing the predicted mouth shape)
  entropy              learned function of the viseme posterior's entropy (generic uncertainty baseline)
  learned              learned from the visual and language features themselves (cold-fusion style)
  hard                 no learning: language model only where intrinsic ambiguity exceeds a threshold
                       (CMTD-style switch)
  intrinsic-extrinsic  proposed: separate channels for intrinsic ambiguity (homophenes the mouth can
                       never resolve) and extrinsic unreliability (frame-quality signals: missed face
                       detection / interpolated landmarks, distance to the nearest detection, crop
                       jitter, and time-masked frames during training); their logits add up
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.viseme_mapping import viseme_mapper

GATES = ('ambiguity', 'entropy', 'learned', 'hard', 'intrinsic-extrinsic')


def _mlp(in_dim, hidden=32):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class AdaptiveFusionModule(nn.Module):
    """Gate-weighted fusion of visual and language features at one temporal scale."""

    def __init__(self, visual_dim, language_dim, fusion_dim, gate):
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        self.gate = gate
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)
        self.language_projection = nn.Linear(language_dim, fusion_dim)
        self.visual_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=Config.DROPOUT_RATE,
                                                      batch_first=True)
        if gate in ('ambiguity', 'entropy'):
            self.gate_net = _mlp(1)
        elif gate == 'learned':
            self.gate_net = _mlp(2 * fusion_dim, 64)
        elif gate == 'intrinsic-extrinsic':
            self.intrinsic_net = _mlp(1)
            self.extrinsic_net = _mlp(Config.NUM_QUALITY_FEATURES)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(fusion_dim, fusion_dim))

    def forward(self, visual_features, language_features, signals, pad_mask=None):
        """
        signals: dict with 'ambiguity' (B, T), 'entropy' (B, T), 'quality' (B, T, Q)
        Returns fused features (B, T, fusion_dim), gate weights (B, T, 2) [visual, language], extras.
        """
        v = self.visual_projection(visual_features)
        l = self.language_projection(language_features)
        v, _ = self.visual_attention(v, v, v, key_padding_mask=pad_mask, need_weights=False)

        extras = {}
        if self.gate == 'hard':
            w_lang = (signals['ambiguity'] > Config.AMBIGUITY_THRESHOLD).to(v.dtype).unsqueeze(-1)
        else:
            if self.gate == 'ambiguity':
                logit = self.gate_net(signals['ambiguity'].unsqueeze(-1))
            elif self.gate == 'entropy':
                logit = self.gate_net(signals['entropy'].unsqueeze(-1))
            elif self.gate == 'learned':
                logit = self.gate_net(torch.cat([v, l], dim=-1))
            else:  # intrinsic-extrinsic
                intrinsic = self.intrinsic_net(signals['ambiguity'].unsqueeze(-1))
                extrinsic = self.extrinsic_net(signals['quality'])
                logit = intrinsic + extrinsic
                extras = {'intrinsic_logit': intrinsic.squeeze(-1), 'extrinsic_logit': extrinsic.squeeze(-1)}
            w_lang = torch.sigmoid(logit)

        weights = torch.cat([1 - w_lang, w_lang], dim=-1)
        fused = self.fusion_network(torch.cat([v * (1 - w_lang), l * w_lang], dim=-1))
        return fused, weights, extras


class MultiScaleAdaptiveFusion(nn.Module):
    """Runs the fusion at 1/3/5/7-frame temporal scales and combines them with learned scale attention."""

    def __init__(self, visual_dim=Config.HIDDEN_DIM, language_dim=Config.HIDDEN_DIM * 2,
                 fusion_dim=Config.HIDDEN_DIM, gate=None):
        super().__init__()
        self.gate = gate or Config.GATE
        self.scales = [1, 3, 5, 7]
        self.fusion_modules = nn.ModuleList([
            AdaptiveFusionModule(visual_dim, language_dim, fusion_dim, self.gate) for _ in self.scales])
        self.scale_attention = nn.Sequential(
            nn.Linear(len(self.scales) * fusion_dim, fusion_dim), nn.ReLU(),
            nn.Linear(fusion_dim, len(self.scales)), nn.Softmax(dim=-1))

    @staticmethod
    def temporal_pooling(features, scale):
        """Centred moving average over time (B, T, D) -> (B, T, D)"""
        if scale == 1:
            return features
        k = min(scale, features.size(1))
        pooled = F.avg_pool1d(features.transpose(1, 2), kernel_size=k, stride=1, padding=k // 2,
                              count_include_pad=False)
        return pooled.transpose(1, 2)[:, :features.size(1)]

    def forward(self, visual_features, language_features, signals, pad_mask=None):
        outputs, weights, extras = [], [], []
        for scale, module in zip(self.scales, self.fusion_modules):
            pool = lambda x: self.temporal_pooling(x, scale)
            scaled = {'ambiguity': pool(signals['ambiguity'].unsqueeze(-1)).squeeze(-1),
                      'entropy': pool(signals['entropy'].unsqueeze(-1)).squeeze(-1),
                      'quality': pool(signals['quality'])}
            out, w, ex = module(pool(visual_features), pool(language_features), scaled, pad_mask)
            outputs.append(out)
            weights.append(w)
            extras.append(ex)

        attention = self.scale_attention(torch.cat(outputs, dim=-1)).unsqueeze(-1)  # (B, T, S, 1)
        fused = (torch.stack(outputs, dim=2) * attention).sum(dim=2)
        info = {'scale_outputs': outputs, 'scale_attention_weights': attention, 'fusion_weights': weights}
        if extras and extras[0]:
            info['intrinsic_logit'] = torch.stack([e['intrinsic_logit'] for e in extras]).mean(0)
            info['extrinsic_logit'] = torch.stack([e['extrinsic_logit'] for e in extras]).mean(0)
        return fused, info


class PriorGuidedPhonemeHead(nn.Module):
    """Per-frame phoneme logits from the fused features, guided by the viseme-to-phoneme prior.

    logits = direct(features) + prior_scale * log( P(viseme | frame) @ P(phoneme | viseme) )

    The direct term sees the full fused features (no 12-viseme bottleneck), so cues beyond the
    viseme class can separate phonemes; the prior term injects the phonetic knowledge of which
    phonemes share a mouth shape. P(phoneme | viseme) starts from the fixed mapping
    (src/viseme_mapping.py) and is adjusted in log space by a learnable matrix.
    """

    PRIOR_FLOOR = 1e-3

    def __init__(self, hidden_dim=Config.HIDDEN_DIM, num_visemes=12, num_phonemes=len(Config.PHONEMES)):
        super().__init__()
        confusion_matrix, phoneme_to_idx = viseme_mapper.confusion_matrix
        assert confusion_matrix.shape == (num_visemes, num_phonemes), \
            f"confusion prior {confusion_matrix.shape} != ({num_visemes}, {num_phonemes})"
        self.num_visemes = num_visemes
        self.num_phonemes = num_phonemes
        self.register_buffer('confusion_matrix', torch.FloatTensor(confusion_matrix))
        self.phoneme_to_idx = phoneme_to_idx
        self.confusion_adjustment = nn.Parameter(torch.zeros_like(self.confusion_matrix))
        self.direct = nn.Linear(hidden_dim, num_phonemes)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def get_confusion_matrix(self):
        """Current P(phoneme | viseme) including the learned adjustment, rows sum to 1"""
        log_prior = torch.log(self.confusion_matrix.clamp_min(self.PRIOR_FLOOR))
        return F.softmax(log_prior + self.confusion_adjustment, dim=-1)

    def forward(self, features, viseme_logits):
        viseme_probs = F.softmax(viseme_logits.float(), dim=-1)
        prior_mix = torch.matmul(viseme_probs, self.get_confusion_matrix())
        return self.direct(features).float() + self.prior_scale * torch.log(prior_mix + 1e-6)
