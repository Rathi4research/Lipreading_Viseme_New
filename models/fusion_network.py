import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from models.visual_encoder import EnhancedVisualEncoder, upgrade_legacy_state_dict
from models.sequence_model import HybridSequenceModel
from src.language_model import ContextualLanguageModel
from src.adaptive_fusion import MultiScaleAdaptiveFusion, ConfusionMatrixDecoder
from src.viseme_mapping import viseme_mapper

MODEL_TYPES = ('full', 'baseline', 'visual_only')


class VisemePhonemeAdaptiveModel(nn.Module):
    """
    Enhanced Viseme-Phoneme Adaptive Language Model for Automated Lip Reading

    model_type selects the variant and is used for both training and evaluation:
      - 'full':        visual + language features combined by ambiguity-gated multi-scale fusion
      - 'baseline':    visual + language features combined by a plain (non-adaptive) linear fusion
      - 'visual_only': no language model; visual sequence features go straight to the classifier

    Nothing derived from the targets is fed into the network. The per-frame ambiguity scores that
    drive the fusion gate, and the soft phoneme sequence fed to the language model, both come from
    the sequence model's own (auxiliary) viseme predictions, which are trained with an extra loss.
    """

    def __init__(self, num_visemes=12, num_phonemes=len(Config.PHONEMES), model_type='full'):
        super(VisemePhonemeAdaptiveModel, self).__init__()

        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
        self.model_type = model_type
        # Speakers seen in training (set by the trainer, stored in checkpoints). Any other speaker
        # is mapped to the language model's "unknown speaker" embedding.
        self.known_speakers = []

        self.num_visemes = num_visemes
        self.num_phonemes = num_phonemes

        # Visual processing pipeline
        self.visual_encoder = EnhancedVisualEncoder()

        # Sequence modeling (its viseme head is the auxiliary prediction used for ambiguity/LM input)
        self.sequence_model = HybridSequenceModel(
            input_dim=Config.VISUAL_FEATURE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_visemes=num_visemes
        )

        # Language model
        self.language_model = ContextualLanguageModel(num_speakers=Config.NUM_SPEAKERS)

        # Adaptive fusion ('full')
        self.adaptive_fusion = MultiScaleAdaptiveFusion(
            visual_dim=Config.HIDDEN_DIM,  # From sequence model
            language_dim=Config.HIDDEN_DIM,
            num_visemes=num_visemes
        )
        # Maps the fusion's num_visemes-dim output up to HIDDEN_DIM
        self.fused_proj = nn.Linear(num_visemes, Config.HIDDEN_DIM)

        # Plain fusion ('baseline'): concat(visual 256, language 512) -> 256
        self.static_fusion = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM * 3, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Confusion matrix decoder for phoneme disambiguation
        self.confusion_decoder = ConfusionMatrixDecoder(
            num_visemes=num_visemes,
            num_phonemes=num_phonemes
        )

        # Final viseme classifier
        self.viseme_classifier = nn.Linear(Config.HIDDEN_DIM, num_visemes)

        # Ambiguity score per viseme id (for vectorized lookup)
        self.register_buffer('ambiguity_table', torch.tensor(viseme_mapper.ambiguity_lookup(), dtype=torch.float))

        # Loss weights
        self.register_buffer('viseme_loss_weight', torch.tensor(1.0))
        self.register_buffer('phoneme_loss_weight', torch.tensor(1.0))
        self.register_buffer('language_loss_weight', torch.tensor(0.0 if model_type == 'visual_only' else 0.5))
        self.register_buffer('aux_viseme_loss_weight', torch.tensor(Config.AUX_VISEME_LOSS_WEIGHT))

    # ------------------------------------------------------------------ helpers
    def speaker_ids_for(self, subjects, device=None):
        """Speaker-embedding ids for a list of subject names ('s1', ...).

        Speakers not seen in training map to the 'unknown speaker' slot. If known_speakers is
        empty (e.g. a model built outside the trainer), every GRID speaker is treated as known.
        """
        ids = []
        for s in subjects:
            if s in Config.SPEAKER_TO_ID and (not self.known_speakers or s in self.known_speakers):
                ids.append(Config.SPEAKER_TO_ID[s])
            else:
                ids.append(Config.UNKNOWN_SPEAKER_ID)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _calculate_ambiguity_scores(self, viseme_sequence):
        """Ambiguity score per position of a (B, T) viseme-id tensor (-1 / padding -> 0)"""
        valid = viseme_sequence >= 0
        scores = self.ambiguity_table[viseme_sequence.clamp_min(0)]
        return scores * valid

    # ------------------------------------------------------------------ forward
    def forward(self, frames, phoneme_targets=None, word_targets=None,
                speaker_ids=None, ambiguity_scores=None, model_type=None):
        """
        Args:
            frames: (B, T, H, W) or (B, T, 1, H, W)
            phoneme_targets, word_targets: accepted for backward compatibility and ignored —
                targets must never be model inputs.
            speaker_ids: (B,) speaker-embedding ids (see speaker_ids_for)
            ambiguity_scores: optional (B, T) override for analysis/ablation. By default they are
                computed from the model's own auxiliary viseme predictions. Do not pass scores
                derived from the ground-truth visemes when measuring accuracy (label leak).
            model_type: override self.model_type (normally leave as None)
        """
        model_type = model_type or self.model_type
        seq_len = frames.shape[1]

        # 1. Visual feature extraction
        visual_features = self.visual_encoder(frames)

        # 2. Visual sequence modeling (+ auxiliary viseme prediction)
        aux_viseme_logits, sequence_hidden = self.sequence_model(visual_features)
        aux_viseme_probs = F.softmax(aux_viseme_logits, dim=-1).detach()

        # 3. Ambiguity scores: expected ambiguity under the model's own viseme distribution
        if ambiguity_scores is None:
            ambiguity_scores = torch.matmul(aux_viseme_probs, self.ambiguity_table)

        # 4. Language model on the model's own soft phoneme estimate (viseme probs x prior)
        language_features = None
        word_logits = None
        if model_type in ('full', 'baseline'):
            soft_phonemes = torch.matmul(aux_viseme_probs, self.confusion_decoder.confusion_matrix)
            language_features, word_logits = self.language_model(
                phoneme_probs=soft_phonemes,
                speaker_ids=speaker_ids,
                sequence_length=seq_len
            )

        # 5. Fusion
        fusion_info = {'fusion_weights': None, 'scale_attention_weights': None}
        if model_type == 'full':
            fused_output, fusion_info = self.adaptive_fusion(
                sequence_hidden, language_features, ambiguity_scores
            )
            fused_hidden = self.fused_proj(fused_output)  # num_visemes -> HIDDEN_DIM
        elif model_type == 'baseline':
            fused_hidden = self.static_fusion(torch.cat([sequence_hidden, language_features], dim=-1))
        else:  # visual_only
            fused_hidden = sequence_hidden

        # 6. Viseme classification and viseme -> phoneme decoding
        viseme_logits = self.viseme_classifier(fused_hidden)
        phoneme_logits = self.confusion_decoder(viseme_logits)  # -> (B, T, num_phonemes)

        return {
            'viseme_logits': viseme_logits,
            'phoneme_logits': phoneme_logits,
            'aux_viseme_logits': aux_viseme_logits,
            'word_logits': word_logits,
            'visual_features': visual_features,
            'sequence_hidden': sequence_hidden,
            'language_features': language_features,
            'fusion_weights': fusion_info.get('fusion_weights', None),
            'scale_attention': fusion_info.get('scale_attention_weights', None),
            'ambiguity_scores': ambiguity_scores,
            'confusion_matrix': self.confusion_decoder.get_confusion_matrix()
        }

    def compute_loss(self, predictions, targets):
        """
        Multi-task loss.

        targets: dict with per-frame 'visemes', 'phonemes' and optionally 'words' (word-vocabulary
        index per frame); -1 marks padding frames and is ignored.
        """
        losses = {}

        if 'visemes' in targets:
            visemes = targets['visemes'].reshape(-1)
            losses['viseme_loss'] = F.cross_entropy(
                predictions['viseme_logits'].reshape(-1, self.num_visemes), visemes, ignore_index=-1)
            losses['aux_viseme_loss'] = F.cross_entropy(
                predictions['aux_viseme_logits'].reshape(-1, self.num_visemes), visemes, ignore_index=-1)

        if 'phonemes' in targets:
            losses['phoneme_loss'] = F.cross_entropy(
                predictions['phoneme_logits'].reshape(-1, self.num_phonemes),
                targets['phonemes'].reshape(-1), ignore_index=-1)

        if targets.get('words') is not None and predictions['word_logits'] is not None:
            losses['word_loss'] = F.cross_entropy(
                predictions['word_logits'].reshape(-1, len(Config.VOCABULARY)),
                targets['words'].reshape(-1), ignore_index=-1)

        # Adaptive fusion regularization (entropy of the gate; pushes towards decisive gating)
        if predictions['fusion_weights'] is not None:
            fusion_weights_list = predictions['fusion_weights']
            fusion_reg_total = 0.0
            for fw in fusion_weights_list:  # fw: (B, T, 2)
                fusion_reg_total += -torch.mean(torch.sum(fw * torch.log(fw + 1e-8), dim=-1))
            losses['fusion_reg'] = 0.01 * fusion_reg_total / len(fusion_weights_list)

        total_loss = 0
        if 'viseme_loss' in losses:
            total_loss = total_loss + self.viseme_loss_weight * losses['viseme_loss']
            total_loss = total_loss + self.aux_viseme_loss_weight * losses['aux_viseme_loss']
        if 'phoneme_loss' in losses:
            total_loss = total_loss + self.phoneme_loss_weight * losses['phoneme_loss']
        if 'word_loss' in losses:
            total_loss = total_loss + self.language_loss_weight * losses['word_loss']
        if 'fusion_reg' in losses:
            total_loss = total_loss + losses['fusion_reg']

        losses['total_loss'] = total_loss
        return losses

    def predict(self, frames, speaker_ids=None, model_type=None):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(frames, speaker_ids=speaker_ids, model_type=model_type)

            return {
                'viseme_predictions': torch.argmax(predictions['viseme_logits'], dim=-1),
                'phoneme_predictions': torch.argmax(predictions['phoneme_logits'], dim=-1),
                'word_predictions': (torch.argmax(predictions['word_logits'], dim=-1)
                                     if predictions['word_logits'] is not None else None),
                'viseme_probabilities': F.softmax(predictions['viseme_logits'], dim=-1),
                'phoneme_probabilities': F.softmax(predictions['phoneme_logits'], dim=-1),
                'ambiguity_scores': predictions['ambiguity_scores'],
                'fusion_weights': predictions['fusion_weights']
            }

    def get_attention_weights(self, frames, speaker_ids=None):
        """Get attention and fusion weights for visualization"""
        predictions = self.forward(frames, speaker_ids=speaker_ids)

        return {
            'fusion_weights': predictions['fusion_weights'],
            'scale_attention': predictions['scale_attention'],
            'ambiguity_scores': predictions['ambiguity_scores'],
            'confusion_matrix': predictions['confusion_matrix']
        }


class ModelFactory:
    """Factory class for creating different model configurations"""

    @staticmethod
    def create(model_type='full'):
        return VisemePhonemeAdaptiveModel(model_type=model_type)

    @staticmethod
    def create_baseline_model():
        """Language model + plain (non-adaptive) fusion"""
        return VisemePhonemeAdaptiveModel(model_type='baseline')

    @staticmethod
    def create_visual_only_model():
        """No language model"""
        return VisemePhonemeAdaptiveModel(model_type='visual_only')

    @staticmethod
    def create_full_model():
        """Full adaptive model"""
        return VisemePhonemeAdaptiveModel(model_type='full')

    @staticmethod
    def load_pretrained_model(checkpoint_path, model_type=None):
        """Load a trained model.

        The checkpoint's own model_type (saved by the trainer) is used so that evaluation runs the
        same path that was trained; an explicit, different model_type only triggers a warning.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # The visual encoder's layer sizes depend on the frame size it was trained with
        saved_config = checkpoint.get('config', {})
        saved_size = (saved_config.get('FRAME_HEIGHT'), saved_config.get('FRAME_WIDTH'))
        if None not in saved_size and saved_size != (Config.FRAME_HEIGHT, Config.FRAME_WIDTH):
            print(f"[load_pretrained_model] Using the checkpoint's frame size {saved_size[0]}x{saved_size[1]} "
                  f"(was {Config.FRAME_HEIGHT}x{Config.FRAME_WIDTH})")
            Config.FRAME_HEIGHT, Config.FRAME_WIDTH = saved_size

        saved_type = checkpoint.get('model_type')
        if saved_type and model_type and model_type != saved_type:
            print(f"[load_pretrained_model] Checkpoint was trained as '{saved_type}'; "
                  f"ignoring requested model_type '{model_type}'.")
        model = ModelFactory.create(saved_type or model_type or 'full')
        model.known_speakers = list(checkpoint.get('known_speakers', []))

        state_dict = upgrade_legacy_state_dict(checkpoint['model_state_dict'])

        # Load compatible weights only (older checkpoints may differ in architecture)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

        skipped = sorted(set(state_dict) - set(filtered_dict))
        missing = sorted(set(model_dict) - set(filtered_dict))
        if skipped or missing:
            print(f"[load_pretrained_model] WARNING: checkpoint does not match this model: "
                  f"{len(skipped)} checkpoint keys skipped (e.g. {skipped[:3]}), "
                  f"{len(missing)} model keys left at random init (e.g. {missing[:3]}). "
                  f"Results will not reflect the trained model; retrain with this code version.")

        model.load_state_dict(filtered_dict, strict=False)
        return model
