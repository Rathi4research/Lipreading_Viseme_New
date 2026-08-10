import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from models.visual_encoder import EnhancedVisualEncoder
from models.sequence_model import HybridSequenceModel
from src.language_model import ContextualLanguageModel
from src.adaptive_fusion import MultiScaleAdaptiveFusion, ConfusionMatrixDecoder
from src.viseme_mapping import viseme_mapper

class VisemePhonemeAdaptiveModel(nn.Module):
    """
    Enhanced Viseme-Phoneme Adaptive Language Model for Automated Lip Reading
    """

    def __init__(self, num_visemes=12, num_phonemes=len(Config.PHONEMES)):
        super(VisemePhonemeAdaptiveModel, self).__init__()

        self.num_visemes = num_visemes
        self.num_phonemes = num_phonemes

        # Visual processing pipeline
        self.visual_encoder = EnhancedVisualEncoder()

        # Sequence modeling
        self.sequence_model = HybridSequenceModel(
            input_dim=Config.VISUAL_FEATURE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_visemes=num_visemes
        )

        # Language model
        self.language_model = ContextualLanguageModel(num_speakers=Config.NUM_SPEAKERS)

        # Adaptive fusion
        self.adaptive_fusion = MultiScaleAdaptiveFusion(
            visual_dim=Config.HIDDEN_DIM,  # From sequence model
            language_dim=Config.HIDDEN_DIM,
            num_visemes=num_visemes
        )


        # Change to:
        self.fused_proj = nn.Linear(num_visemes, Config.HIDDEN_DIM)  # 12 -> 256

        # Confusion matrix decoder for phoneme disambiguation
        self.confusion_decoder = ConfusionMatrixDecoder(
            num_visemes=num_visemes,
            num_phonemes=num_phonemes
        )

        # Final prediction layers
        self.viseme_classifier = nn.Linear(Config.HIDDEN_DIM, num_visemes)
        self.num_phonemes = self.confusion_decoder.num_phonemes
        self.phoneme_classifier = nn.Linear(self.num_phonemes, self.num_phonemes)

        # Loss weights
        self.register_buffer('viseme_loss_weight', torch.tensor(1.0))
        self.register_buffer('phoneme_loss_weight', torch.tensor(1.0))
        self.register_buffer('language_loss_weight', torch.tensor(0.5))

    def forward(self, frames, phoneme_targets=None, word_targets=None,
                speaker_ids=None, ambiguity_scores=None, model_type='full'):

        batch_size, seq_len, height, width = frames.shape

        # 1. Visual feature extraction
        visual_features = self.visual_encoder(frames)

        # 2. Visual sequence modeling
        sequence_output, sequence_hidden = self.sequence_model(visual_features)

        language_features = None
        word_probs = None

        if model_type == 'full':
            # 3. Language model features only in full mode.
            # phoneme_targets/word_targets are the ground truth being predicted downstream
            # (phoneme_logits, word_logits) — feeding them in here would let the model
            # trivially echo the answer instead of learning from vision, so the language
            # model always runs unconditionally (dummy phoneme/word sequences) and only
            # contributes structural/grammar priors, not the actual target sequence.
            language_features, word_probs = self.language_model(
                word_indices=None,
                phoneme_indices=None,
                speaker_ids=speaker_ids,
                sequence_length=seq_len
            )

        # 4. Ambiguity scores
        if ambiguity_scores is None:
            predicted_visemes = torch.argmax(sequence_output, dim=-1)
            ambiguity_scores = self._calculate_ambiguity_scores(predicted_visemes)

        # 5. Adaptive fusion
        if language_features is not None:
            fused_output, fusion_info = self.adaptive_fusion(
                sequence_hidden, language_features, ambiguity_scores
            )
        else:
            fused_output = sequence_hidden
            fusion_info = {'fusion_weights': None, 'scale_attention_weights': None}

        # fused_proj maps AdaptiveFusionModule's num_visemes-dim output up to HIDDEN_DIM.
        # When fusion is skipped (language_features is None, e.g. model_type='visual_only'
        # or the baseline variant's nn.Identity), fused_output is sequence_hidden, which is
        # already HIDDEN_DIM-dim — applying fused_proj to it would crash on a shape mismatch.
        if fused_output.size(-1) == self.num_visemes:
            fused_output_proj = self.fused_proj(fused_output)  # 12 -> 256
        else:
            fused_output_proj = fused_output

        viseme_logits = self.viseme_classifier(fused_output_proj)
        phoneme_logits = self.confusion_decoder(viseme_logits)  # -> (B, T, num_phonemes)

        return {
            'viseme_logits': viseme_logits,
            'phoneme_logits': phoneme_logits,
            'word_logits': word_probs,
            'visual_features': visual_features,
            'sequence_hidden': sequence_hidden,
            'language_features': language_features,
            'fusion_weights': fusion_info.get('fusion_weights', None),
            'scale_attention': fusion_info.get('scale_attention_weights', None),
            'ambiguity_scores': ambiguity_scores,
            'confusion_matrix': self.confusion_decoder.get_confusion_matrix()
        }

    def _calculate_ambiguity_scores(self, viseme_sequence):
        """Calculate ambiguity scores for viseme sequence"""


        batch_size, seq_len = viseme_sequence.shape
        ambiguity_scores = torch.zeros_like(viseme_sequence, dtype=torch.float)

        for b in range(batch_size):
            for t in range(seq_len):
                viseme_id = viseme_sequence[b, t].item()
                score = viseme_mapper.get_ambiguity_score(viseme_id)
                ambiguity_scores[b, t] = score

        return ambiguity_scores

    def compute_loss(self, predictions, targets):
        """
        Compute multi-task loss

        Args:
            predictions: Model predictions dictionary
            targets: Target dictionary with 'visemes', 'phonemes', 'words'

        Returns:
            Total loss and loss components
        """

        losses = {}

        # Viseme classification loss
        if 'visemes' in targets:
            viseme_loss = F.cross_entropy(
                predictions['viseme_logits'].reshape(-1, self.num_visemes),
                targets['visemes'].reshape(-1),
                ignore_index=-1
            )
            losses['viseme_loss'] = viseme_loss

        # Phoneme classification loss
        if 'phonemes' in targets:
            logits = predictions['phoneme_logits']
            target = targets['phonemes']

            # Align shapes if mismatch
            if logits.size(-1) != self.num_phonemes:
                print(f"Adjusting self.num_phonemes from {self.num_phonemes} to {logits.size(-1)}")
                self.num_phonemes = logits.size(-1)

            phoneme_loss = F.cross_entropy(
                logits.reshape(-1, self.num_phonemes),
                target.reshape(-1),
                ignore_index=-1
            )
            losses['phoneme_loss'] = phoneme_loss

        # Word-level language model loss
        if 'words' in targets and predictions['word_logits'] is not None:
            word_loss = F.cross_entropy(
                predictions['word_logits'].reshape(-1, len(Config.VOCABULARY)),
                targets['words'].reshape(-1),
                ignore_index=-1
            )
            losses['word_loss'] = word_loss

        # Adaptive fusion regularization
        if predictions['fusion_weights'] is not None:
            fusion_weights_list = predictions['fusion_weights']
            fusion_reg_total = 0.0
            for fw in fusion_weights_list:
                # fw: (B, T, 2)
                fusion_reg_total += -torch.mean(torch.sum(fw * torch.log(fw + 1e-8), dim=-1))
            fusion_reg_total /= len(fusion_weights_list)
            losses['fusion_reg'] = 0.01 * fusion_reg_total

        # Total loss
        total_loss = 0
        if 'viseme_loss' in losses:
            total_loss += self.viseme_loss_weight * losses['viseme_loss']
        if 'phoneme_loss' in losses:
            total_loss += self.phoneme_loss_weight * losses['phoneme_loss']
        if 'word_loss' in losses:
            total_loss += self.language_loss_weight * losses['word_loss']
        if 'fusion_reg' in losses:
            total_loss += losses['fusion_reg']

        losses['total_loss'] = total_loss

        return losses

    def predict(self, frames, speaker_ids=None, model_type='full'):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(frames, speaker_ids=speaker_ids, model_type=model_type)

            viseme_preds = torch.argmax(predictions['viseme_logits'], dim=-1)
            phoneme_preds = torch.argmax(predictions['phoneme_logits'], dim=-1)
            word_preds = (torch.argmax(predictions['word_logits'], dim=-1)
                          if predictions['word_logits'] is not None else None)

            return {
                'viseme_predictions': viseme_preds,
                'phoneme_predictions': phoneme_preds,
                'word_predictions': word_preds,
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
    def create_baseline_model():
        """Create baseline model without adaptive fusion"""
        model = VisemePhonemeAdaptiveModel()
        # Disable adaptive components for baseline
        model.adaptive_fusion = nn.Identity()
        return model

    @staticmethod
    def create_visual_only_model():
        """Create visual-only model"""
        model = VisemePhonemeAdaptiveModel()
        # Zero out language model influence
        model.language_loss_weight = torch.tensor(0.0)
        return model

    @staticmethod
    def create_full_model():
        """Create full adaptive model"""
        return VisemePhonemeAdaptiveModel()

    @staticmethod
    def load_pretrained_model(checkpoint_path, model_type='full'):
        """Load pretrained model from checkpoint"""
        if model_type == 'baseline':
            model = ModelFactory.create_baseline_model()
        elif model_type == 'visual_only':
            model = ModelFactory.create_visual_only_model()
        else:
            model = ModelFactory.create_full_model()

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Filter out keys not in model or whose shapes do not match
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

        # Load compatible weights only
        model.load_state_dict(filtered_dict, strict=False)

        return model
