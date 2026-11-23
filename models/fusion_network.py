import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.decoding import NUM_CLASSES as CTC_CLASSES, batch_greedy_indices
from models.visual_encoder import EnhancedVisualEncoder, upgrade_legacy_state_dict
from models.sequence_model import SEQ_MODELS, build_sequence_encoder
from src.language_model import ContextualLanguageModel, DraftContextEncoder
from src.adaptive_fusion import GATES, MultiScaleAdaptiveFusion, PriorGuidedPhonemeHead
from src.viseme_mapping import viseme_mapper

MODEL_TYPES = ('full', 'baseline', 'visual_only')
LM_CONTEXTS = ('none', 'draft')
SIL_VISEME = viseme_mapper.get_viseme('sil')
SIL_PHONEME = Config.PHONEMES.index('sil')
SIL_WORD = Config.VOCABULARY.index('sil')


class VisemePhonemeAdaptiveModel(nn.Module):
    """
    Viseme-Phoneme Adaptive Model for lip reading.

    frames -> lip attention + 3D CNN -> sequence encoder (--seq-model) -> fused with a language model
    (--model-type, --gate) -> heads:
      ctc_head        sentence-level CTC over characters (main objective; decoded to text)
      viseme head     per-frame visemes (auxiliary)
      phoneme head    per-frame phonemes from the fused features + viseme-to-phoneme prior (auxiliary)
      word logits     per-frame words from the language model (auxiliary)

    model_type:
      full         visual + language features combined by the multi-scale gated fusion (--gate)
      baseline     visual + language features combined by a plain linear fusion
      visual_only  no language model

    Only frames, clip lengths, speaker ids and preprocessing frame-quality signals are inputs.
    The ambiguity score and the language model's soft-phoneme input come from the model's own
    auxiliary viseme prediction, never from labels.
    """

    def __init__(self, num_visemes=12, num_phonemes=len(Config.PHONEMES), model_type='full',
                 seq_model=None, visual_pool=None, gate=None, lm_context=None):
        super().__init__()
        seq_model = seq_model or Config.SEQ_MODEL
        visual_pool = visual_pool or Config.VISUAL_POOL
        gate = gate or Config.GATE
        lm_context = lm_context or Config.LM_CONTEXT
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
        if seq_model not in SEQ_MODELS:
            raise ValueError(f"seq_model must be one of {SEQ_MODELS}, got {seq_model!r}")
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        if lm_context not in LM_CONTEXTS:
            raise ValueError(f"lm_context must be one of {LM_CONTEXTS}, got {lm_context!r}")
        self.model_type = model_type
        self.lm_context = lm_context
        self.model_config = {'model_type': model_type, 'seq_model': seq_model,
                            'visual_pool': visual_pool, 'gate': gate, 'lm_context': lm_context}
        # Speakers seen in training (set by the trainer, stored in checkpoints); others use the
        # language model's "unknown speaker" embedding.
        self.known_speakers = []
        self.num_visemes = num_visemes
        self.num_phonemes = num_phonemes
        H = Config.HIDDEN_DIM

        self.visual_encoder = EnhancedVisualEncoder(pool=visual_pool)
        self.sequence_model = build_sequence_encoder(seq_model, Config.VISUAL_FEATURE_DIM, H)
        self.aux_viseme_head = nn.Linear(H, num_visemes)

        context_dim = 0
        if model_type in ('full', 'baseline'):
            if lm_context == 'draft':
                # Interim CTC head on the unfused sequence features: its greedy decode is the
                # self-generated sentence context the language model reads on top of the
                # per-frame phoneme estimates.
                self.draft_ctc_head = nn.Linear(H, CTC_CLASSES)
                self.context_encoder = DraftContextEncoder(out_dim=Config.HIDDEN_DIM * 2)
                context_dim = Config.HIDDEN_DIM * 2
            self.language_model = ContextualLanguageModel(num_speakers=Config.NUM_SPEAKERS,
                                                           context_dim=context_dim)
        if model_type == 'full':
            self.adaptive_fusion = MultiScaleAdaptiveFusion(H, H * 2, H, gate)
        elif model_type == 'baseline':
            self.static_fusion = nn.Sequential(nn.Linear(H * 3, H), nn.ReLU(), nn.Dropout(Config.DROPOUT_RATE))

        self.viseme_classifier = nn.Linear(H, num_visemes)
        self.phoneme_head = PriorGuidedPhonemeHead(H, num_visemes, num_phonemes)
        self.ctc_head = nn.Linear(H, CTC_CLASSES)

        self.register_buffer('ambiguity_table', torch.tensor(viseme_mapper.ambiguity_lookup(), dtype=torch.float))

        self.loss_weights = dict(Config.LOSS_WEIGHTS)
        if model_type == 'visual_only':
            self.loss_weights['word'] = 0.0
        self.uses_soft_gate = model_type == 'full' and gate != 'hard'

    # ------------------------------------------------------------------ helpers
    def speaker_ids_for(self, subjects, device=None):
        """Speaker-embedding ids for subject names; speakers not seen in training -> unknown slot."""
        ids = []
        for s in subjects:
            if s in Config.SPEAKER_TO_ID and (not self.known_speakers or s in self.known_speakers):
                ids.append(Config.SPEAKER_TO_ID[s])
            else:
                ids.append(Config.UNKNOWN_SPEAKER_ID)
        return torch.tensor(ids, dtype=torch.long, device=device)

    @staticmethod
    def padding_mask(lengths, seq_len):
        """(B, T) bool, True on padding frames"""
        if lengths is None:
            return None
        return torch.arange(seq_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    # ------------------------------------------------------------------ forward
    def forward(self, frames, lengths=None, speaker_ids=None, quality=None, ambiguity_scores=None):
        """
        Args:
            frames: (B, T, 1, H, W) or (B, T, H, W)
            lengths: (B,) real number of frames per clip (padding after that)
            speaker_ids: (B,) speaker-embedding ids (see speaker_ids_for)
            quality: (B, T, NUM_QUALITY_FEATURES) frame unreliability signals (0 = reliable); zeros if None
            ambiguity_scores: optional (B, T) override for analysis/ablation only (never ground truth
                when measuring accuracy)
        """
        B, T = frames.shape[:2]
        if lengths is not None:
            lengths = lengths.to(frames.device)
        pad_mask = self.padding_mask(lengths, T)
        if quality is None:
            quality = frames.new_zeros(B, T, Config.NUM_QUALITY_FEATURES)

        visual_features = self.visual_encoder(frames)
        sequence_hidden = self.sequence_model(visual_features, lengths, pad_mask)
        aux_viseme_logits = self.aux_viseme_head(sequence_hidden)
        aux_viseme_probs = F.softmax(aux_viseme_logits.float(), dim=-1).detach()

        if ambiguity_scores is None:
            ambiguity_scores = torch.matmul(aux_viseme_probs, self.ambiguity_table)
        entropy = -(aux_viseme_probs * torch.log(aux_viseme_probs + 1e-8)).sum(-1) / torch.log(
            torch.tensor(float(self.num_visemes)))

        draft_log_probs = context_feats = None
        if getattr(self, 'draft_ctc_head', None) is not None:
            draft_log_probs = F.log_softmax(self.draft_ctc_head(sequence_hidden).float(), dim=-1)
            with torch.no_grad():
                draft_tokens, draft_lengths = batch_greedy_indices(draft_log_probs, lengths)
            context_feats = self.context_encoder(draft_tokens, draft_lengths, target_len=T)
            if self.training and Config.CONTEXT_DROPOUT > 0:
                keep_ctx = torch.rand(B, 1, 1, device=context_feats.device) >= Config.CONTEXT_DROPOUT
                context_feats = context_feats * keep_ctx.to(context_feats.dtype)

        language_features = word_logits = None
        if self.model_type in ('full', 'baseline'):
            soft_phonemes = torch.matmul(aux_viseme_probs, self.phoneme_head.confusion_matrix)
            language_features, word_logits = self.language_model(
                phoneme_probs=soft_phonemes, speaker_ids=speaker_ids, sequence_length=T,
                lengths=lengths, pad_mask=pad_mask, context_feats=context_feats)

        fusion_info = {'fusion_weights': None, 'scale_attention_weights': None}
        if self.model_type == 'full':
            signals = {'ambiguity': ambiguity_scores.to(sequence_hidden.dtype),
                       'entropy': entropy.to(sequence_hidden.dtype),
                       'quality': quality.to(sequence_hidden.dtype)}
            fused, fusion_info = self.adaptive_fusion(sequence_hidden, language_features, signals, pad_mask)
        elif self.model_type == 'baseline':
            fused = self.static_fusion(torch.cat([sequence_hidden, language_features], dim=-1))
        else:
            fused = sequence_hidden

        viseme_logits = self.viseme_classifier(fused)
        phoneme_logits = self.phoneme_head(fused, viseme_logits)
        ctc_log_probs = F.log_softmax(self.ctc_head(fused).float(), dim=-1)

        return {
            'ctc_log_probs': ctc_log_probs,
            'viseme_logits': viseme_logits.float(),
            'phoneme_logits': phoneme_logits,
            'aux_viseme_logits': aux_viseme_logits.float(),
            'word_logits': None if word_logits is None else word_logits.float(),
            'draft_log_probs': draft_log_probs,
            'lengths': lengths,
            'fusion_weights': fusion_info.get('fusion_weights'),
            'scale_attention': fusion_info.get('scale_attention_weights'),
            'intrinsic_logit': fusion_info.get('intrinsic_logit'),
            'extrinsic_logit': fusion_info.get('extrinsic_logit'),
            'ambiguity_scores': ambiguity_scores,
            'entropy': entropy,
        }

    # ------------------------------------------------------------------ loss
    @staticmethod
    def _frame_ce(logits, labels, frame_ok, sil_label):
        """Cross-entropy averaged over usable frames, silence frames down-weighted."""
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.clamp_min(0).reshape(-1),
                             reduction='none').view_as(labels)
        weight = frame_ok.float() * torch.where(labels == sil_label, Config.SILENCE_LOSS_WEIGHT, 1.0)
        return (ce * weight).sum() / weight.sum().clamp_min(1.0)

    def compute_loss(self, predictions, targets):
        """
        targets: dict with
          'targets' (B, S), 'target_lengths' (B,)   CTC character targets
          'visemes', 'phonemes', 'words' (B, T)     per-frame labels, -1 on padding
          'detected' (B, T) optional                frames with a real face detection; frames without
                                                     one are left out of the per-frame losses
        """
        losses = {}
        w = self.loss_weights

        # Sentence-level CTC (on CPU for Apple MPS, which has no CTC kernel)
        log_probs = predictions['ctc_log_probs'].transpose(0, 1)  # (T, B, C)
        lengths = predictions['lengths']
        if lengths is None:
            lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
        ctc_device = 'cpu' if log_probs.device.type == 'mps' else log_probs.device
        losses['ctc_loss'] = F.ctc_loss(
            log_probs.to(ctc_device), targets['targets'].to(ctc_device), lengths.to(ctc_device),
            targets['target_lengths'].to(ctc_device), blank=Config.CTC_BLANK, zero_infinity=True
        ).to(log_probs.device)

        # Interim draft CTC (same objective as above, on the unfused sequence features; gives
        # the draft-CTC head its gradients and is only present with --lm-context draft)
        if predictions.get('draft_log_probs') is not None and w.get('draft_ctc', 0) > 0:
            draft_lp = predictions['draft_log_probs'].transpose(0, 1)
            losses['draft_ctc_loss'] = F.ctc_loss(
                draft_lp.to(ctc_device), targets['targets'].to(ctc_device), lengths.to(ctc_device),
                targets['target_lengths'].to(ctc_device), blank=Config.CTC_BLANK, zero_infinity=True
            ).to(log_probs.device)

        visemes = targets['visemes']
        frame_ok = visemes >= 0
        if targets.get('detected') is not None:
            frame_ok = frame_ok & targets['detected'].to(frame_ok.device)

        losses['viseme_loss'] = self._frame_ce(predictions['viseme_logits'], visemes, frame_ok, SIL_VISEME)
        losses['aux_viseme_loss'] = self._frame_ce(predictions['aux_viseme_logits'], visemes, frame_ok, SIL_VISEME)
        losses['phoneme_loss'] = self._frame_ce(predictions['phoneme_logits'], targets['phonemes'], frame_ok,
                                                SIL_PHONEME)
        if predictions['word_logits'] is not None and targets.get('words') is not None and w['word'] > 0:
            losses['word_loss'] = self._frame_ce(predictions['word_logits'], targets['words'], frame_ok, SIL_WORD)

        # Gate-entropy regulariser: pushes the soft gate to commit to vision or language
        if self.uses_soft_gate and predictions['fusion_weights'] is not None:
            valid = (visemes >= 0).float()
            reg = 0.0
            for fw in predictions['fusion_weights']:
                ent = -(fw * torch.log(fw + 1e-8)).sum(-1)
                reg = reg + (ent * valid).sum() / valid.sum().clamp_min(1.0)
            losses['fusion_reg'] = reg / len(predictions['fusion_weights'])

        key = {'ctc_loss': 'ctc', 'viseme_loss': 'viseme', 'aux_viseme_loss': 'aux_viseme',
               'phoneme_loss': 'phoneme', 'word_loss': 'word', 'fusion_reg': 'fusion_reg',
               'draft_ctc_loss': 'draft_ctc'}
        losses['total_loss'] = sum(w[key[name]] * value for name, value in losses.items())
        return losses

    # ------------------------------------------------------------------ inference
    def predict(self, frames, lengths=None, speaker_ids=None, quality=None):
        from src.decoding import constrained_decode, greedy_decode
        self.eval()
        with torch.no_grad():
            p = self.forward(frames, lengths=lengths, speaker_ids=speaker_ids, quality=quality)
            lp = p['ctc_log_probs'].cpu().numpy()
            lens = p['lengths'].cpu().tolist() if p['lengths'] is not None else [lp.shape[1]] * lp.shape[0]
            return {
                'greedy_text': [greedy_decode(lp[i], lens[i]) for i in range(lp.shape[0])],
                'constrained_text': [constrained_decode(lp[i], lens[i]) for i in range(lp.shape[0])],
                'viseme_predictions': torch.argmax(p['viseme_logits'], dim=-1),
                'phoneme_predictions': torch.argmax(p['phoneme_logits'], dim=-1),
                'viseme_probabilities': F.softmax(p['viseme_logits'], dim=-1),
                'phoneme_probabilities': F.softmax(p['phoneme_logits'], dim=-1),
                'ambiguity_scores': p['ambiguity_scores'],
                'fusion_weights': p['fusion_weights'],
            }


class ModelFactory:
    """Builds and loads model variants"""

    @staticmethod
    def create(model_type='full', seq_model=None, visual_pool=None, gate=None, lm_context=None):
        return VisemePhonemeAdaptiveModel(model_type=model_type, seq_model=seq_model,
                                          visual_pool=visual_pool, gate=gate, lm_context=lm_context)

    @staticmethod
    def create_baseline_model():
        return ModelFactory.create('baseline')

    @staticmethod
    def create_visual_only_model():
        return ModelFactory.create('visual_only')

    @staticmethod
    def create_full_model():
        return ModelFactory.create('full')

    @staticmethod
    def load_pretrained_model(checkpoint_path, model_type=None):
        """Load a trained model with the architecture and input settings stored in the checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # The visual encoder's layer sizes depend on the frame size it was trained with
        saved_config = checkpoint.get('config', {})
        saved_size = (saved_config.get('FRAME_HEIGHT'), saved_config.get('FRAME_WIDTH'))
        if None not in saved_size and saved_size != (Config.FRAME_HEIGHT, Config.FRAME_WIDTH):
            print(f"[load_pretrained_model] Using the checkpoint's frame size {saved_size[0]}x{saved_size[1]} "
                  f"(was {Config.FRAME_HEIGHT}x{Config.FRAME_WIDTH})")
            Config.FRAME_HEIGHT, Config.FRAME_WIDTH = saved_size
        for key in ('PIXEL_MEAN', 'PIXEL_STD'):  # same input normalisation as in training
            if key in saved_config:
                setattr(Config, key, saved_config[key])

        model_config = dict(checkpoint.get('model_config') or {'model_type': checkpoint.get('model_type', 'full')})
        if model_type and model_type != model_config.get('model_type'):
            print(f"[load_pretrained_model] Checkpoint was trained as '{model_config.get('model_type')}'; "
                  f"ignoring requested model_type '{model_type}'.")
        model = ModelFactory.create(**model_config)
        model.known_speakers = list(checkpoint.get('known_speakers', []))

        state_dict = upgrade_legacy_state_dict(checkpoint['model_state_dict'])
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        skipped = sorted(set(state_dict) - set(filtered_dict))
        missing = sorted(set(model_dict) - set(filtered_dict))
        if skipped or missing:
            print(f"[load_pretrained_model] WARNING: checkpoint does not match this model: "
                  f"{len(skipped)} checkpoint keys skipped (e.g. {skipped[:3]}), "
                  f"{len(missing)} model keys left at random init (e.g. {missing[:3]}). "
                  f"Results will not reflect the trained model; retrain with this code version.")
        model.load_state_dict(filtered_dict, strict=False)
        return model
