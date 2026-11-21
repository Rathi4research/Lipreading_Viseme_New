import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class VisualEncoder(nn.Module):
    """3D-CNN over the lip-crop sequence -> one feature vector per input frame.

    Pooling is spatial only (kernel (1, 2, 2)), so the temporal resolution of the input is kept
    end to end: every one of the SEQUENCE_LENGTH frames gets its own feature vector (and its own
    per-frame viseme/phoneme prediction). The earlier design pooled time 75 -> 9 and linearly
    interpolated back, i.e. ~330 ms per step, coarser than most phonemes.
    """

    def __init__(self, input_dim=None,
                 output_dim=None,
                 input_channels=1,
                 pool=None):
        super(VisualEncoder, self).__init__()

        # 'avg' (default): average the final feature map over its spatial grid, then project
        #   (512 -> 1024 -> 512, about 1M parameters).
        # 'flatten': flatten channels x H' x W' per frame before the projection (the original
        #   design; about 13M parameters in the first projection at 88x88 input).
        self.pool = pool or Config.VISUAL_POOL
        if self.pool not in ('avg', 'flatten'):
            raise ValueError(f"visual pool must be 'avg' or 'flatten', got {self.pool!r}")

        # Read Config at construction time (not import time) so runtime overrides such as
        # main.py --frame-size take effect.
        input_dim = input_dim or (Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
        output_dim = output_dim or Config.VISUAL_FEATURE_DIM

        self.input_channels = input_channels
        self.input_dim = input_dim

        # 3D CNN for spatio-temporal feature extraction (temporal kernel 3, temporal stride 1)
        self.conv3d_1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Batch normalization layers
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)

        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Per-frame feature size after the conv stack (channels * H' * W'). Built here, in
        # __init__, so the projection is registered before the optimizer is created and is
        # therefore actually trained (previously it was created lazily on the first forward
        # pass and never reached the optimizer).
        self.frame_feature_size = self._calculate_frame_feature_size(self.input_dim)

        # Per-frame projection
        self.fc1 = nn.Linear(self.frame_feature_size, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def _conv_stack(self, x):
        x = self.dropout(self.pool3d_1(F.relu(self.bn1(self.conv3d_1(x)))))
        x = self.dropout(self.pool3d_2(F.relu(self.bn2(self.conv3d_2(x)))))
        x = self.dropout(self.pool3d_3(F.relu(self.bn3(self.conv3d_3(x)))))
        x = self.dropout(self.pool3d_4(F.relu(self.bn4(self.conv3d_4(x)))))
        return x

    def _calculate_frame_feature_size(self, input_dim):
        """Channels * H' * W' of the conv output for a single frame position."""
        original_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                # 3 frames is enough: time is never pooled, so only H/W matter
                dummy_input = torch.zeros(1, self.input_channels, 3, input_dim[0], input_dim[1])
                _, c, _, h, w = self._conv_stack(dummy_input).shape
                return c if self.pool == 'avg' else c * h * w
        finally:
            self.train(original_training)

    def forward(self, x):
        """
        Args:
            x: frames (B, T, H, W) or (B, T, 1, H, W)
        Returns:
            features: (B, T, output_dim)
        """
        if x.dim() == 5:
            x = x.squeeze(2)
        batch_size, seq_len = x.shape[:2]
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        x = self._conv_stack(x)

        b, c, t, h, w = x.shape
        if self.pool == 'avg':
            x = x.mean(dim=(3, 4)).permute(0, 2, 1).reshape(b * t, c)
        else:
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c * h * w)

        x = self.dropout(F.relu(self.fc1(x)))
        features = self.fc2(x).view(b, t, -1)
        return features


def upgrade_legacy_state_dict(state_dict):
    """Map checkpoints from the old VisualEncoder layout onto the current one.

    Old layout: an unused static fc1/fc2 sized for the whole (temporally pooled) clip, plus the
    per-frame projection created lazily as fc1_dynamic/fc2_dynamic. Current layout: the per-frame
    projection is fc1/fc2. Old weights are only shape-compatible if the old model used the same
    frame size; load_pretrained_model() drops anything that still doesn't fit.
    """
    if not any('.fc1_dynamic.' in k for k in state_dict):
        return state_dict
    upgraded = {}
    for key, value in state_dict.items():
        if '.fc1_dynamic.' in key or '.fc2_dynamic.' in key:
            upgraded[key.replace('_dynamic.', '.')] = value
        elif '.visual_encoder.fc1.' in key or '.visual_encoder.fc2.' in key:
            continue  # old static (never-used) layers
        else:
            upgraded[key] = value
    return upgraded


class LipROI(nn.Module):
    """Lip Region of Interest extraction module (per-frame spatial attention gate)"""

    def __init__(self):
        super(LipROI, self).__init__()
        self.attention_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: (B, T, H, W) or (B, T, 1, H, W) -> (B, T, H, W)"""
        if x.dim() == 5:
            x = x.squeeze(2)
        batch_size, seq_len, height, width = x.shape
        x_reshaped = x.reshape(batch_size * seq_len, 1, height, width)
        attention = self.sigmoid(self.attention_conv(x_reshaped))
        x_attended = x_reshaped * attention
        return x_attended.view(batch_size, seq_len, height, width)


class EnhancedVisualEncoder(nn.Module):
    """Visual encoder with lip ROI extraction"""

    def __init__(self, pool=None):
        super(EnhancedVisualEncoder, self).__init__()
        self.lip_roi = LipROI()
        self.visual_encoder = VisualEncoder(pool=pool)

    def forward(self, x):
        x = self.lip_roi(x)
        features = self.visual_encoder(x)
        return features
