import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


class VisualEncoder(nn.Module):
    def __init__(self, input_dim=None,
                 output_dim=None,
                 input_channels=1,
                 dummy_seq_len=None):
        super(VisualEncoder, self).__init__()

        # Read Config at construction time (not import time) so runtime overrides such as
        # main.py --frame-size take effect.
        input_dim = input_dim or (Config.FRAME_HEIGHT, Config.FRAME_WIDTH)
        output_dim = output_dim or Config.VISUAL_FEATURE_DIM

        self.input_channels = input_channels
        self.input_dim = input_dim
        self.dummy_seq_len = dummy_seq_len or Config.SEQUENCE_LENGTH

        # 3D CNN for spatio-temporal feature extraction
        self.conv3d_1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3d_4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3d_4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Batch normalization layers
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)

        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

        # Calculate flattened dimension after conv layers
        self.conv_output_size = self._calculate_conv_output_size(self.input_dim, self.dummy_seq_len)
        print(f"[VisualEncoder] Calculated conv output size: {self.conv_output_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def _calculate_conv_output_size(self, input_dim, seq_len):
        """Calculate output size by doing a forward pass with the conv layers."""
        original_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, seq_len, input_dim[0], input_dim[1])
                x = self.pool3d_1(F.relu(self.bn1(self.conv3d_1(dummy_input))))
                x = self.pool3d_2(F.relu(self.bn2(self.conv3d_2(x))))
                x = self.pool3d_3(F.relu(self.bn3(self.conv3d_3(x))))
                x = self.pool3d_4(F.relu(self.bn4(self.conv3d_4(x))))
                return x.view(1, -1).size(1)
        finally:
            self.train(original_training)

    def forward(self, x):
        batch_size, seq_len, *spatial = x.shape
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        # Convolutions
        x = self.pool3d_1(F.relu(self.bn1(self.conv3d_1(x))))
        x = self.dropout(x)
        x = self.pool3d_2(F.relu(self.bn2(self.conv3d_2(x))))
        x = self.dropout(x)
        x = self.pool3d_3(F.relu(self.bn3(self.conv3d_3(x))))
        x = self.dropout(x)
        x = self.pool3d_4(F.relu(self.bn4(self.conv3d_4(x))))
        x = self.dropout(x)

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, -1)

        # Dynamically create fc1 if not exist
        self.ensure_dynamic_layers(x.size(1), x.device)

        x = F.relu(self.fc1_dynamic(x))
        x = self.dropout(x)
        features = self.fc2_dynamic(x)

        features = features.view(b, t, -1)
        if t != seq_len:
            features = F.interpolate(features.permute(0, 2, 1),
                                     size=seq_len,
                                     mode='linear',
                                     align_corners=False)
            features = features.permute(0, 2, 1)

        return features

    def ensure_dynamic_layers(self, in_features, device=None):
        """Create fc1_dynamic/fc2_dynamic (per-timestep projection) if they don't exist yet."""
        if not hasattr(self, 'fc1_dynamic'):
            self.fc1_dynamic = nn.Linear(in_features, 1024)
            self.fc2_dynamic = nn.Linear(1024, Config.VISUAL_FEATURE_DIM)
            if device is not None:
                self.fc1_dynamic.to(device)
                self.fc2_dynamic.to(device)


def materialize_lazy_layers(model, state_dict):
    """Create any lazily-built VisualEncoder layers present in state_dict before loading it.

    VisualEncoder builds fc1_dynamic/fc2_dynamic on its first forward pass, so a freshly
    constructed model has no such parameters and load_state_dict would reject (strict) or
    silently drop (strict=False / key filtering) the trained weights.
    """
    suffix = '.fc1_dynamic.weight'
    for key, tensor in state_dict.items():
        if key.endswith(suffix):
            encoder = model.get_submodule(key[:-len(suffix)])
            device = next(encoder.parameters()).device
            encoder.ensure_dynamic_layers(tensor.shape[1], device)


class LipROI(nn.Module):
    """Lip Region of Interest extraction module"""

    def __init__(self):
        super(LipROI, self).__init__()
        self.attention_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 5:
            batch_size, seq_len, channels, height, width = x.shape
            x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        else:
            batch_size, seq_len, height, width = x.shape
            x_reshaped = x.view(batch_size * seq_len, 1, height, width)
        attention = self.sigmoid(self.attention_conv(x_reshaped))
        x_attended = x_reshaped * attention
        x_attended = x_attended.view(batch_size, seq_len, height, width)
        return x_attended


class EnhancedVisualEncoder(nn.Module):
    """Visual encoder with lip ROI extraction"""

    def __init__(self):
        super(EnhancedVisualEncoder, self).__init__()
        self.lip_roi = LipROI()
        self.visual_encoder = VisualEncoder()

    def forward(self, x):
        x = self.lip_roi(x)
        features = self.visual_encoder(x)
        return features
