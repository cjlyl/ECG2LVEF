import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            Swish(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, pool=True, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = Swish()
        self.dropout = nn.Dropout(dropout)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity
        out = self.act(out)
        out = self.pool(out)
        return out


class MultiScaleStem(nn.Module):
    """
    澶氬昂搴﹀嵎绉墠绔?    杈撳叆:  (B, 12, 5000)
    杈撳嚭:  (B, out_channels, 5000)
    """
    def __init__(self, in_channels=12, branch_channels=32, dropout=0.1):
        super().__init__()
        self.branch_k3 = ConvBNAct(in_channels, branch_channels, kernel_size=3, dropout=dropout)
        self.branch_k5 = ConvBNAct(in_channels, branch_channels, kernel_size=5, dropout=dropout)
        self.branch_k9 = ConvBNAct(in_channels, branch_channels, kernel_size=9, dropout=dropout)

        fused_channels = branch_channels * 3
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            Swish(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.branch_k3(x)
        x2 = self.branch_k5(x)
        x3 = self.branch_k9(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        return x


class AttentionPooling(nn.Module):
    """
    杈撳叆: (B, T, D)
    杈撳嚭:
      pooled: (B, D)
      attn_weights: (B, T)
    """
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=1)     # (B, T)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, weights


class MultiScaleCNN_BiLSTM_Attn(nn.Module):
    """
    Multi-scale CNN + BiLSTM + Attention Pooling
    杈撳叆:  (B, 12, 5000)
    杈撳嚭:  (B, n_classes)
    """
    def __init__(self, n_classes=3, lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()

        self.stem = MultiScaleStem(in_channels=12, branch_channels=32, dropout=dropout)

        self.block1 = ResidualConvBlock(128, 128, kernel_size=5, pool=True, dropout=dropout)  # 5000 -> 2500
        self.block2 = ResidualConvBlock(128, 256, kernel_size=5, pool=True, dropout=dropout)  # 2500 -> 1250

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.attn_pool = AttentionPooling(in_dim=lstm_hidden * 2, attn_dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, return_attention=False):
        # x: (B, 12, 5000)
        x = self.stem(x)        # (B, 128, 5000)
        x = self.block1(x)      # (B, 128, 2500)
        x = self.block2(x)      # (B, 256, 1250)

        # -> (B, T, C)
        x = x.transpose(1, 2)   # (B, 1250, 256)

        x, _ = self.rnn(x)      # (B, 1250, 2*lstm_hidden)
        pooled, attn_weights = self.attn_pool(x)

        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.image_branch import ECGImageFeatureBranch
from Models.disms_ts_no_fft import DisMSTSNoFFTBranch


# =========================
# Basic blocks
# =========================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            Swish(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """
    Channel attention for feature maps
    Input: (B, C, T)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            Swish(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class LeadSEBlock(nn.Module):
    """
    ECG lead-aware reweighting.
    Emphasizes informative leads for LVEF prediction.
    Input: (B, C, T)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            Swish(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.pool(x).squeeze(-1)   # (B, C)
        w = self.fc(w).unsqueeze(-1)   # (B, C, 1)
        return x * w


class DepthwiseSeparableConv1d(nn.Module):
    """
    ECG-friendly convolution:
    depthwise temporal filtering + pointwise channel mixing
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=pad,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = Swish()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# =========================
# ECG-specific front-end
# =========================
class ECGWaveStem(nn.Module):
    """
    ECG-inspired multi-scale front-end.
    Kernels roughly capture:
      - small kernel: QRS morphology
      - medium kernel: P/T wave structures
      - large kernel: rhythm/context
    Input:  (B, 12, T)
    Output: (B, out_channels, T)
    """
    def __init__(self, in_channels=12, branch_channels=32, out_channels=128, dropout=0.1):
        super().__init__()

        self.branch_qrs = DepthwiseSeparableConv1d(
            in_channels, branch_channels, kernel_size=7, dropout=dropout
        )
        self.branch_pt = DepthwiseSeparableConv1d(
            in_channels, branch_channels, kernel_size=15, dropout=dropout
        )
        self.branch_rhythm = DepthwiseSeparableConv1d(
            in_channels, branch_channels, kernel_size=31, dropout=dropout
        )

        fused_channels = branch_channels * 3
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            Swish(),
            nn.Dropout(dropout)
        )

        self.lead_attn = LeadSEBlock(out_channels, reduction=8)

    def forward(self, x):
        x1 = self.branch_qrs(x)
        x2 = self.branch_pt(x)
        x3 = self.branch_rhythm(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)
        x = self.lead_attn(x)
        return x


# =========================
# Residual feature extractor
# =========================
class ECGResidualBlock(nn.Module):
    """
    Residual block with:
      - ECG-friendly convolution
      - channel attention
      - optional temporal downsampling
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, pool=True, dropout=0.1):
        super().__init__()

        pad = ((kernel_size - 1) // 2) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = Swish()
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(out_channels, reduction=8)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity
        out = self.act(out)
        out = self.pool(out)
        return out


# =========================
# Temporal modeling
# =========================
class RhythmContextBlock(nn.Module):
    """
    Lightweight rhythm context modeling with BiLSTM.
    Input:  (B, T, C)
    Output: (B, T, 2*hidden)
    """
    def __init__(self, in_dim, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attentive temporal pooling
    Input:  (B, T, D)
    Output:
        pooled: (B, D * num_heads)
        attn_weights: (B, num_heads, T)
    """
    def __init__(self, in_dim, attn_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, num_heads)
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn(x)                    # (B, T, H)
        scores = scores.transpose(1, 2)          # (B, H, T)
        weights = torch.softmax(scores, dim=-1)  # (B, H, T)

        pooled = torch.matmul(weights, x)        # (B, H, D)
        pooled = pooled.reshape(x.size(0), -1)   # (B, H*D)
        return pooled, weights


# =========================
# Heads
# =========================
class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Full model
# =========================
class ECGClassifierNet(nn.Module):
    """
    ECG model for standard multi-class classification.

    Input:
      x: (B, 12, 5000) or generally (B, 12, T)

    Output:
      logits: (B, num_classes)
    """
    def __init__(
        self,
        in_channels=12,
        num_classes=3,
        stem_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        attn_heads=4,
        dropout=0.2,
        disms_down_sampling_layers=3,
        disms_temporal_channels=128,
        disms_temporal_size=128,
        disms_output_dim=1024,
        disms_down_sampling_window=2,
        disms_gru_chunk_size=24,
        disms_use_gru_checkpoint=True,
    ):
        super().__init__()
        self.num_classes = num_classes

        rng_before_temporal = torch.random.get_rng_state()
        legacy_stem = ECGWaveStem(
            in_channels=in_channels,
            branch_channels=32,
            out_channels=stem_channels,
            dropout=dropout
        )

        legacy_block1 = ECGResidualBlock(
            stem_channels, 128, kernel_size=7, dilation=1, pool=True, dropout=dropout
        )

        legacy_block2 = ECGResidualBlock(
            128, 256, kernel_size=9, dilation=1, pool=True, dropout=dropout
        )

        legacy_block3 = ECGResidualBlock(
            256, 256, kernel_size=11, dilation=2, pool=True, dropout=dropout
        )

        legacy_temporal_encoder = RhythmContextBlock(
            in_dim=256,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        legacy_attn_pool = MultiHeadAttentionPooling(
            in_dim=lstm_hidden * 2,
            attn_dim=128,
            num_heads=attn_heads
        )

        rng_after_temporal = torch.random.get_rng_state()
        del legacy_stem, legacy_block1, legacy_block2, legacy_block3
        del legacy_temporal_encoder, legacy_attn_pool

        torch.random.set_rng_state(rng_before_temporal)
        self.temporal_branch = DisMSTSNoFFTBranch(
            num_variables=in_channels,
            down_sampling_layers=disms_down_sampling_layers,
            temporal_channels=disms_temporal_channels,
            temporal_size=disms_temporal_size,
            output_dim=disms_output_dim,
            down_sampling_window=disms_down_sampling_window,
            gru_chunk_size=disms_gru_chunk_size,
            use_gru_checkpoint=disms_use_gru_checkpoint,
        )
        torch.random.set_rng_state(rng_after_temporal)

        self.feature_dim = self.temporal_branch.feature_dim
        self.classifier = ClassifierHead(
            in_dim=self.feature_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=dropout
        )

    def extract_features(self, x):
        feature, shared_features, specific_features = self.temporal_branch(x)
        auxiliary = {
            "shared_features": shared_features,
            "specific_features": specific_features,
        }
        return feature, auxiliary

    def forward(self, x, return_attention=False):
        pooled, attn = self.extract_features(x)
        logits = self.classifier(pooled)

        if return_attention:
            return {"logits": logits, "attn": attn}
        return logits


class ECGImageDualBranchNet(ECGClassifierNet):
    """Fuse the original ECG representation with a time-image branch."""

    def __init__(
        self,
        in_channels=12,
        num_classes=3,
        stem_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        attn_heads=4,
        dropout=0.2,
        image_input_points=256,
        image_time_bins=128,
        image_hidden_dim=64,
        image_dropout=0.1,
        image_channel_preset="LEGACY_WAVEFORM_CORE4_TIME1",
        image_channels=5,
        image_time_coordinate_min=-1.0,
        image_time_coordinate_max=1.0,
        disms_down_sampling_layers=3,
        disms_temporal_channels=128,
        disms_temporal_size=128,
        disms_output_dim=1024,
        disms_down_sampling_window=2,
        disms_gru_chunk_size=24,
        disms_use_gru_checkpoint=True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            stem_channels=stem_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            disms_down_sampling_layers=disms_down_sampling_layers,
            disms_temporal_channels=disms_temporal_channels,
            disms_temporal_size=disms_temporal_size,
            disms_output_dim=disms_output_dim,
            disms_down_sampling_window=disms_down_sampling_window,
            disms_gru_chunk_size=disms_gru_chunk_size,
            disms_use_gru_checkpoint=disms_use_gru_checkpoint,
        )

        self.image_branch = ECGImageFeatureBranch(
            num_leads=in_channels,
            input_points=image_input_points,
            num_time_bins=image_time_bins,
            hidden_dim=image_hidden_dim,
            dropout=image_dropout,
            channel_preset=image_channel_preset,
            in_channels=image_channels,
            time_coordinate_min=image_time_coordinate_min,
            time_coordinate_max=image_time_coordinate_max,
        )

        fused_dim = self.feature_dim + image_hidden_dim
        self.classifier = ClassifierHead(
            in_dim=fused_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x, return_attention=False):
        ecg_features, ecg_attention = self.extract_features(x)
        image_features, image_attention = self.image_branch(x)
        fused_features = torch.cat([ecg_features, image_features], dim=-1)
        logits = self.classifier(fused_features)

        if return_attention:
            return {
                "logits": logits,
                "attn": ecg_attention,
                "image_attn": image_attention,
            }
        return logits


class SymmetricCosineClassifier(nn.Module):
    """Single permutation-invariant classifier with equal-status class weights."""

    def __init__(
        self,
        in_dim,
        feature_dim=128,
        num_classes=3,
        temperature=16.0,
        dropout=0.2,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            Swish(),
            nn.Dropout(dropout),
        )
        self.class_weights = nn.Parameter(
            torch.empty(num_classes, feature_dim)
        )
        nn.init.xavier_uniform_(self.class_weights)
        self.temperature = float(temperature)

    def forward(self, x):
        embedding = self.projector(x)
        normalized_embedding = torch.nn.functional.normalize(
            embedding, dim=-1, eps=1e-6
        )
        normalized_weights = torch.nn.functional.normalize(
            self.class_weights, dim=-1, eps=1e-6
        )
        cosine_scores = normalized_embedding @ normalized_weights.transpose(0, 1)
        logits = self.temperature * cosine_scores
        return logits, cosine_scores, normalized_embedding, normalized_weights


class ECGStageA2PermutationInvariantAmbiguityCosineDisMSNet(
    ECGImageDualBranchNet
):
    """Strictly unordered DisMS model with one symmetric cosine classifier."""

    def __init__(
        self,
        *args,
        cosine_feature_dim=128,
        cosine_temperature=16.0,
        **kwargs,
    ):
        image_hidden_dim = kwargs.get("image_hidden_dim", 64)
        classifier_dropout = kwargs.get("dropout", 0.2)
        num_classes = kwargs.get("num_classes", 3)
        super().__init__(*args, **kwargs)
        fused_dim = self.feature_dim + image_hidden_dim
        self.classifier = SymmetricCosineClassifier(
            in_dim=fused_dim,
            feature_dim=cosine_feature_dim,
            num_classes=num_classes,
            temperature=cosine_temperature,
            dropout=classifier_dropout,
        )

    def forward(self, x, return_attention=False):
        ecg_features, ecg_auxiliary = self.extract_features(x)
        image_features, image_attention = self.image_branch(x)
        fused_features = torch.cat([ecg_features, image_features], dim=-1)
        fused_features = torch.nan_to_num(
            fused_features, nan=0.0, posinf=1e4, neginf=-1e4
        )
        (
            logits,
            cosine_scores,
            normalized_embedding,
            normalized_class_weights,
        ) = self.classifier(fused_features)

        outputs = {
            "logits": logits,
            "cosine_scores": cosine_scores,
            "normalized_embedding": normalized_embedding,
            "normalized_class_weights": normalized_class_weights,
            "disms_shared_features": ecg_auxiliary["shared_features"],
            "disms_specific_features": ecg_auxiliary["specific_features"],
        }
        if return_attention:
            outputs["attn"] = ecg_auxiliary
            outputs["image_attn"] = image_attention
        return outputs
