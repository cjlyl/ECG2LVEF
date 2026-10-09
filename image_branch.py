import torch
import torch.nn as nn
import torch.nn.functional as F


class IrregularSeriesToImageMultiScale(nn.Module):
    """Build the EXP-025 waveform-core plus explicit-time representation."""

    def __init__(
        self,
        num_time_bins=128,
        sigma_value=0.05,
        sigma_short=0.03,
        sigma_long=0.12,
        tau=0.02,
        eps=1e-6,
        time_coordinate_min=-1.0,
        time_coordinate_max=1.0,
    ):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.sigma_value = sigma_value
        self.sigma_short = sigma_short
        self.sigma_long = sigma_long
        self.tau = tau
        self.eps = eps
        self.time_coordinate_min = float(time_coordinate_min)
        self.time_coordinate_max = float(time_coordinate_max)
        if self.time_coordinate_max <= self.time_coordinate_min:
            raise ValueError("time_coordinate_max must exceed time_coordinate_min")
        self.channel_names = (
            "ValueMap",
            "SlopeShortMap",
            "SlopeLongMap",
            "VarianceMap",
            "ExplicitTimeCoordinate",
        )

    @staticmethod
    def _gaussian_weight(dt, sigma):
        return torch.exp(-(dt ** 2) / (2 * sigma ** 2))

    def _standardize_per_var(self, values):
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True) + self.eps
        return (values - mean) / std

    def forward(self, values, time_steps, mask):
        """
        Args:
            values: [B, M, L, N]
            time_steps: [B, M, L, N]
            mask: [B, M, L, N]

        Returns:
            Tensor with shape [B, 5, N, T].
        """
        device = values.device
        batch_size, num_segments, segment_length, num_vars = values.shape
        num_bins = self.num_time_bins
        num_points = num_segments * segment_length

        values_flat = values.permute(0, 3, 1, 2).reshape(batch_size, num_vars, num_points)
        times_flat = time_steps.permute(0, 3, 1, 2).reshape(batch_size, num_vars, num_points)
        mask_flat = mask.permute(0, 3, 1, 2).reshape(batch_size, num_vars, num_points).float()

        sorted_times, sort_idx = torch.sort(times_flat, dim=-1)
        values_flat = torch.gather(values_flat, dim=-1, index=sort_idx)
        mask_flat = torch.gather(mask_flat, dim=-1, index=sort_idx)
        times_flat = sorted_times

        grid = torch.linspace(0.0, 1.0, num_bins, device=device).view(1, 1, 1, num_bins)
        time_expand = times_flat.unsqueeze(-1)
        value_expand = values_flat.unsqueeze(-1)
        mask_expand = mask_flat.unsqueeze(-1)

        dt = grid - time_expand
        abs_dt = dt.abs()

        value_weights = self._gaussian_weight(dt, self.sigma_value) * mask_expand
        value_weight_sum = value_weights.sum(dim=2)
        value_map = (value_weights * value_expand).sum(dim=2) / (value_weight_sum + self.eps)
        variance_map = (
            value_weights * (value_expand - value_map.unsqueeze(2)) ** 2
        ).sum(dim=2) / (value_weight_sum + self.eps)

        previous_values = torch.roll(values_flat, shifts=1, dims=-1)
        previous_times = torch.roll(times_flat, shifts=1, dims=-1)
        previous_mask = torch.roll(mask_flat, shifts=1, dims=-1)

        valid_pair = mask_flat * previous_mask
        valid_pair[..., 0] = 0.0
        slope = (values_flat - previous_values) / (
            (times_flat - previous_times).abs() + self.eps
        )
        slope = slope * valid_pair

        short_weights = self._gaussian_weight(dt, self.sigma_short) * mask_expand
        short_weight_sum = (short_weights * valid_pair.unsqueeze(-1)).sum(dim=2)
        slope_short_map = (short_weights * slope.unsqueeze(-1)).sum(dim=2) / (
            short_weight_sum + self.eps
        )

        long_weights = self._gaussian_weight(dt, self.sigma_long) * mask_expand
        long_weight_sum = (long_weights * valid_pair.unsqueeze(-1)).sum(dim=2)
        slope_long_map = (long_weights * slope.unsqueeze(-1)).sum(dim=2) / (
            long_weight_sum + self.eps
        )

        value_map = self._standardize_per_var(value_map)
        variance_map = self._standardize_per_var(variance_map)
        slope_short_map = self._standardize_per_var(slope_short_map)
        slope_long_map = self._standardize_per_var(slope_long_map)
        time_coordinate_map = grid.squeeze(2).expand(
            batch_size, num_vars, num_bins
        )
        time_coordinate_map = (
            self.time_coordinate_min
            + (self.time_coordinate_max - self.time_coordinate_min)
            * time_coordinate_map
        )

        return torch.stack(
            [
                value_map,
                slope_short_map,
                slope_long_map,
                variance_map,
                time_coordinate_map,
            ],
            dim=1,
        )


class VariableImageEncoderMultiScale(nn.Module):
    """Encode the five-channel EXP-025 image into variable features."""

    def __init__(
        self,
        in_channels=5,
        hidden_dim=64,
        num_vars=12,
        temperature=0.1,
        dropout=0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.time_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.var_graph_conv1 = nn.Linear(64, 64)
        self.var_graph_conv2 = nn.Linear(64, 64)
        self.graph_norm = nn.LayerNorm(64)

        self.fine_grained_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 2), dilation=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.residual_fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(64, hidden_dim)
        self.var_embedding = nn.Embedding(num_vars, 64)

    def _build_var_adjacency(self, features):
        num_vars = features.size(1)
        normalized = F.normalize(features, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        adjacency = F.softmax(similarity / self.temperature, dim=-1)
        adjacency = adjacency + torch.eye(num_vars, device=features.device).unsqueeze(0)
        return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def forward(self, image):
        batch_size, _, num_vars, _ = image.shape
        cnn_features = self.cnn_backbone(image)

        variable_features = self.time_pool(cnn_features).squeeze(-1).permute(0, 2, 1)
        variable_ids = torch.arange(num_vars, device=image.device)
        variable_features = variable_features + self.var_embedding(variable_ids).unsqueeze(0)

        adjacency = self._build_var_adjacency(variable_features)
        graph_features = F.gelu(self.var_graph_conv1(torch.bmm(adjacency, variable_features)))
        graph_features = self.var_graph_conv2(torch.bmm(adjacency, graph_features))
        graph_features = self.graph_norm(graph_features + variable_features)

        fine_features = self.fine_grained_branch(cnn_features)
        fine_features = fine_features.mean(dim=-1).permute(0, 2, 1)

        fused = self.residual_fusion(torch.cat([graph_features, fine_features], dim=-1))
        fused = fused + variable_features
        return self.output_proj(self.dropout(fused))


class ECGImageFeatureBranch(nn.Module):
    """Adapt a regular 12-lead ECG into the provided image branch."""

    def __init__(
        self,
        num_leads=12,
        input_points=256,
        num_time_bins=128,
        hidden_dim=64,
        dropout=0.1,
        channel_preset="LEGACY_WAVEFORM_CORE4_TIME1",
        in_channels=5,
        time_coordinate_min=-1.0,
        time_coordinate_max=1.0,
    ):
        super().__init__()
        self.input_points = input_points
        expected_preset = "LEGACY_WAVEFORM_CORE4_TIME1"
        if channel_preset != expected_preset:
            raise ValueError(
                f"EXP-025 requires channel_preset={expected_preset!r}, "
                f"got {channel_preset!r}"
            )
        if int(in_channels) != 5:
            raise ValueError(f"EXP-025 requires five image channels, got {in_channels}")
        self.channel_preset = channel_preset
        self.channel_names = (
            "ValueMap",
            "SlopeShortMap",
            "SlopeLongMap",
            "VarianceMap",
            "ExplicitTimeCoordinate",
        )
        self.to_image = IrregularSeriesToImageMultiScale(
            num_time_bins=num_time_bins,
            time_coordinate_min=time_coordinate_min,
            time_coordinate_max=time_coordinate_max,
        )
        self.encoder = VariableImageEncoderMultiScale(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_vars=num_leads,
            dropout=dropout,
        )
        self.variable_attention = nn.Linear(hidden_dim, 1)

    def forward(self, ecg):
        downsampled = F.adaptive_avg_pool1d(ecg, self.input_points)
        batch_size, num_leads, num_points = downsampled.shape

        values = downsampled.transpose(1, 2).unsqueeze(1)
        time_grid = torch.linspace(
            0.0,
            1.0,
            num_points,
            device=ecg.device,
            dtype=ecg.dtype,
        )
        time_steps = time_grid.view(1, 1, num_points, 1).expand(
            batch_size, 1, num_points, num_leads
        )
        mask = torch.ones_like(values)

        with torch.no_grad():
            image = self.to_image(values, time_steps, mask)
        if image.shape[1] != len(self.channel_names):
            raise RuntimeError(
                f"EXP-025 image channel mismatch: expected {len(self.channel_names)}, "
                f"got {image.shape[1]}"
            )

        variable_features = self.encoder(image)
        attention = torch.softmax(self.variable_attention(variable_features), dim=1)
        pooled = torch.sum(attention * variable_features, dim=1)
        return pooled, attention.squeeze(-1)
