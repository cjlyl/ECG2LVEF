"""DisMS-TS temporal branch without the official repository's unused FFT path."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GRUModel(nn.Module):
    """Two-stage GRU used by the official DisMS-TS implementation."""

    def __init__(self, input_size=128, hidden_size=128, output_size=128):
        super().__init__()
        self.rnn1 = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        self.relu1 = nn.ReLU()
        self.rnn2 = nn.GRU(hidden_size, output_size, 1, batch_first=True)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.relu1(x)
        x, _ = self.rnn2(x)
        return self.relu2(x)


class DisMSTSNoFFTBranch(nn.Module):
    """Paper-faithful multi-scale temporal disentanglement feature encoder.

    Input: [B, N, T]. Output feature: [B, output_dim]. Auxiliary shared and
    specific tensors are returned for the paper's similarity/disparity losses.
    """

    def __init__(
        self,
        num_variables=12,
        down_sampling_layers=3,
        temporal_channels=128,
        temporal_size=128,
        output_dim=1024,
        down_sampling_window=2,
        gru_chunk_size=24,
        use_gru_checkpoint=True,
    ):
        super().__init__()
        if down_sampling_layers < 0:
            raise ValueError("down_sampling_layers must be non-negative")
        if temporal_size % 4 != 0:
            raise ValueError("temporal_size must be divisible by 4")
        if down_sampling_window < 2:
            raise ValueError("down_sampling_window must be at least 2")

        self.num_variables = int(num_variables)
        self.down_sampling_layers = int(down_sampling_layers)
        self.num_scales = self.down_sampling_layers + 1
        self.temporal_channels = int(temporal_channels)
        self.temporal_size = int(temporal_size)
        self.disentangled_dim = self.temporal_size // 4
        self.down_sampling_window = int(down_sampling_window)
        self.gru_chunk_size = int(gru_chunk_size)
        self.use_gru_checkpoint = bool(use_gru_checkpoint)
        if self.gru_chunk_size < 1:
            raise ValueError("gru_chunk_size must be positive")

        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.temporal_channels,
                    kernel_size=(1, 3),
                    padding=(0, 1),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.normalize_layers = nn.ModuleList(
            [nn.BatchNorm2d(self.temporal_channels) for _ in range(self.num_scales)]
        )
        self.temporal = GRUModel(
            input_size=self.temporal_channels,
            hidden_size=self.temporal_channels,
            output_size=self.temporal_size,
        )
        self.variable_attention = nn.Parameter(
            torch.ones(1, self.num_variables, 1)
        )
        self.temporal_mapping = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.temporal_size * self.num_variables,
                        self.temporal_size,
                    ),
                    nn.Linear(self.temporal_size, self.disentangled_dim),
                )
                for _ in range(self.num_scales)
            ]
        )
        self.temporal_normalizes = nn.ModuleList(
            [nn.BatchNorm1d(self.disentangled_dim) for _ in range(self.num_scales)]
        )
        self.disentanglement_mask = nn.Linear(
            self.disentangled_dim, self.disentangled_dim
        )

        # One averaged shared vector plus every scale-specific vector.
        representation_dim = self.disentangled_dim * (self.num_scales + 1)
        self.output_adapter = nn.Linear(representation_dim, output_dim)
        self.feature_dim = int(output_dim)

    def multi_scale_process(self, x):
        """Generate temporal scales only; official spectral tensors were unused."""
        scales = [x]
        current = x
        for _ in range(self.down_sampling_layers):
            current = F.avg_pool1d(
                current,
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
            )
            scales.append(current)
        return scales

    def _encode_last_step(self, sequence_batch):
        return self.temporal(sequence_batch)[:, -1]

    def _memory_safe_temporal_encode(self, local):
        """Exactly encode independent variable sequences in smaller batches."""
        encoded_chunks = []
        for chunk in local.split(self.gru_chunk_size, dim=0):
            if self.training and self.use_gru_checkpoint:
                encoded = checkpoint(self._encode_last_step, chunk)
            else:
                encoded = self._encode_last_step(chunk)
            encoded_chunks.append(encoded)
        return torch.cat(encoded_chunks, dim=0)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, N, T], got {tuple(x.shape)}")
        if x.size(1) != self.num_variables:
            raise ValueError(
                f"Expected {self.num_variables} variables, got {x.size(1)}"
            )

        batch_size, num_variables, _ = x.shape
        temporal_scales = self.multi_scale_process(x)
        scale_representations = []

        for scale_index, scale_x in enumerate(temporal_scales):
            local = self.scale_convs[scale_index](scale_x.unsqueeze(1))
            local = self.normalize_layers[scale_index](local)
            local = local.permute(0, 2, 3, 1).reshape(
                batch_size * num_variables,
                scale_x.size(-1),
                self.temporal_channels,
            )
            encoded = self._memory_safe_temporal_encode(local)
            encoded = encoded.reshape(batch_size, num_variables, -1)
            encoded = encoded * self.variable_attention
            mapped = self.temporal_mapping[scale_index](
                encoded.reshape(batch_size, -1)
            )
            mapped = self.temporal_normalizes[scale_index](mapped)
            scale_representations.append(mapped)

        shared_features = []
        specific_features = []
        for representation in scale_representations:
            mask_logits = self.disentanglement_mask(representation)
            shared_features.append(torch.sigmoid(mask_logits) * representation)
            specific_features.append(torch.sigmoid(-mask_logits) * representation)

        shared_stack = torch.stack(shared_features, dim=1)
        specific_stack = torch.stack(specific_features, dim=1)
        shared_mean = shared_stack.mean(dim=1)
        specific_concat = specific_stack.reshape(batch_size, -1)
        representation = torch.cat([shared_mean, specific_concat], dim=-1)
        feature = self.output_adapter(representation)
        return feature, shared_stack, specific_stack


def disms_similarity_loss(shared_features):
    """Official Sim loss, vectorized without changing its objective."""
    normalized = F.normalize(
        shared_features,
        p=2,
        dim=-1,
        eps=1e-6,
    )
    similarities = torch.bmm(normalized, normalized.transpose(1, 2))
    return F.mse_loss(similarities, torch.ones_like(similarities))


def disms_disparity_loss(specific_features):
    """Official Dis loss, vectorized without changing its objective."""
    normalized = F.normalize(
        specific_features,
        p=2,
        dim=-1,
        eps=1e-6,
    )
    similarities = torch.bmm(normalized, normalized.transpose(1, 2))
    identity = torch.eye(
        similarities.size(1),
        dtype=similarities.dtype,
        device=similarities.device,
    ).unsqueeze(0).expand_as(similarities)
    return F.mse_loss(similarities, identity)
