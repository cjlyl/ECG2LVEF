import copy
import random
from contextlib import contextmanager

import numpy as np
import torch

from meter import Meter
from mscnn_bilstm_attn import (
    ECGStageA2PermutationInvariantAmbiguityCosineDisMSNet,
)


def build_model(config):
    """Build the exact EXP025 prediction architecture (1024 + 64 -> 1088)."""
    return ECGStageA2PermutationInvariantAmbiguityCosineDisMSNet(
        in_channels=12,
        num_classes=config.num_classes,
        dropout=config.dropout,
        image_input_points=config.image_input_points,
        image_time_bins=config.image_time_bins,
        image_hidden_dim=config.image_hidden_dim,
        image_dropout=config.image_dropout,
        image_channel_preset=config.image_channel_preset,
        image_channels=config.image_channels,
        image_time_coordinate_min=config.image_time_coordinate_min,
        image_time_coordinate_max=config.image_time_coordinate_max,
        disms_down_sampling_layers=config.disms_down_sampling_layers,
        disms_temporal_channels=config.disms_temporal_channels,
        disms_temporal_size=config.disms_temporal_size,
        disms_output_dim=config.disms_output_dim,
        disms_down_sampling_window=config.disms_down_sampling_window,
        disms_gru_chunk_size=config.disms_gru_chunk_size,
        disms_use_gru_checkpoint=config.disms_use_gru_checkpoint,
        cosine_feature_dim=config.cosine_feature_dim,
        cosine_temperature=config.cosine_temperature,
    )


def clone_state_dict_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


class ModelEMA:
    """Full-state EMA including floating BatchNorm running statistics."""

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError('EMA decay must be in (0, 1)')
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError('EMA/model state_dict keys changed during training')
        for name, tensor in current.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                shadow.mul_(self.decay).add_(
                    tensor.detach().to(device=shadow.device, dtype=shadow.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                shadow.copy_(tensor.detach().to(device=shadow.device))

    def state_dict_cpu(self):
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.shadow.items()
        }

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def make_ema_eval_model(model, ema):
    """Separate eval model: evaluation cannot mutate the Raw model/buffers."""
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    ema_model.eval()
    return ema_model


def capture_rng_state():
    """Capture all RNG sources that may affect the next training epoch."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': None,
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@contextmanager
def preserve_rng_state():
    """Run extra diagnostics/evaluation without perturbing future training RNG."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def _accumulate_losses(totals, loss_dict):
    for key, value in loss_dict.items():
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'item'):
            value = value.item()
        totals[key] = totals.get(key, 0.0) + float(value)


def _original_style_total_grad_norm(parameters, norm_type=2.0):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    first_device = parameters[0].grad.device
    norms = [
        torch.norm(p.grad.detach(), norm_type).to(first_device)
        for p in parameters
    ]
    return torch.norm(torch.stack(norms), norm_type)


def _nonfinite_gradient_names(model, max_names=12):
    names = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad.detach()).all():
            names.append(name)
            if len(names) >= int(max_names):
                break
    return names


def _overflow_fallback_clip_grad_norm_(parameters, max_norm=1.0, norm_type=2.0):
    """EXP025 clipping; FP64 fallback only for finite-element FP32 norm overflow."""
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0), False

    original_norm = _original_style_total_grad_norm(parameters, norm_type)
    if torch.isfinite(original_norm):
        clipped = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=False,
        )
        return clipped, False

    # Genuine NaN/Inf gradient elements must not be hidden by FP64 fallback.
    for p in parameters:
        if not torch.isfinite(p.grad.detach()).all():
            return original_norm, False

    first_device = parameters[0].grad.device
    norms = [
        torch.linalg.vector_norm(
            p.grad.detach().double(), ord=norm_type
        ).to(first_device)
        for p in parameters
    ]
    stable_norm = torch.linalg.vector_norm(torch.stack(norms), ord=norm_type)
    if not torch.isfinite(stable_norm):
        return stable_norm, False

    coefficient = float(max_norm) / (stable_norm + 1e-6)
    coefficient = torch.clamp(coefficient, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(coefficient.to(p.grad.device))
    return stable_norm, True


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    ema=None,
    grad_max_norm=1.0,
    max_nonfinite_batches=5,
):
    """Train one epoch while skipping isolated non-finite batches.

    The whole run is stopped only after more than max_nonfinite_batches invalid
    batches occur in the same epoch. Invalid parameter names are returned for
    diagnosis and are never silently repaired.
    """
    model.train()
    totals = {}
    skipped_nonfinite_batches = 0
    overflow_fallback_batches = 0
    valid_batches = 0
    nonfinite_events = []

    for batch_index, (signals, labels) in enumerate(train_loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(signals)
        loss, loss_dict = criterion(outputs, labels)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            skipped_nonfinite_batches += 1
            nonfinite_events.append(
                f'batch={batch_index}: nonfinite_loss={float(loss.detach().cpu())}'
            )
            if skipped_nonfinite_batches > int(max_nonfinite_batches):
                raise FloatingPointError(
                    f'Non-finite batch limit exceeded: '
                    f'{skipped_nonfinite_batches} > {max_nonfinite_batches}; '
                    f'events={nonfinite_events[-3:]}'
                )
            continue

        loss.backward()
        grad_norm, used_fallback = _overflow_fallback_clip_grad_norm_(
            model.parameters(), max_norm=grad_max_norm, norm_type=2.0
        )
        if used_fallback:
            overflow_fallback_batches += 1

        if not torch.isfinite(grad_norm):
            bad_names = _nonfinite_gradient_names(model)
            optimizer.zero_grad()
            skipped_nonfinite_batches += 1
            nonfinite_events.append(
                f'batch={batch_index}: nonfinite_grad; params={bad_names}'
            )
            if skipped_nonfinite_batches > int(max_nonfinite_batches):
                raise FloatingPointError(
                    f'Non-finite gradient limit exceeded: '
                    f'{skipped_nonfinite_batches} > {max_nonfinite_batches}; '
                    f'events={nonfinite_events[-3:]}'
                )
            continue

        optimizer.step()
        if ema is not None:
            ema.update(model)
        _accumulate_losses(totals, loss_dict)
        valid_batches += 1

    if valid_batches == 0:
        raise FloatingPointError('No finite training batch remained in epoch')

    averaged = {key: value / valid_batches for key, value in totals.items()}
    averaged['skipped_nonfinite_batches'] = float(skipped_nonfinite_batches)
    averaged['overflow_fallback_batches'] = float(overflow_fallback_batches)
    averaged['lr'] = float(optimizer.param_groups[0]['lr'])
    averaged['nonfinite_events'] = list(nonfinite_events)
    return averaged


def evaluate_epoch(model, data_loader, criterion, device, num_classes=3):
    model.eval()
    meter = Meter(n_classes=num_classes)
    totals = {}

    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(signals)
            _, loss_dict = criterion(outputs, labels)
            _accumulate_losses(totals, loss_dict)
            meter.update(outputs, labels)

    n_batches = max(len(data_loader), 1)
    avg_loss = {key: value / n_batches for key, value in totals.items()}
    metrics, reports, _, y_true, y_pred = meter.get_metrics()
    return avg_loss, metrics, reports, y_true, y_pred


def predict_probabilities(model, data_loader, device):
    """Return labels and softmax probabilities in data-loader order."""
    model.eval()
    all_true = []
    all_probs = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device, non_blocking=True)
            outputs = model(signals)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)
            all_true.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())
    return torch.cat(all_true, dim=0), torch.cat(all_probs, dim=0)


def metrics_from_probabilities(y_true, probabilities, num_classes=3):
    """Use the project's Meter so ensemble metrics exactly match normal evaluation."""
    y_true = torch.as_tensor(y_true, dtype=torch.long).cpu()
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32).cpu()
    if probabilities.ndim != 2 or probabilities.size(1) != int(num_classes):
        raise ValueError(f'Expected probabilities [N,{num_classes}], got {tuple(probabilities.shape)}')
    if probabilities.size(0) != y_true.numel():
        raise ValueError('Probability/label count mismatch')

    meter = Meter(n_classes=num_classes)
    meter.update({'class_probs': probabilities}, y_true)
    metrics, reports, _, true_list, pred_list = meter.get_metrics()
    return metrics, reports, true_list, pred_list
