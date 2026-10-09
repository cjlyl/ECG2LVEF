import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config, seed_everything
from dataset import ECGArrayDataset, load_prepared_arrays
from trainer import (
    ModelEMA,
    build_model,
    clone_state_dict_cpu,
    evaluate_epoch,
    make_ema_eval_model,
    metrics_from_probabilities,
    predict_probabilities,
    preserve_rng_state,
    train_epoch,
)
from utils import (
    StageA2PermutationInvariantAmbiguityLoss,
    compute_class_weights,
    print_and_log_report,
)


BASELINES = {
    'echonext': {
        'Accuracy': 0.6233,
        'Macro-F1': 0.5760,
        'AUC': 0.7764,
        'MAE': 0.4536,
    },
    'private_data': {
        'Accuracy': 0.7318,
        'Macro-F1': 0.6892,
        'AUC': 0.8828,
        'MAE': 0.2932,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        'EXP040: EXP025 + late-start EMA + validation-selected global Raw/EMA ensemble'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='private_data',
        choices=['private_data', 'echonext']
    )
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument(
        '--log_name', type=str,
        default='EXP040_LateStartEMA_GlobalEnsemble'
    )
    return parser.parse_args()


def _class_distribution(labels):
    return dict(pd.Series(labels).value_counts().sort_index())


def _make_loader(signals, labels, indices, config, shuffle):
    return DataLoader(
        ECGArrayDataset(signals[indices], labels[indices]),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _metric_auc_for_sort(value):
    value = float(value)
    return -float('inf') if math.isnan(value) else value


def _candidate_sort_key(candidate):
    m = candidate['metrics']
    return (
        float(m['Accuracy']),
        _metric_auc_for_sort(m['AUC']),
        -float(m['MAE']),
        float(m['Macro-F1']),
        -int(candidate['epoch']),
    )


def _update_track(track, candidate, tolerance):
    """Macro-F1-first checkpoint selection with a fixed near-tie tolerance."""
    f1 = float(candidate['metrics']['Macro-F1'])
    previous_selected_epoch = (
        track['selected']['epoch'] if track['selected'] is not None else None
    )
    previous_max_f1 = float(track['max_f1'])

    if f1 > track['max_f1']:
        track['max_f1'] = f1

    threshold = track['max_f1'] - float(tolerance)
    track['candidates'] = [
        c for c in track['candidates']
        if float(c['metrics']['Macro-F1']) >= threshold - 1e-12
    ]
    if f1 >= threshold - 1e-12:
        track['candidates'].append(candidate)

    if not track['candidates']:
        raise RuntimeError('No eligible checkpoint candidate remained')

    track['selected'] = max(track['candidates'], key=_candidate_sort_key)
    selected_changed = track['selected']['epoch'] != previous_selected_epoch
    max_f1_improved = track['max_f1'] > previous_max_f1 + 1e-12
    return selected_changed, max_f1_improved


def _compare_to_baseline(metrics, dataset_name):
    b = BASELINES[dataset_name]
    delta = {
        'Accuracy': float(metrics['Accuracy']) - b['Accuracy'],
        'Macro-F1': float(metrics['Macro-F1']) - b['Macro-F1'],
        'AUC': float(metrics['AUC']) - b['AUC'],
        'MAE': float(metrics['MAE']) - b['MAE'],
    }
    beat_all = (
        delta['Accuracy'] > 0
        and delta['Macro-F1'] > 0
        and delta['AUC'] > 0
        and delta['MAE'] < 0
    )
    return delta, beat_all


def _delta_line(name, metrics, dataset_name):
    d, beat = _compare_to_baseline(metrics, dataset_name)
    return (
        f'{name} Delta vs EXP025: '
        f'dF1={d["Macro-F1"]:+.4f}, '
        f'dAcc={d["Accuracy"]:+.4f}, '
        f'dAUC={d["AUC"]:+.4f}, '
        f'dMAE={d["MAE"]:+.4f} | '
        f'Beat all four? {beat}'
    )


def _pair_delta_line(name_a, metrics_a, name_b, metrics_b):
    return (
        f'{name_a} - {name_b}: '
        f'dF1={metrics_a["Macro-F1"] - metrics_b["Macro-F1"]:+.4f}, '
        f'dAcc={metrics_a["Accuracy"] - metrics_b["Accuracy"]:+.4f}, '
        f'dAUC={metrics_a["AUC"] - metrics_b["AUC"]:+.4f}, '
        f'dMAE={metrics_a["MAE"] - metrics_b["MAE"]:+.4f}'
    )


def _select_ensemble_alpha(y_true, raw_probs, ema_probs, alphas, tolerance):
    """Validation-only alpha selection.

    p(alpha) = alpha * p_raw + (1-alpha) * p_ema.
    First restrict to alpha values within tolerance of maximum Macro-F1, then
    choose Accuracy -> AUC -> lower MAE -> higher F1. A final exact tie prefers
    alpha closest to 0.5 (the genuine ensemble) only for determinism.
    """
    candidates = []
    for alpha in alphas:
        alpha = float(alpha)
        probs = alpha * raw_probs + (1.0 - alpha) * ema_probs
        metrics, reports, true_list, pred_list = metrics_from_probabilities(
            y_true, probs, num_classes=raw_probs.size(1)
        )
        candidates.append({
            'alpha': alpha,
            'metrics': metrics,
            'reports': reports,
            'y_true': true_list,
            'y_pred': pred_list,
        })

    max_f1 = max(float(c['metrics']['Macro-F1']) for c in candidates)
    eligible = [
        c for c in candidates
        if float(c['metrics']['Macro-F1']) >= max_f1 - float(tolerance) - 1e-12
    ]

    def key(c):
        m = c['metrics']
        return (
            float(m['Accuracy']),
            _metric_auc_for_sort(m['AUC']),
            -float(m['MAE']),
            float(m['Macro-F1']),
            -abs(float(c['alpha']) - 0.5),
        )

    selected = max(eligible, key=key)
    return selected, candidates


def _ensemble_table(candidates):
    lines = [
        'Validation Raw/EMA probability ensemble search',
        'alpha = Raw weight; (1-alpha) = EMA weight',
        'alpha    Macro-F1   Accuracy      AUC      MAE',
    ]
    for c in candidates:
        m = c['metrics']
        lines.append(
            f'{c["alpha"]:>5.2f}    {m["Macro-F1"]:>8.4f}   '
            f'{m["Accuracy"]:>8.4f}   {m["AUC"]:>7.4f}   {m["MAE"]:>7.4f}'
        )
    return '\n'.join(lines) + '\n'


def _load_model_for_checkpoint(config, checkpoint_path):
    model = build_model(config).to(config.device)
    state = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    config = Config(
        dataset_name=args.dataset_name,
        cuda=args.cuda,
        log_name=args.log_name,
    )
    seed_everything(config.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_log_path = os.path.join(
        config.log_dir,
        f'{config.model_name}_{config.dataset_name}_{timestamp}.txt'
    )
    csv_log_path = os.path.join(
        config.log_dir,
        f'{config.model_name}_{config.dataset_name}_{timestamp}_epochs.csv'
    )
    best_raw_path = os.path.join(
        config.save_dir,
        f'{config.model_name}_{config.dataset_name}_{timestamp}_BEST_RAW.pth'
    )
    best_ema_path = os.path.join(
        config.save_dir,
        f'{config.model_name}_{config.dataset_name}_{timestamp}_BEST_EMA.pth'
    )
    ensemble_meta_path = os.path.join(
        config.save_dir,
        f'{config.model_name}_{config.dataset_name}_{timestamp}_ENSEMBLE.json'
    )

    header = (
        'EXP040: exact EXP025 + Late-Start EMA + validation-selected global Raw/EMA ensemble\n'
        f'Dataset={config.dataset_name} ({config.dataset_display_name})\n'
        f'NativeHz={config.sampling_rate_hz}, NativeInputPoints={config.expected_input_length}\n'
        'TemporalAdapter=None; BalancedFusion=None; ExtraBranchProjection=None\n'
        'DisMSNormalization=EXACT_EXP025 BatchNorm2d + BatchNorm1d\n'
        'Fusion=EXACT_EXP025 direct concat: Temporal1024 + Image64 -> 1088\n'
        f'LR={config.lr}, Scheduler=None, Epochs={config.epochs}, '
        f'Batch={config.batch_size}, WD={config.weight_decay}\n'
        f'EMA={config.ema_decay}; EMAStartEpoch={config.ema_start_epoch}; EMA validation RNG-isolated\n'
        f'MinEpochs={config.min_epochs}, Patience={config.patience}, '
        f'MaxNonFiniteBatchesPerEpoch={config.max_nonfinite_batches_per_epoch}\n'
        'Selection=MacroF1 first; tol=0.001; tie=Accuracy->AUC->lower_MAE\n'
        f'EnsembleAlphas={list(config.ensemble_alphas)}; validation only\n'
        'Loss=EXACT_EXP025 weighted_ce+ambiguity+compactness+DisMS_sim/dis\n'
        f'Seed={config.seed}\n'
        + '=' * 96 + '\n'
    )
    print(header, end='')
    with open(txt_log_path, 'w', encoding='utf-8') as f:
        f.write(header)

    signals, labels = load_prepared_arrays(config.prepared_data_dir, mmap_mode=None)
    labels = np.asarray(labels, dtype=np.int64)
    if signals.ndim != 3 or signals.shape[1] != 12:
        raise ValueError(f'Expected [N,12,T], got {signals.shape}')
    if signals.shape[2] != config.expected_input_length:
        raise ValueError(
            f'{config.dataset_name} expected T={config.expected_input_length}, '
            f'got {signals.shape[2]}. No interpolation/downsampling is allowed.'
        )
    if len(signals) != len(labels):
        raise ValueError('Signal/label count mismatch')
    if not np.isfinite(signals).all():
        raise ValueError('Signals contain NaN/Inf')
    if set(np.unique(labels).tolist()) != {0, 1, 2}:
        raise ValueError(f'Expected labels {{0,1,2}}, got {np.unique(labels)}')

    indices = np.arange(len(labels))
    train_pool_idx, test_idx, train_pool_labels, test_labels = train_test_split(
        indices,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    train_idx, val_idx, train_labels, val_labels = train_test_split(
        train_pool_idx,
        train_pool_labels,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=train_pool_labels,
    )

    split_summary = (
        '\nDataset split (identical rule/seed to EXP025)\n'
        + '=' * 96 + '\n'
        f'Total: {len(labels)} | {_class_distribution(labels)}\n'
        f'Train: {len(train_idx)} | {_class_distribution(train_labels)}\n'
        f'Val: {len(val_idx)} | {_class_distribution(val_labels)}\n'
        f'Test: {len(test_idx)} | {_class_distribution(test_labels)}\n'
        f'Input shape: {signals.shape}\n'
    )
    print(split_summary, end='')
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(split_summary)

    # Keep Raw validation behavior identical to EXP025. Only the extra EMA
    # validation is wrapped by preserve_rng_state().
    train_loader = _make_loader(signals, labels, train_idx, config, True)
    val_loader = _make_loader(signals, labels, val_idx, config, False)
    test_loader = _make_loader(signals, labels, test_idx, config, False)

    class_weights = compute_class_weights(
        train_labels, config.num_classes, config.device
    )
    model = build_model(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    temporal_params = sum(p.numel() for p in model.temporal_branch.parameters())
    image_params = sum(p.numel() for p in model.image_branch.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    param_summary = (
        '\nParameter summary\n' + '=' * 96 + '\n'
        f'Total parameters: {total_params}\n'
        f'Temporal branch: {temporal_params}\n'
        f'Image branch: {image_params}\n'
        f'Classifier: {classifier_params}\n'
        f'Classifier input dim: {model.classifier.projector[0].in_features}\n'
        f'Class weights: {class_weights.detach().cpu().tolist()}\n'
    )
    print(param_summary, end='')
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(param_summary)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = StageA2PermutationInvariantAmbiguityLoss(
        class_weights=class_weights,
        disms_similarity_weight=config.disms_similarity_loss_weight,
        disms_disparity_weight=config.disms_disparity_loss_weight,
        ambiguity_margin=config.ambiguity_margin,
        ambiguity_temperature=config.ambiguity_temperature,
        ambiguity_loss_weight=config.ambiguity_loss_weight,
        compactness_loss_weight=config.compactness_loss_weight,
        ambiguity_warmup_epochs=config.ambiguity_warmup_epochs,
    ).to(config.device)
    # EXP040 ONLY NEW TRAINING VARIABLE:
    # EMA does not exist during epochs 1..ema_start_epoch-1. Immediately before
    # epoch ema_start_epoch training, initialize the complete EMA state from the
    # current Raw model, then update it after every valid optimizer step.
    ema = None
    ema_initialized_before_epoch = None

    raw_track = {'max_f1': -float('inf'), 'candidates': [], 'selected': None}
    ema_track = {'max_f1': -float('inf'), 'candidates': [], 'selected': None}
    stop_counter = 0
    rows = []
    training_stop_reason = 'Normal completion'

    for epoch in range(config.epochs):
        epoch_num = epoch + 1
        criterion.set_epoch(epoch)

        # Late-start EMA: initialize from the post-epoch-(start-1) Raw state.
        # ModelEMA construction only clones tensors and consumes no RNG.
        if ema is None and epoch_num == config.ema_start_epoch:
            ema = ModelEMA(model, decay=config.ema_decay)
            ema_initialized_before_epoch = epoch_num
            init_msg = (
                f'[EMA-INIT] Before epoch {epoch_num}: initialized full EMA state '
                f'from current Raw model; decay={config.ema_decay}.\n'
            )
            print(init_msg, end='')
            with open(txt_log_path, 'a', encoding='utf-8') as f:
                f.write(init_msg)

        try:
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=config.device,
                ema=ema,
                grad_max_norm=config.grad_max_norm,
                max_nonfinite_batches=config.max_nonfinite_batches_per_epoch,
            )
        except FloatingPointError as exc:
            training_stop_reason = (
                f'Safe numerical stop before epoch {epoch_num} validation: {exc}'
            )
            break

        # RAW validation is intentionally NOT RNG-isolated: this preserves the
        # exact EXP025/EXP038-B training trajectory.
        (
            raw_val_loss,
            raw_val_metrics,
            raw_val_reports,
            raw_val_true,
            raw_val_pred,
        ) = evaluate_epoch(
            model, val_loader, criterion, config.device, config.num_classes
        )
        raw_candidate = {
            'epoch': epoch_num,
            'metrics': dict(raw_val_metrics),
            'reports': raw_val_reports,
            'y_true': list(raw_val_true),
            'y_pred': list(raw_val_pred),
            'state': clone_state_dict_cpu(model),
        }
        raw_changed, raw_f1_improved = _update_track(
            raw_track, raw_candidate, config.selection_f1_tolerance
        )
        if raw_changed:
            torch.save(raw_track['selected']['state'], best_raw_path)

        # Before the late-start epoch, EMA is genuinely inactive: no evaluation,
        # no EMA candidate, and no EMA checkpoint selection. From the start epoch
        # onward, extra EMA validation remains fully RNG-isolated.
        ema_changed = False
        ema_f1_improved = False
        ema_val_loss = None
        ema_val_metrics = None
        ema_val_reports = None
        if ema is not None:
            with preserve_rng_state():
                ema_eval_model = make_ema_eval_model(model, ema).to(config.device)
                (
                    ema_val_loss,
                    ema_val_metrics,
                    ema_val_reports,
                    ema_val_true,
                    ema_val_pred,
                ) = evaluate_epoch(
                    ema_eval_model, val_loader, criterion,
                    config.device, config.num_classes
                )
                ema_candidate = {
                    'epoch': epoch_num,
                    'metrics': dict(ema_val_metrics),
                    'reports': ema_val_reports,
                    'y_true': list(ema_val_true),
                    'y_pred': list(ema_val_pred),
                    'state': clone_state_dict_cpu(ema_eval_model),
                }
                del ema_eval_model

            ema_changed, ema_f1_improved = _update_track(
                ema_track, ema_candidate, config.selection_f1_tolerance
            )
            if ema_changed:
                torch.save(ema_track['selected']['state'], best_ema_path)

        any_progress = (
            raw_changed or raw_f1_improved or ema_changed or ema_f1_improved
        )
        stop_counter = 0 if any_progress else stop_counter + 1

        rr = raw_val_reports['Per-class Recall']
        er = (
            ema_val_reports['Per-class Recall']
            if ema_val_reports is not None else None
        )
        event_text = ' || '.join(train_loss['nonfinite_events'])
        row = {
            'epoch': epoch_num,
            'ema_active': int(ema is not None),
            'train_loss': train_loss['loss'],
            'train_ce': train_loss['weighted_ce_loss'],
            'train_ambiguity': train_loss['ambiguity_loss'],
            'train_compactness': train_loss['compactness_loss'],
            'train_mean_margin': train_loss['mean_margin'],
            'train_ambiguous_fraction': train_loss['ambiguous_fraction'],
            'train_sim': train_loss['disms_similarity_loss'],
            'train_dis': train_loss['disms_disparity_loss'],
            'train_skipped_nonfinite': train_loss['skipped_nonfinite_batches'],
            'train_overflow_fallback': train_loss['overflow_fallback_batches'],
            'nonfinite_events': event_text,
            'lr': train_loss['lr'],
            'raw_val_loss': raw_val_loss['loss'],
            'raw_val_f1': raw_val_metrics['Macro-F1'],
            'raw_val_accuracy': raw_val_metrics['Accuracy'],
            'raw_val_auc': raw_val_metrics['AUC'],
            'raw_val_mae': raw_val_metrics['MAE'],
            'raw_r0': rr['EF_le_40'],
            'raw_r1': rr['EF_be_4050'],
            'raw_r2': rr['EF_ge_50'],
            'raw_selected_epoch': raw_track['selected']['epoch'],
            'ema_val_loss': np.nan if ema_val_loss is None else ema_val_loss['loss'],
            'ema_val_f1': np.nan if ema_val_metrics is None else ema_val_metrics['Macro-F1'],
            'ema_val_accuracy': np.nan if ema_val_metrics is None else ema_val_metrics['Accuracy'],
            'ema_val_auc': np.nan if ema_val_metrics is None else ema_val_metrics['AUC'],
            'ema_val_mae': np.nan if ema_val_metrics is None else ema_val_metrics['MAE'],
            'ema_r0': np.nan if er is None else er['EF_le_40'],
            'ema_r1': np.nan if er is None else er['EF_be_4050'],
            'ema_r2': np.nan if er is None else er['EF_ge_50'],
            'ema_selected_epoch': (
                np.nan if ema_track['selected'] is None
                else ema_track['selected']['epoch']
            ),
            'stop_counter': stop_counter,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_log_path, index=False)

        raw_text = (
            f'RAW F1={raw_val_metrics["Macro-F1"]:.4f} '
            f'Acc={raw_val_metrics["Accuracy"]:.4f} '
            f'AUC={raw_val_metrics["AUC"]:.4f} '
            f'MAE={raw_val_metrics["MAE"]:.4f} '
            f'R={rr["EF_le_40"]:.3f}/{rr["EF_be_4050"]:.3f}/{rr["EF_ge_50"]:.3f} '
            f'BestRawEp={raw_track["selected"]["epoch"]}'
        )
        if ema_val_metrics is None:
            ema_text = f'EMA=INACTIVE(start_epoch={config.ema_start_epoch})'
        else:
            ema_text = (
                f'EMA F1={ema_val_metrics["Macro-F1"]:.4f} '
                f'Acc={ema_val_metrics["Accuracy"]:.4f} '
                f'AUC={ema_val_metrics["AUC"]:.4f} '
                f'MAE={ema_val_metrics["MAE"]:.4f} '
                f'R={er["EF_le_40"]:.3f}/{er["EF_be_4050"]:.3f}/{er["EF_ge_50"]:.3f} '
                f'BestEmaEp={ema_track["selected"]["epoch"]}'
            )

        line = (
            f'[Epoch {epoch_num:03d}] Train={train_loss["loss"]:.4f} '
            f'CE={train_loss["weighted_ce_loss"]:.4f} LR={train_loss["lr"]:.6f} '
            f'SkipNF={int(train_loss["skipped_nonfinite_batches"])} '
            f'FP64FB={int(train_loss["overflow_fallback_batches"])} | '
            f'{raw_text} | {ema_text} | '
            f'Stop={stop_counter}/{config.patience}; MinEp={config.min_epochs}\n'
        )
        print(line, end='')
        with open(txt_log_path, 'a', encoding='utf-8') as f:
            f.write(line)
            if event_text:
                f.write(f'[NonFiniteEvents][Epoch {epoch_num}] {event_text}\n')

        if epoch_num >= config.min_epochs and stop_counter >= config.patience:
            best_ema_epoch = (
                'NA' if ema_track['selected'] is None
                else str(ema_track['selected']['epoch'])
            )
            training_stop_reason = (
                f'Early stopping at epoch {epoch_num}; '
                f'MinEpochs={config.min_epochs}; '
                f'BestRawEpoch={raw_track["selected"]["epoch"]}; '
                f'BestEMAEpoch={best_ema_epoch}.'
            )
            break

    if raw_track['selected'] is None or ema_track['selected'] is None:
        raise RuntimeError('Training ended without both Raw and EMA checkpoints')

    raw_sel = raw_track['selected']
    ema_sel = ema_track['selected']
    print_and_log_report(
        'Central-Val-Best-Raw',
        raw_sel['metrics'], raw_sel['reports'],
        raw_sel['y_true'], raw_sel['y_pred'], txt_log_path,
    )
    print_and_log_report(
        'Central-Val-Best-EMA',
        ema_sel['metrics'], ema_sel['reports'],
        ema_sel['y_true'], ema_sel['y_pred'], txt_log_path,
    )

    # ------------------------------------------------------------------
    # Validation-only probability ensemble using the independently selected
    # Raw and EMA checkpoints. No retraining and no test-based alpha choice.
    # ------------------------------------------------------------------
    raw_best_model = _load_model_for_checkpoint(config, best_raw_path)
    val_true_raw, raw_val_probs = predict_probabilities(
        raw_best_model, val_loader, config.device
    )
    del raw_best_model

    ema_best_model = _load_model_for_checkpoint(config, best_ema_path)
    val_true_ema, ema_val_probs = predict_probabilities(
        ema_best_model, val_loader, config.device
    )
    del ema_best_model

    if not torch.equal(val_true_raw, val_true_ema):
        raise RuntimeError('Raw/EMA validation label order mismatch')

    ensemble_sel, ensemble_candidates = _select_ensemble_alpha(
        val_true_raw,
        raw_val_probs,
        ema_val_probs,
        config.ensemble_alphas,
        config.selection_f1_tolerance,
    )
    ensemble_search_text = _ensemble_table(ensemble_candidates)
    ensemble_search_text += (
        f'Selected alpha={ensemble_sel["alpha"]:.2f} '
        f'(Raw weight), EMA weight={1.0 - ensemble_sel["alpha"]:.2f}\n'
    )
    print('\n' + ensemble_search_text)
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write('\n' + ensemble_search_text)

    print_and_log_report(
        'Central-Val-Best-Ensemble',
        ensemble_sel['metrics'], ensemble_sel['reports'],
        ensemble_sel['y_true'], ensemble_sel['y_pred'], txt_log_path,
    )

    # Test Raw and get probabilities.
    raw_best_model = _load_model_for_checkpoint(config, best_raw_path)
    _, raw_test_metrics, raw_test_reports, raw_test_true, raw_test_pred = evaluate_epoch(
        raw_best_model, test_loader, criterion, config.device, config.num_classes
    )
    test_true_raw, raw_test_probs = predict_probabilities(
        raw_best_model, test_loader, config.device
    )
    print_and_log_report(
        'Central-Test-Best-Raw',
        raw_test_metrics, raw_test_reports,
        raw_test_true, raw_test_pred, txt_log_path,
    )
    del raw_best_model

    # Test EMA and get probabilities.
    ema_best_model = _load_model_for_checkpoint(config, best_ema_path)
    _, ema_test_metrics, ema_test_reports, ema_test_true, ema_test_pred = evaluate_epoch(
        ema_best_model, test_loader, criterion, config.device, config.num_classes
    )
    test_true_ema, ema_test_probs = predict_probabilities(
        ema_best_model, test_loader, config.device
    )
    print_and_log_report(
        'Central-Test-Best-EMA',
        ema_test_metrics, ema_test_reports,
        ema_test_true, ema_test_pred, txt_log_path,
    )
    del ema_best_model

    if not torch.equal(test_true_raw, test_true_ema):
        raise RuntimeError('Raw/EMA test label order mismatch')

    alpha = float(ensemble_sel['alpha'])
    ensemble_test_probs = alpha * raw_test_probs + (1.0 - alpha) * ema_test_probs
    (
        ensemble_test_metrics,
        ensemble_test_reports,
        ensemble_test_true,
        ensemble_test_pred,
    ) = metrics_from_probabilities(
        test_true_raw, ensemble_test_probs, num_classes=config.num_classes
    )
    print_and_log_report(
        'Central-Test-Best-Ensemble',
        ensemble_test_metrics, ensemble_test_reports,
        ensemble_test_true, ensemble_test_pred, txt_log_path,
    )

    metadata = {
        'experiment': 'EXP040-LateStartEMA',
        'dataset_name': config.dataset_name,
        'raw_checkpoint': best_raw_path,
        'ema_checkpoint': best_ema_path,
        'ema_start_epoch': int(config.ema_start_epoch),
        'ema_initialized_before_epoch': int(ema_initialized_before_epoch),
        'raw_best_epoch': int(raw_sel['epoch']),
        'ema_best_epoch': int(ema_sel['epoch']),
        'alpha_raw': alpha,
        'alpha_ema': 1.0 - alpha,
        'alpha_grid': [float(x) for x in config.ensemble_alphas],
        'selection_f1_tolerance': float(config.selection_f1_tolerance),
        'selection_source': 'validation_only',
        'test_used_for_selection': False,
    }
    with open(ensemble_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    raw_delta = _delta_line('RAW', raw_test_metrics, config.dataset_name)
    ema_delta = _delta_line('EMA', ema_test_metrics, config.dataset_name)
    ens_delta = _delta_line('ENSEMBLE', ensemble_test_metrics, config.dataset_name)

    finish = (
        '\n' + '=' * 96 + '\n'
        'EXP040 Late-Start EMA finished\n'
        f'Best Raw Epoch: {raw_sel["epoch"]}\n'
        f'Best EMA Epoch: {ema_sel["epoch"]}\n'
        f'EMA Start Epoch: {config.ema_start_epoch} (initialized before training this epoch)\n'
        f'Validation-selected Ensemble alpha_raw: {alpha:.2f}\n'
        f'Validation-selected Ensemble alpha_ema: {1.0 - alpha:.2f}\n'
        f'Training Stop Reason: {training_stop_reason}\n'
        f'{raw_delta}\n'
        f'{ema_delta}\n'
        f'{ens_delta}\n'
        f'{_pair_delta_line("EMA", ema_test_metrics, "RAW", raw_test_metrics)}\n'
        f'{_pair_delta_line("ENSEMBLE", ensemble_test_metrics, "RAW", raw_test_metrics)}\n'
        f'{_pair_delta_line("ENSEMBLE", ensemble_test_metrics, "EMA", ema_test_metrics)}\n'
        f'Best RAW model: {best_raw_path}\n'
        f'Best EMA model: {best_ema_path}\n'
        f'Ensemble metadata: {ensemble_meta_path}\n'
        f'Log file: {txt_log_path}\n'
        f'Epoch CSV: {csv_log_path}\n'
        + '=' * 96 + '\n'
    )
    print(finish)
    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(finish)


if __name__ == '__main__':
    main()
