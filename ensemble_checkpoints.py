"""Validation-select Raw/EMA probability ensemble from existing checkpoints.

This script does NOT retrain. It recreates the fixed EXP025 split, searches alpha
only on validation, then applies the chosen alpha once to test.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import Config, seed_everything
from dataset import ECGArrayDataset, load_prepared_arrays
from main import _ensemble_table, _select_ensemble_alpha
from trainer import build_model, metrics_from_probabilities, predict_probabilities
from utils import print_and_log_report


def parse_args():
    p = argparse.ArgumentParser('Validation-selected Raw/EMA checkpoint ensemble')
    p.add_argument('--dataset_name', choices=['private_data', 'echonext'], required=True)
    p.add_argument('--raw_ckpt', required=True)
    p.add_argument('--ema_ckpt', required=True)
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--log_name', default='EXP040_ExistingCheckpointEnsemble')
    return p.parse_args()


def loader(signals, labels, idx, cfg):
    return DataLoader(
        ECGArrayDataset(signals[idx], labels[idx]),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def load_model(cfg, path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(path, map_location=cfg.device), strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    cfg = Config(args.dataset_name, args.cuda, args.log_name)
    seed_everything(cfg.seed)

    signals, labels = load_prepared_arrays(cfg.prepared_data_dir, mmap_mode=None)
    labels = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(labels))
    train_pool_idx, test_idx, train_pool_labels, _ = train_test_split(
        indices, labels, test_size=cfg.test_size,
        random_state=cfg.seed, stratify=labels,
    )
    _, val_idx, _, _ = train_test_split(
        train_pool_idx, train_pool_labels, test_size=cfg.val_size,
        random_state=cfg.seed, stratify=train_pool_labels,
    )
    val_loader = loader(signals, labels, val_idx, cfg)
    test_loader = loader(signals, labels, test_idx, cfg)

    raw = load_model(cfg, args.raw_ckpt)
    val_true_raw, raw_val = predict_probabilities(raw, val_loader, cfg.device)
    test_true_raw, raw_test = predict_probabilities(raw, test_loader, cfg.device)
    del raw

    ema = load_model(cfg, args.ema_ckpt)
    val_true_ema, ema_val = predict_probabilities(ema, val_loader, cfg.device)
    test_true_ema, ema_test = predict_probabilities(ema, test_loader, cfg.device)
    del ema

    if not torch.equal(val_true_raw, val_true_ema):
        raise RuntimeError('Validation order mismatch')
    if not torch.equal(test_true_raw, test_true_ema):
        raise RuntimeError('Test order mismatch')

    selected, candidates = _select_ensemble_alpha(
        val_true_raw, raw_val, ema_val,
        cfg.ensemble_alphas, cfg.selection_f1_tolerance,
    )
    print(_ensemble_table(candidates))
    alpha = float(selected['alpha'])
    print(f'Selected alpha_raw={alpha:.2f}, alpha_ema={1-alpha:.2f}')

    val_probs = alpha * raw_val + (1.0 - alpha) * ema_val
    val_metrics, val_reports, val_true, val_pred = metrics_from_probabilities(
        val_true_raw, val_probs, cfg.num_classes
    )
    test_probs = alpha * raw_test + (1.0 - alpha) * ema_test
    test_metrics, test_reports, test_true, test_pred = metrics_from_probabilities(
        test_true_raw, test_probs, cfg.num_classes
    )

    log_path = os.path.join(cfg.log_dir, 'existing_checkpoint_ensemble.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(
            f'Raw={args.raw_ckpt}\nEMA={args.ema_ckpt}\n'
            f'alpha_raw={alpha:.2f}\nalpha_ema={1-alpha:.2f}\n'
            'Alpha selected on validation only; test not used for selection.\n'
        )
    print_and_log_report(
        'Existing-Checkpoint-Val-Ensemble', val_metrics, val_reports,
        val_true, val_pred, log_path,
    )
    print_and_log_report(
        'Existing-Checkpoint-Test-Ensemble', test_metrics, test_reports,
        test_true, test_pred, log_path,
    )
    print(f'Log: {log_path}')


if __name__ == '__main__':
    main()
