# -*- coding: utf-8 -*-
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from Models.disms_ts_no_fft import disms_disparity_loss, disms_similarity_loss


class StageA2PermutationInvariantAmbiguityLoss(nn.Module):
    """Symmetric class-balanced CE, ambiguity margin, and compactness."""

    def __init__(
        self,
        class_weights=None,
        disms_similarity_weight=0.1,
        disms_disparity_weight=0.1,
        ambiguity_margin=0.20,
        ambiguity_temperature=0.10,
        ambiguity_loss_weight=0.20,
        compactness_loss_weight=0.05,
        ambiguity_warmup_epochs=5,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(3),
        )
        self.classification_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.disms_similarity_weight = float(disms_similarity_weight)
        self.disms_disparity_weight = float(disms_disparity_weight)
        self.ambiguity_margin = float(ambiguity_margin)
        self.ambiguity_temperature = max(float(ambiguity_temperature), 1e-6)
        self.ambiguity_loss_weight = float(ambiguity_loss_weight)
        self.compactness_loss_weight = float(compactness_loss_weight)
        self.ambiguity_warmup_epochs = int(ambiguity_warmup_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, outputs, targets):
        weighted_ce = self.classification_loss(
            outputs["logits"],
            targets,
        )

        cosine_scores = outputs["cosine_scores"]
        true_scores = cosine_scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        rival_mask = F.one_hot(
            targets, num_classes=cosine_scores.size(1)
        ).bool()
        strongest_rival = cosine_scores.masked_fill(
            rival_mask, -torch.inf
        ).max(dim=1).values
        sample_margin = true_scores - strongest_rival

        ambiguity_weight = torch.sigmoid(
            (self.ambiguity_margin - sample_margin.detach())
            / self.ambiguity_temperature
        )
        boundary_violation = F.relu(
            self.ambiguity_margin - sample_margin
        ).square()
        sample_weights = self.class_weights[targets]
        normalizer = sample_weights.sum().clamp_min(1e-7)
        ambiguity_loss = (
            sample_weights * ambiguity_weight * boundary_violation
        ).sum() / normalizer
        compactness_loss = (
            sample_weights * (1.0 - true_scores)
        ).sum() / normalizer

        auxiliary_enabled = self.current_epoch >= self.ambiguity_warmup_epochs
        auxiliary_scale = 1.0 if auxiliary_enabled else 0.0

        shared_features = torch.nan_to_num(
            outputs["disms_shared_features"],
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).clamp(-1e4, 1e4)
        specific_features = torch.nan_to_num(
            outputs["disms_specific_features"],
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        ).clamp(-1e4, 1e4)
        similarity_loss = disms_similarity_loss(
            shared_features
        )
        disparity_loss = disms_disparity_loss(
            specific_features
        )
        loss = (
            weighted_ce
            + auxiliary_scale
            * self.ambiguity_loss_weight
            * ambiguity_loss
            + auxiliary_scale
            * self.compactness_loss_weight
            * compactness_loss
            + self.disms_similarity_weight * similarity_loss
            + self.disms_disparity_weight * disparity_loss
        )
        return loss, {
            "loss": loss.detach(),
            "weighted_ce_loss": weighted_ce.detach(),
            "ambiguity_loss": ambiguity_loss.detach(),
            "compactness_loss": compactness_loss.detach(),
            "mean_margin": sample_margin.mean().detach(),
            "ambiguous_fraction": (
                sample_margin < self.ambiguity_margin
            ).float().mean().detach(),
            "auxiliary_enabled": torch.tensor(
                auxiliary_scale,
                device=weighted_ce.device,
            ),
            "disms_similarity_loss": similarity_loss.detach(),
            "disms_disparity_loss": disparity_loss.detach(),
        }


def compute_class_weights(labels, num_classes, device):
    labels = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts[class_counts == 0] = 1.0

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32, device=device)


def _safe_filename(text):
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
    return text or "confusion_matrix"


def save_confusion_matrix_figure(
        cm,
        class_names,
        save_path,
        title="Confusion Matrix",
        normalize=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.int64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        np.maximum(row_sum, 1),
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sum != 0,
    )

    display_matrix = cm_norm if normalize else cm
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=220)
    im = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues", vmin=0)
    if normalize:
        im.set_clim(0.0, 1.0)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    threshold = display_matrix.max() * 0.55 if display_matrix.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm_norm[i, j] * 100:.1f}%\n(n={cm[i, j]})"
            else:
                text = f"{cm[i, j]}"
            color = "white" if display_matrix[i, j] > threshold else "#1f2937"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def print_and_log_report(
        fold,
        metrics,
        class_reports,
        y_true,
        y_pred,
        txt_log_path):

    line = "-" * 65
    report = f"\n{line}\n"

    if isinstance(fold, int):
        title = f"Fold {fold} Validation Result"
    elif fold == "Test":
        title = "Final Test Result"
    elif fold == "Central-Test-Last":
        title = "Central-Test-Last"
    elif fold == "Central-Test-Best":
        title = "Central-Test-Best"
    elif fold == "Final-TrainVal":
        title = "Final Train/Validation Result"
    else:
        title = str(fold)

    report += f"{title}\n"
    report += f"{line}\n"

    report += classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["EF_le_40", "EF_be_4050", "EF_ge_50"],
        digits=4,
        zero_division=0
    )

    report += "\n[Overall Metrics]\n"
    report += f"Accuracy : {metrics['Accuracy']:.4f}\n"
    report += f"Macro-F1 : {metrics['Macro-F1']:.4f}\n"
    report += f"MAE      : {metrics['MAE']:.4f}\n"
    report += f"Macro ROC-AUC (OvR): {metrics['AUC']:.4f}\n"

    report += "\n[Per-class Recall]\n"
    for cls_name, recall_val in class_reports['Per-class Recall'].items():
        report += f"{cls_name:<10}: {recall_val:.4f}\n"

    report += "\n[Per-class ROC-AUC (OvR), diagnostic only]\n"
    for cls_name, auc_val in class_reports[
        'Per-class ROC-AUC (OvR)'
    ].items():
        report += f"{cls_name:<10}: {auc_val:.4f}\n"

    labels = [0, 1, 2]
    target_names = ["EF_le_40", "EF_be_4050", "EF_ge_50"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report += "\n[Confusion Matrix]\n"
    report += "Rows=true, Cols=pred\n"
    report += f"{'':>12}" + "".join(f"{name:>12}" for name in target_names) + "\n"
    for name, row in zip(target_names, cm):
        report += f"{name:>12}" + "".join(f"{int(v):>12}" for v in row) + "\n"

    log_stem = os.path.splitext(
        os.path.basename(os.path.abspath(txt_log_path))
    )[0]
    figure_path = os.path.join(
        os.path.dirname(os.path.abspath(txt_log_path)),
        f"{log_stem}_confusion_matrix_{_safe_filename(title)}.png",
    )
    try:
        save_confusion_matrix_figure(
            cm=cm,
            class_names=["LVEF <= 40%", "40% < LVEF < 50%", "LVEF >= 50%"],
            save_path=figure_path,
            title=f"{title} Confusion Matrix",
            normalize=True,
        )
        report += "\n[Confusion Matrix Figure]\n"
        report += f"Saved: {figure_path}\n"
    except Exception as exc:
        report += "\n[Confusion Matrix Figure]\n"
        report += f"Failed to save figure: {exc}\n"

    report += f"{line}\n"

    print(report)

    with open(txt_log_path, 'a', encoding='utf-8') as f:
        f.write(report)
