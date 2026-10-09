import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)


class Meter:
    def __init__(self, n_classes=3, class_names=None):
        self.n_classes = n_classes

        if class_names is None:
            if n_classes == 3:
                class_names = ['EF_le_40', 'EF_be_4050', 'EF_ge_50']
            else:
                class_names = [f'class_{i}' for i in range(n_classes)]

        self.class_names = class_names
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_score = []

    def update(self, output, target):
        if isinstance(output, dict):
            if "class_probs" in output:
                scores = output["class_probs"]
            elif "logits" in output:
                scores = torch.softmax(output["logits"], dim=1)
            else:
                raise KeyError(
                    "Model output dict must contain logits or class_probs."
                )

            if "class_pred" in output:
                preds = output["class_pred"]
            else:
                preds = torch.argmax(scores, dim=1)
        else:
            scores = torch.softmax(output, dim=1)
            preds = torch.argmax(scores, dim=1)

        self.y_true.extend(target.detach().cpu().numpy().astype(int).tolist())
        self.y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
        self.y_score.extend(
            scores.detach().cpu().numpy().astype(float).tolist()
        )

    def _compute_auc(self, y_true, y_score):
        """Return macro OvR ROC-AUC and diagnostic per-class OvR values."""
        per_class_auc = {}
        for class_id, class_name in enumerate(self.class_names):
            binary_target = (y_true == class_id).astype(np.int64)
            try:
                value = roc_auc_score(binary_target, y_score[:, class_id])
            except ValueError:
                value = float('nan')
            per_class_auc[class_name] = float(value)

        try:
            macro_auc = roc_auc_score(
                y_true,
                y_score,
                labels=list(range(self.n_classes)),
                multi_class='ovr',
                average='macro',
            )
        except ValueError:
            macro_auc = float('nan')
        return float(macro_auc), per_class_auc

    def get_metrics(self):
        y_true = np.array(self.y_true, dtype=int)
        y_pred = np.array(self.y_pred, dtype=int)
        y_score = np.asarray(self.y_score, dtype=float)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        mae = mean_absolute_error(y_true, y_pred)
        macro_auc, per_class_auc = self._compute_auc(y_true, y_score)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))

        per_class_recall = {}
        for i in range(self.n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            per_class_recall[class_name] = recall_i

        metrics = {
            'Accuracy': acc,
            'Macro-F1': macro_f1,
            'MAE': mae,
            # The single primary AUC: multiclass macro ROC-AUC, one-vs-rest.
            'AUC': macro_auc,
        }

        class_reports = {
            'Per-class Recall': per_class_recall,
            'Per-class ROC-AUC (OvR)': per_class_auc,
        }

        return metrics, class_reports, cm, y_true.tolist(), y_pred.tolist()
