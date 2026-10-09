import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGArrayDataset(Dataset):
    """
    全量数组高速训练版数据集
    signals: shape [N, 12, T], where T is dataset-native
    labels:  shape [N]
    """

    def __init__(self, signals, labels):
        if isinstance(signals, np.ndarray):
            self.signals = torch.from_numpy(signals).float()
        else:
            self.signals = signals.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_prepared_arrays(data_dir, mmap_mode=None):
    """
    加载预处理好的全量数组

    参数：
        data_dir: 保存 all_signals.npy / all_labels.npy 的目录
        mmap_mode:
            None   -> 直接全部载入内存（速度最快）
            'r'    -> 使用内存映射（更省内存）

    返回：
        signals: np.ndarray, shape [N, 12, T]
        labels:  np.ndarray, shape [N]
    """
    signals_path = os.path.join(data_dir, 'all_signals.npy')
    labels_path = os.path.join(data_dir, 'all_labels.npy')

    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"未找到文件: {signals_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"未找到文件: {labels_path}")

    signals = np.load(signals_path, mmap_mode=mmap_mode)
    labels = np.load(labels_path)

    return signals, labels
