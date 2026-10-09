import os
import random
import numpy as np
import torch


class Config:
    """EXP040: exact EXP025 + late-start EMA + validation-selected global ensemble.

    Everything in the prediction/training recipe stays identical to EXP038-B /
    EXP025 except the EMA start time. Epochs 1-5 train Raw only. Immediately
    before epoch 6 training, EMA is initialized from the current Raw full state
    (parameters + floating BN running statistics), then updated with decay 0.999.
    Extra EMA validation remains RNG-isolated and test never selects checkpoints
    or ensemble alpha.
    """

    def __init__(
        self,
        dataset_name='private_data',
        cuda=0,
        log_name='EXP040_LateStartEMA_GlobalEnsemble',
    ):
        self.model_name = 'ECGStageA2EXP040_LateStartEMA'
        self.num_classes = 3

        # ============================================================
        # EXACT EXP025 optimization recipe
        # ============================================================
        self.lr = 1e-3
        self.epochs = 50
        self.batch_size = 16
        self.weight_decay = 1e-4
        self.patience = 10
        # Do not early-stop before the original private-data optimum region.
        self.min_epochs = 30
        self.grad_max_norm = 1.0
        self.test_size = 0.2
        self.val_size = 0.1

        # EXP040 only new variable: late-start EMA.
        # 1-based epoch. Epochs 1-5 are Raw-only; EMA is initialized immediately
        # before epoch 6 training from the current Raw full state.
        self.ema_start_epoch = 6
        self.ema_decay = 0.999

        # Macro-F1 first; within tolerance: Acc -> AUC -> lower MAE.
        self.selection_f1_tolerance = 0.001
        # Validation-only probability ensemble: p = alpha*Raw + (1-alpha)*EMA.
        self.ensemble_alphas = (0.0, 0.25, 0.50, 0.75, 1.0)

        # ============================================================
        # EXACT EXP025 DisMS temporal branch
        # ============================================================
        self.lstm_hidden = 128
        self.lstm_layers = 2
        self.dropout = 0.2
        self.disms_down_sampling_layers = 3
        self.disms_down_sampling_window = 2
        self.disms_temporal_channels = 128
        self.disms_temporal_size = 128
        self.disms_output_dim = 1024
        self.disms_similarity_loss_weight = 0.1
        self.disms_disparity_loss_weight = 0.1
        self.disms_gru_chunk_size = 24
        self.disms_use_gru_checkpoint = True

        # ============================================================
        # EXACT EXP025 cosine classifier + loss
        # ============================================================
        self.cosine_feature_dim = 128
        self.cosine_temperature = 16.0
        self.ambiguity_margin = 0.20
        self.ambiguity_temperature = 0.10
        self.ambiguity_loss_weight = 0.20
        self.compactness_loss_weight = 0.05
        self.ambiguity_warmup_epochs = 5
        # Skip isolated non-finite batches; stop only if an epoch is clearly unstable.
        self.max_nonfinite_batches_per_epoch = 5

        # ============================================================
        # EXACT EXP025 image branch
        # ============================================================
        self.image_input_points = 256
        self.image_time_bins = 128
        self.image_hidden_dim = 64
        self.image_dropout = 0.1
        self.image_channel_preset = 'LEGACY_WAVEFORM_CORE4_TIME1'
        self.image_channels = 5
        self.image_channel_names = (
            'ValueMap',
            'SlopeShortMap',
            'SlopeLongMap',
            'VarianceMap',
            'ExplicitTimeCoordinate',
        )
        self.image_time_coordinate_min = -1.0
        self.image_time_coordinate_max = 1.0

        # ============================================================
        # Reproducibility / device
        # ============================================================
        self.seed = 42
        self.cuda = cuda
        if torch.cuda.is_available() and cuda is not None and cuda >= 0:
            if cuda >= torch.cuda.device_count():
                raise ValueError(
                    f'Requested cuda:{cuda}, but only '
                    f'{torch.cuda.device_count()} CUDA device(s) are available.'
                )
            self.device = torch.device(f'cuda:{cuda}')
        else:
            self.device = torch.device('cpu')
        self.num_workers = 4
        self.pin_memory = torch.cuda.is_available()

        # ============================================================
        # Existing project paths / EXACT native sampling grids
        # ============================================================
        self.base_path = '/mnt/storage/syy/ECG_code/github_LSTM-CNN'
        self.experiment_output_name = log_name
        self.log_dir = os.path.join(self.base_path, 'log', log_name)
        self.save_dir = os.path.join(self.base_path, 'checkpoints', log_name)
        self.log_file = os.path.join(self.log_dir, f'{log_name}.log')

        self.prepared_data_dirs = {
            'private_data': os.path.join(self.base_path, 'prepared_data'),
            'echonext': os.path.join(self.base_path, 'prepared_data_EchoNext_2500'),
        }
        self.dataset_display_names = {
            'private_data': 'Private Data',
            'echonext': 'EchoNext',
        }
        self.dataset_input_lengths = {
            'private_data': 5000,
            'echonext': 2500,
        }
        self.dataset_sampling_rates_hz = {
            'private_data': 500,
            'echonext': 250,
        }

        self.dataset_name = dataset_name.strip().lower()
        if self.dataset_name not in self.prepared_data_dirs:
            raise ValueError(
                f'Unknown dataset_name={self.dataset_name!r}; '
                f'available={sorted(self.prepared_data_dirs)}'
            )
        self.dataset_display_name = self.dataset_display_names[self.dataset_name]
        self.prepared_data_dir = self.prepared_data_dirs[self.dataset_name]
        self.expected_input_length = self.dataset_input_lengths[self.dataset_name]
        self.sampling_rate_hz = self.dataset_sampling_rates_hz[self.dataset_name]

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def __repr__(self):
        return (
            f'Config(model={self.model_name}, dataset={self.dataset_name}, '
            f'lr={self.lr}, ema={self.ema_decay}, device={self.device})'
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
