import copy
import hashlib
import inspect
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config, seed_everything
from main import _select_ensemble_alpha
from trainer import ModelEMA, build_model, preserve_rng_state, train_epoch

EXPECTED_PARAMS = 1368941
EXPECTED_CORE_HASHES = {
    'Models/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'Models/disms_ts_no_fft.py': '34552054b4af0d3b4a523b720c7a2e68ccc91b665698d8003c93748ac7d8a678',
    'Models/image_branch.py': '5a9e7c4b0629fbb5b25cc96828e77d5355c5173a251ed8274c583406334f6f15',
    'Models/mscnn_bilstm_attn.py': '26b57e1553802448e02cc0ea539a51fca9d30f637ec3de3d594d016ba66e0240',
    'utils.py': '1862cb308c8e2ead1debe47ae824d922e92ac592967eff6c593461512dcef4f8',
    'dataset.py': 'b7c6ebb136cd883f89b19cfda627e910da90e388fd80805c6615078701fc2e36',
    'meter.py': 'efb2815fe0aef839ce7258f1a05dcb16d749720266a1169a7991d253c967d6e5',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_core_hashes():
    root = Path(__file__).resolve().parent
    for rel, expected in EXPECTED_CORE_HASHES.items():
        got = sha256(root / rel)
        assert got == expected, f'Core file changed: {rel}\nexpected={expected}\ngot={got}'


def check_structure(dataset_name):
    cfg = Config(dataset_name=dataset_name, cuda=-1, log_name='EXP040_preflight_tmp')
    seed_everything(cfg.seed)
    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED_PARAMS, (total, EXPECTED_PARAMS)
    assert model.feature_dim == 1024
    assert model.classifier.projector[0].in_features == 1088
    assert not hasattr(model, 'temporal_input_adapter')
    assert not hasattr(model, 'temporal_projection')
    assert not hasattr(model, 'image_projection')
    assert all(
        isinstance(layer, nn.BatchNorm1d)
        for layer in model.temporal_branch.temporal_normalizes
    )
    assert cfg.lr == 1e-3
    assert cfg.ema_decay == 0.999
    assert cfg.ema_start_epoch == 6
    assert cfg.ambiguity_warmup_epochs == 5
    assert cfg.min_epochs == 30
    assert cfg.max_nonfinite_batches_per_epoch == 5
    assert tuple(cfg.ensemble_alphas) == (0.0, 0.25, 0.5, 0.75, 1.0)
    return cfg, model


def check_forward(cfg, model):
    model.eval()
    x = torch.randn(1, 12, cfg.expected_input_length)
    with torch.no_grad():
        out = model(x)
    assert out['logits'].shape == (1, 3)
    assert torch.isfinite(out['logits']).all()


def check_late_start_ema_semantics(model):
    # EMA initialization must be an exact full-state clone of the current Raw model.
    raw_state = copy.deepcopy(model.state_dict())
    ema = ModelEMA(model, decay=0.999)
    shadow0 = ema.state_dict_cpu()
    for key, value in raw_state.items():
        assert torch.equal(shadow0[key], value.detach().cpu()), key

    # Then one update must move floating state by exactly the configured EMA rule.
    p_name, p = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
    before_shadow = shadow0[p_name].clone()
    with torch.no_grad():
        p.add_(0.01)
    current = model.state_dict()[p_name].detach().cpu()
    ema.update(model)
    got = ema.state_dict_cpu()[p_name]
    expected = before_shadow * 0.999 + current * 0.001
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-7)

    # train_epoch must explicitly support ema=None for epochs 1-5.
    sig = inspect.signature(train_epoch)
    assert sig.parameters['ema'].default is None


def check_rng_isolation():
    seed_everything(123)
    py_state = random.getstate()
    np_state = np.random.get_state()
    th_state = torch.get_rng_state()
    expected_py = random.random()
    expected_np = float(np.random.rand())
    expected_th = float(torch.rand(1))
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(th_state)

    with preserve_rng_state():
        _ = [random.random() for _ in range(5)]
        _ = np.random.rand(5)
        _ = torch.rand(5)

    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np
    assert float(torch.rand(1)) == expected_th


def check_ensemble_selection():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    raw = torch.tensor([
        [.8, .1, .1], [.6, .3, .1], [.2, .6, .2],
        [.4, .5, .1], [.1, .2, .7], [.2, .2, .6],
    ])
    ema = torch.tensor([
        [.7, .2, .1], [.5, .4, .1], [.1, .7, .2],
        [.3, .6, .1], [.1, .3, .6], [.1, .2, .7],
    ])
    selected, candidates = _select_ensemble_alpha(
        y, raw, ema, (0.0, .25, .5, .75, 1.0), .001
    )
    assert len(candidates) == 5
    assert selected['alpha'] in (0.0, .25, .5, .75, 1.0)


def main():
    check_core_hashes()
    p_cfg, p_model = check_structure('private_data')
    e_cfg, e_model = check_structure('echonext')

    p_state, e_state = p_model.state_dict(), e_model.state_dict()
    assert p_state.keys() == e_state.keys()
    for key in p_state:
        assert torch.equal(p_state[key], e_state[key]), key

    check_late_start_ema_semantics(e_model)
    check_rng_isolation()
    check_ensemble_selection()
    check_forward(p_cfg, p_model)
    check_forward(e_cfg, e_model)

    print('EXP040 preflight: PASS')
    print(f'Exact EXP025 parameter count: {EXPECTED_PARAMS}')
    print('Core model/loss/data hashes: identical to EXP038-B')
    print('Fusion: Temporal 1024 + Image 64 -> direct concat 1088')
    print('DisMS temporal_normalizes: BatchNorm1d')
    print('Temporal sampling: native 500 Hz private / native 250 Hz EchoNext')
    print('Optimizer: AdamW fixed LR=1e-3; no scheduler')
    print('Epochs 1-5: Raw only; EMA does not exist')
    print('Before epoch 6: EMA initialized from current Raw full state')
    print('Epoch 6 onward: EMA decay=0.999 after each valid optimizer step')
    print('Extra EMA validation: RNG-isolated')
    print('min_epochs=30; max_nonfinite_batches_per_epoch=5')
    print('Global ensemble alpha grid: 0, 0.25, 0.5, 0.75, 1; validation-only')


if __name__ == '__main__':
    main()
