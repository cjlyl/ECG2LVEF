# EXP040 — Late-Start EMA + Global Raw/EMA Ensemble

## Purpose

EXP040 is a controlled one-variable follow-up to EXP038-B.

The **prediction model and Raw training path are unchanged** from EXP038-B / EXP025:

- exact DisMS temporal branch;
- exact `BatchNorm2d + BatchNorm1d` normalization;
- exact native sampling grids (`private_data`: 500 Hz / 5000 points; `echonext`: 250 Hz / 2500 points);
- exact image branch;
- direct `Temporal 1024 + Image 64 -> 1088` concat;
- exact symmetric cosine classifier;
- exact EXP025 loss;
- AdamW, fixed LR `1e-3`, weight decay `1e-4`;
- batch size 16, seed 42;
- RNG-isolated extra EMA validation;
- `min_epochs=30` and robust non-finite batch handling;
- validation-only global Raw/EMA probability ensemble.

### Only new variable

EXP038-B started EMA from initialization. EXP040 uses **late-start EMA**:

```text
Epochs 1-5:
  train Raw only
  EMA does not exist

Immediately before epoch 6 training:
  EMA <- current Raw full state
  (parameters + floating BN running statistics)

Epoch 6 onward:
  optimizer.step()
  EMA.update(Raw), decay = 0.999
```

The start epoch is deliberately aligned with the existing loss setting:
`ambiguity_warmup_epochs=5`, so auxiliary ambiguity/compactness terms become active when `criterion.set_epoch(5)` is used for the sixth training epoch.

## Directory

Recommended:

```bash
/mnt/storage/syy/ECG_code/github_LSTM-CNN/EXP040_LateStartEMA_GlobalEnsemble
```

Existing data directories are reused:

```text
/mnt/storage/syy/ECG_code/github_LSTM-CNN/prepared_data
/mnt/storage/syy/ECG_code/github_LSTM-CNN/prepared_data_EchoNext_2500
```

No data regeneration is required.

## Preflight

```bash
cd /mnt/storage/syy/ECG_code/github_LSTM-CNN/EXP040_LateStartEMA_GlobalEnsemble
/opt/anaconda3/envs/ecg_fl/bin/python validate_exp040.py
```

Expected key lines:

```text
EXP040 preflight: PASS
Exact EXP025 parameter count: 1368941
Core model/loss/data hashes: identical to EXP038-B
Epochs 1-5: Raw only; EMA does not exist
Before epoch 6: EMA initialized from current Raw full state
Epoch 6 onward: EMA decay=0.999 after each valid optimizer step
```

## Run EchoNext

```bash
bash run_exp040.sh echonext 0 EXP040_LateStartEMA_GlobalEnsemble_echonext
```

## Run private_data

```bash
bash run_exp040.sh private_data 0 EXP040_LateStartEMA_GlobalEnsemble_private
```

## Run both sequentially

```bash
bash run_both_exp040.sh 0 0
```

With two GPUs, launch the two `run_exp040.sh` commands in separate terminals.

## What the log should look like

Epochs 1-5 explicitly show:

```text
EMA=INACTIVE(start_epoch=6)
```

Before epoch 6:

```text
[EMA-INIT] Before epoch 6: initialized full EMA state from current Raw model; decay=0.999.
```

From epoch 6 onward each epoch reports both Raw and EMA validation metrics.

## Checkpoint selection

Raw and late-start EMA have independent validation tracks.

Selection rule for each track:

```text
Macro-F1 first; tolerance = 0.001
within tolerance: Accuracy -> AUC -> lower MAE -> higher F1 -> earlier epoch
```

Test is never used to choose a checkpoint.

## Global ensemble

After training, the independently selected Raw and EMA checkpoints are evaluated on validation and the program searches:

```text
p = alpha * p_raw + (1-alpha) * p_ema
alpha in {0, 0.25, 0.50, 0.75, 1.00}
```

Alpha is selected on validation only using the same `Macro-F1 first + 0.001 tolerance` rule, then frozen and applied once to test.

EXP039 class-conditional alpha is **not** used in EXP040.

## Final outputs

The program prints:

```text
Central-Val-Best-Raw
Central-Val-Best-EMA
Central-Val-Best-Ensemble

Central-Test-Best-Raw
Central-Test-Best-EMA
Central-Test-Best-Ensemble
```

and final deltas versus EXP025.

Reference baselines:

```text
EchoNext:
Acc 0.6233 | Macro-F1 0.5760 | AUC 0.7764 | MAE 0.4536

private_data:
Acc 0.7318 | Macro-F1 0.6892 | AUC 0.8828 | MAE 0.2932
```

## Main diagnostic for EXP040

Raw is expected to reproduce EXP025 / EXP038-B. The key question is whether late-start EMA retains the C0/C1 improvement seen on EchoNext while reducing the large C2-recall loss caused by EMA-from-initialization.

For EchoNext, compare especially:

```text
C0 recall
C1 recall
C2 recall
Accuracy
Macro-F1
```

For private_data, verify that the successful EXP038-B global-ensemble behavior is not lost.
