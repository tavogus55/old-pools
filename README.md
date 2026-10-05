# Old Pools

`old-pools` is a PyTorch Geometric graph-classification and graph-regression benchmark for comparing graph pooling methods. It records accuracy/regression quality together with training, inference, memory, pooling, and throughput measurements.

## Requirements

Use the project’s `rw-env` environment with PyTorch, PyTorch Geometric, scikit-learn, NumPy, and pandas installed. The project currently targets the existing Python/PyTorch environment used for the experiments.

Run commands from the project directory:

```powershell
python main.py --dataset DD --model topk --epochs 20
```

The PowerShell batch runner is:

```powershell
.\run_experiments.ps1
```

## Command-line arguments

The following are the arguments accepted by `main.py` and their current defaults.

| Argument | Type | Default | Description |
|---|---:|---|---|
| `--dataset` | choice | `DD` | Dataset to use. See the dataset lists below. |
| `--model` | choice | `topk` | Pooling model to use. |
| `--epochs` | integer | `2000` | Maximum number of training epochs. |
| `--hidden` | integer | `32` | Hidden/node-embedding dimension. |
| `--pratio` | float | `0.5` | Pooling ratio. |
| `--lr`, `--learning-rate` | float | `1e-3` | Adam learning rate. |
| `--weight_decay`, `--weight-decay` | float | `1e-4` | Adam weight decay. |
| `--dropout` | float | `0.2` | Dropout probability. |
| `--batch-size` | integer | `128` | Batch size. |
| `--mp-layer` | choice | `graphconv` | Sparse message-passing layer: `gcn` or `graphconv`. |
| `--k_folds`, `--k-folds` | integer | `10` | Number of shuffled cross-validation folds. |
| `--seeds` | one or more strings | `42 43 44 45 46 47 48 49 50 51` | Seeds used for every fold. Comma-separated values are also accepted. |
| `--log_level` | choice | `info` | Logging level: `debug`, `info`, `warning`, `error`, or `critical`. |
| `--log_path` | string | `local` | Log-path setting. |
| `--exp_name` | string | `exp` | Experiment name; results are written to `results/<exp_name>.csv`. |
| `--tolerance` | float | `1e-4` | Minimum validation improvement used by early stopping. |
| `--early_stop` | integer | `50` | Number of non-improving validation checks before stopping. |
| `--cuda` | flag | `False` | CUDA option accepted by the CLI. The current runtime uses CUDA when PyTorch reports it as available. |
| `--ddp` | flag | `False` | Distributed-training option accepted by the CLI; distributed execution is not currently implemented. |

### Seed examples

Use the default ten seeds:

```powershell
python main.py --dataset DD --model count1
```

Use two custom seeds:

```powershell
python main.py --dataset DD --model count1 --seeds 42,43
```

The following is equivalent:

```powershell
python main.py --dataset DD --model count1 --seeds 42 43
```

## Supported datasets

### Graph classification

```text
PROTEINS
DD
IMDB-MULTI
IMDB-BINARY
MUTAG
NCI1
NCI109
COLLAB
ENZYMES
PTC_MR
AIDS
MUTAGENICITY
REDDIT-BINARY
REDDIT-MULTI-5K
BZR
COX2
DHFR
MSRC_9
MSRC_21
COIL-DEL
Synthie
```

### Graph regression

```text
ESOL
FreeSolv
lipo
QM7
QM8
BACE
QM7b
```

Regression datasets use MSE, RMSE, and MAE rather than classification accuracy and F1 scores. `BACE` is configured as a regression dataset in this project.

## Supported pooling models

| Argument/model name | Pooling method | Operation type |
|---|---|---|
| **Node Drop Pooling** |  |  |
| `topk` | TopKPool | Node dropping |
| `sag` | SAGPool | Node dropping |
| `asapool` | ASAPooling | Node dropping |
| `pan` | PANPool | Node dropping |
| `cop` | COPool | Node dropping |
| `cgi` | CGIPool | Node dropping |
| `kmis` | KMISPool | Node dropping |
| `gsap` | GSAPool | Node dropping |
| `hgpsl` | HGPSLPool | Node dropping |
| `hdpsl` | HDPSLPool | Node dropping |
| `ndrp` | Node-drop random pooling | Node dropping |
| `ndp` | Node decimation pooling | Node dropping |
| **Node Clustering Pooling** |  |  |
| `diff` | DiffPool | Node clustering |
| `mincut` | MinCutPool | Node clustering |
| `dmon` | DMoN pooling | Node clustering |
| `hosc` | HOSC pooling | Node clustering |
| `justb` | JustBalance pooling | Node clustering |
| `graclus` | Graclus pooling | Node clustering |
| `pars` | Graph parsing pooling | Node clustering |
| `gaus` | Gaussian random projection pooling | Node clustering |
| `unif` | Uniform random projection pooling | Node clustering |
| `count1` | Sparse CountSketch, q=1 | Node clustering |
| `count2` | Sparse CountSketch, q=2 | Node clustering |
| `count4` | Sparse CountSketch, q=4 | Node clustering |

`count1`, `count2`, and `count4` use the sparse CountSketch command-line path. Gaussian and Uniform use their dense pooled representation, with sparse graph input where supported by the current hybrid implementation.

## Dataset-size filtering

Every experiment applies the dataset-specific `max_nodes` limit before creating the train/validation/test folds. This keeps all methods in an experiment on the same graph subset and makes dense memory requirements manageable. The selected limit is logged at the beginning of each run.

## Splits and early stopping

The experiment creates shuffled K-fold splits with a fixed splitter seed of `42`. For each fold, 10% of the training portion is used as validation data. The requested training limit is `--epochs`; regression runs can stop earlier when validation performance does not improve by `--tolerance` for `--early_stop` checks. The CSV records the actual number of completed epochs.

## Metrics and output files

Logs are written under `logs/<exp_name>/`. Results are written to `results/<exp_name>.csv`.

Classification results include:

- Accuracy
- Micro-F1
- Macro-F1

Regression results include:

- MSE
- RMSE
- MAE

Efficiency fields include:

- End-to-end time
- Training time per epoch
- Total completed training epochs
- Total training time
- Test inference time
- Peak training GPU memory
- Peak inference GPU memory
- Pooling time
- Test throughput in graphs per second
- Preprocessing time

## Example commands

Classification with CountSketch1:

```powershell
python main.py `
    --dataset PROTEINS `
    --model count1 `
    --epochs 50 `
    --k-folds 2 `
    --seeds 42,43 `
    --exp_name proteins-count1
```

Regression with Gaussian pooling:

```powershell
python main.py `
    --dataset ESOL `
    --model gaus `
    --epochs 50 `
    --k-folds 2 `
    --seeds 42 `
    --exp_name esol-gaus
```
