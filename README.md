# Global daily chlorophyll-a with per-pixel uncertainty from a Bayesian neural network

This repository contains the code used to train the Bayesian neural network (BNN) and to generate the global daily chlorophyll-a (Chl-a) dataset described in:

> Papagiannopoulos, N., Hoteit, I. & Zhan, P. Global daily Chlorophyll-*a* retrievals with per-pixel uncertainty from a Bayesian Neural Network. *Scientific Data* (submitted).

The dataset is available at 10.5281/zenodo.23416584.

## Overview

A heteroscedastic BNN maps remote sensing reflectance (Rrs) at 443, 490, 510 and 560 nm to a predictive distribution of log10(Chl-a). The network has a shared two-layer backbone and separate heads for the predictive mean and the aleatoric variance, Gaussian priors on all weights, and a low-rank multivariate normal variational posterior fitted by stochastic variational inference with the Beta-NLL objective. For every retrieval, the predictive variance is decomposed into epistemic and aleatoric components with the law of total variance, using Monte Carlo samples from the posterior.

The workflow has two steps:

1. `train.py` fits the BNN on the in situ Rrs / Chl-a database and writes the trained model to `model/`.
2. `predict_global.py` applies the trained model to the ESA OC-CCI v6.0 daily Rrs product and writes one NetCDF file per year.

The trained model used for the published dataset is available with the dataset release (10.5281/zenodo.23416584); `train.py` writes an equivalent `model/` folder, so step 1 is only needed to reproduce the training or to retrain on another database.

## Repository contents

| Path | Description |
|---|---|
| `bnn_model.py` | Network architecture, hyperparameters, preprocessing and posterior predictive inference |
| `data_utils.py` | Loading, validation and splitting of the in situ database |
| `oc_cci.py` | Date handling, grid orientation and block averaging for the OC-CCI input |
| `train.py` | Training, early stopping and evaluation on the test partition |
| `predict_global.py` | Generation of the global dataset from OC-CCI v6.0 Rrs |
| `training_points/` | In situ training database: `rrs_in.npy` and `chl_out.npy` |
| `pyproject.toml` | Package metadata, dependencies and pytest/ruff configuration |
| `tests/` | Offline test suite (`pytest`) |
| `requirements.txt` | Python dependencies |
| `model/` | Written by `train.py`: trained model and standardisation statistics |

## Installation

The code requires Python 3.10 or later.

```bash
git clone <repository URL>
cd <repository name>
pip install -r requirements.txt
```

The repository is also installable as a package (`pip install .`), and the test
suite is available as an extra (`pip install -e ".[dev]"`). The workflow scripts are run from
the checkout.

Inference runs on CPU; no GPU is required.

## Input data

**In situ training database.** `training_points/rrs_in.npy` holds Rrs (sr⁻¹) at 443, 490, 510 and 560 nm as an array of shape (N, 4), and `training_points/chl_out.npy` holds the corresponding Chl-a concentration (mg m⁻³) as an array of shape (N, 1), for N = 3,350 records compiled from the sources described in the paper (Valente et al., 2022, *Earth Syst. Sci. Data* 14, 5737–5770). The two files are paired record by record, so they must hold the same number of records; `train.py --check-data` validates and summarises them.

**OC-CCI v6.0 Rrs.** The global dataset is generated from the ESA Ocean Colour Climate Change Initiative v6.0 merged daily Rrs product on the 4 km (1/24°) geographic grid. The daily files (for example `ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210101-fv6.0.nc`) can be downloaded from the CEDA archive at <https://data.ceda.ac.uk/neodc/esacci/ocean_colour/data/v6.0-release/geographic/netcdf/rrs/> or from <https://www.oceancolour.org>. The script reads the variables `Rrs_443`, `Rrs_490`, `Rrs_510` and `Rrs_560`; other variables in the files are ignored.

## Usage

### 1. Train the model

```bash
python train.py --rrs training_points/rrs_in.npy --chl training_points/chl_out.npy --out model
```

`--check-data` validates the database without training (it prints the record counts, the usable
records, the Chl-a range and the per-band log10 Rrs statistics, and needs only NumPy):

```bash
python train.py --rrs training_points/rrs_in.npy --chl training_points/chl_out.npy --check-data
```

The records are shuffled with a fixed seed and split into training (70 %), validation (15 %) and test (15 %) partitions. Input standardisation statistics are computed from the training partition only. Training stops when the validation negative log-likelihood has not improved for 10 consecutive checks (one check every 1,000 steps, at most 100,000 steps), and the parameters with the lowest validation NLL are kept. The selected model is then evaluated once on the test partition.

Outputs written to `model/`:

| File | Content |
|---|---|
| `bnn_best_checkpoint.pt` | Variational parameters of the selected model |
| `x_train_mu.npy`, `x_train_std.npy` | Mean and standard deviation of log10 Rrs in the training partition |
| `training_curve.csv` | ELBO per sample and validation NLL during training |
| `test_metrics.json` | RMSE, bias, R², 95 % interval coverage and mean uncertainties on the test partition |
| `test_predictions.csv` | One row per test record: in situ and predicted log10(Chl-a) and the epistemic, aleatoric and total predictive standard deviations |

### 2. Generate the global dataset

```bash
python predict_global.py --input-dir /path/to/occci_rrs_daily --output-dir output --years 2000-2025
```

The input folder is searched recursively for daily files. For each day, the four Rrs bands are averaged in linear space over the valid 4 km pixels of non-overlapping 6 × 6 blocks (a pixel is valid when all four bands are valid), and the BNN is applied to each block-mean spectrum with 100 posterior samples. This gives the 0.25° product of the published dataset.

To generate the dataset at the native 4 km resolution, run the BNN on every pixel with `--bin 1`:

```bash
python predict_global.py --input-dir /path/to/occci_rrs_daily --output-dir output_4km --years 2000-2025 --bin 1
```

Other options are listed with `python predict_global.py --help`:

- `--list-days` prints the plan without running anything: the daily files found, the output grid,
  the days per year and how many of them are already in the yearly files. It does not read the
  trained model, so it can be used to check a large input archive first.
- `--start-date` / `--end-date` restrict the run to an inclusive `YYYY-MM-DD` window, which is
  useful to test the chain on a few days or to backfill a gap. When `--years` is omitted, the
  window defines the years to process.
- `--overwrite` recomputes the days already present in a yearly file instead of skipping them.
- `--mc-seed` sets the Monte Carlo seed of the posterior sampling (default 42, the seed of the
  published dataset).

If a run is interrupted, running the same command again continues from the last day written. A
yearly file is only extended when its grid, block size, minimum number of valid pixels and number
of posterior samples match the ones requested, so a single file never mixes two processing chains.

## Output format

One NetCDF-4 file is written per year, named `BNN_OC-CCI_chl_uncertainty_0.25deg_YYYY.nc` (or `..._4km_YYYY.nc` with `--bin 1`), with dimensions `time` × `lat` × `lon` and one time record per day with input data.

| Variable | Units | Description |
|---|---|---|
| `time` | days since 1970-01-01 | Date |
| `lat`, `lon` | degrees north, degrees east | Cell centre |
| `chl_mean` | log10(mg m⁻³) | Posterior predictive mean of log10(Chl-a) |
| `unc_epistemic` | log10(mg m⁻³) | Epistemic uncertainty (1σ) |
| `unc_aleatoric` | log10(mg m⁻³) | Aleatoric uncertainty (1σ) |
| `n_valid` | 1 | Number of valid 4 km pixels averaged in the cell |

Chl-a in mg m⁻³ is `10**chl_mean`. The total predictive uncertainty is `sqrt(unc_epistemic**2 + unc_aleatoric**2)`, and the 95 % predictive interval is `chl_mean ± 1.96 × total` in log10 space. Cells without a retrieval are set to the fill value 9.96921e+36.

## Testing

The test suite runs offline (no network, no OC-CCI download, no GPU) and covers the data
contract, the grid and date helpers, the uncertainty decomposition and both command line tools
end to end on synthetic data:

```bash
pip install -e ".[dev]"
python -m pytest tests
```

Tests that need PyTorch, Pyro or netCDF4 are skipped when the corresponding package is missing.

## Reproducibility and computational requirements

All random seeds are fixed (seed 1 for the data split and training, seed 42 for posterior sampling at inference), so repeated runs with the same software versions on the same hardware give identical results. Results can differ slightly between PyTorch and Pyro versions.

## Citation

If you use this code, please cite the paper above and the code release:

> Papagiannopoulos, N., Hoteit, I. & Zhan, P. Code for: Global daily Chlorophyll-*a* retrievals with per-pixel uncertainty from a Bayesian Neural Network. Zenodo, 10.5281/zenodo.23416584 (2026).

Citation metadata is also provided in `CITATION.cff`.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The dataset is distributed separately under CC-BY 4.0.

## Contact

Nikolaos Papagiannopoulos (nikpap@sustech.edu.cn)
