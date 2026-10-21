"""
Train the Bayesian neural network on the in situ Rrs / Chl-a database.

The database is shuffled with a fixed seed and split into training (70 %),
validation (15 %) and test (15 %) partitions. The posterior is fitted with
stochastic variational inference; early stopping uses the validation NLL, and
the final model is evaluated once on the test partition.

Outputs (in --out):
  bnn_best_checkpoint.pt   variational parameters of the selected model
  x_train_mu.npy           training mean of log10 Rrs (used for standardisation)
  x_train_std.npy          training std of log10 Rrs
  training_curve.csv       ELBO per sample and validation NLL during training
  test_metrics.json        accuracy and calibration on the test partition
  test_predictions.csv     one row per test record: in situ and predicted
                           log10(Chl-a) plus the epistemic, aleatoric and total
                           predictive standard deviations

Example:
  python train.py --rrs data/rrs_in.npy --chl data/chl_out.npy --out model
  python train.py --rrs data/rrs_in.npy --chl data/chl_out.npy --check-data
"""

import argparse
import csv
import json
import os

import numpy as np

# data_utils is NumPy-only, so --check-data (which validates the in situ
# database and exits) works without PyTorch or Pyro installed; the deep
# learning stack is imported in main() once the arguments are settled.
from data_utils import (DEFAULT_TEST_FRAC, DEFAULT_VAL_FRAC, DatasetError,
                        describe_dataset, load_dataset, split_dataset,
                        standardisation, to_log10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rrs", required=True,
                   help="in situ Rrs, .npy array (N, 4) in sr-1 at 443, 490, 510, 560 nm")
    p.add_argument("--chl", required=True,
                   help="in situ Chl-a, .npy array (N, 1) in mg m-3")
    p.add_argument("--out", default="model", help="output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--check-every", type=int, default=1_000,
                   help="SVI steps between validation-NLL evaluations")
    p.add_argument("--patience", type=int, default=10,
                   help="validation checks without improvement before stopping")
    p.add_argument("--val-samples", type=int, default=200,
                   help="posterior samples for the validation NLL")
    p.add_argument("--test-samples", type=int, default=500,
                   help="posterior samples for the test evaluation")
    p.add_argument("--check-data", action="store_true",
                   help="validate the in situ database, print a summary and exit without "
                        "training (needs only NumPy)")
    return p.parse_args()


def validate_args(args):
    """Reject argument values that cannot produce a model."""
    for flag, value in (("--max-steps", args.max_steps),
                        ("--check-every", args.check_every),
                        ("--val-samples", args.val_samples),
                        ("--test-samples", args.test_samples),
                        ("--patience", args.patience)):
        if value < 1:
            raise SystemExit(f"error: {flag} must be at least 1, got {value}")


def print_dataset_summary(rrs_path, chl_path, info):
    """Human-readable form of data_utils.describe_dataset()."""
    print(f"Rrs   : {rrs_path}  ({info['n_records']} records x {info['n_bands']} bands "
          f"{'/'.join(str(b) for b in info['bands_nm'])} nm)")
    print(f"Chl-a : {chl_path}")
    print(f"usable for training: {info['n_usable']} of {info['n_records']} "
          f"(dropped {info['n_dropped']})")
    if info["n_invalid_chl"]:
        print(f"  non-positive or non-finite Chl-a: {info['n_invalid_chl']} records")
    if info["n_nonfinite_rrs"]:
        print(f"  non-finite Rrs band: {info['n_nonfinite_rrs']} records")
    if "chl_min" in info:
        print(f"Chl-a range      : {info['chl_min']:.4g} .. {info['chl_max']:.4g} mg m-3")
    if "log10_rrs_min" in info:
        print(f"log10(Rrs) range : {info['log10_rrs_min']:.3f} .. {info['log10_rrs_max']:.3f}")
        print(f"log10(Rrs) mean  : {info['log10_rrs_mean']}")
        print(f"log10(Rrs) std   : {info['log10_rrs_std']}")


def main():
    args = parse_args()
    validate_args(args)

    # -- Data (NumPy only, so --check-data needs no deep-learning stack) ----
    try:
        rrs, chl = load_dataset(args.rrs, args.chl)
        info = describe_dataset(rrs, chl)
    except DatasetError as exc:
        raise SystemExit(f"error: {exc}") from None

    if args.check_data:
        print_dataset_summary(args.rrs, args.chl, info)
        if info["n_usable"] < 3:
            raise SystemExit(f"error: only {info['n_usable']} usable records: at least 3 are "
                             "needed for the train / validation / test split")
        print(f"\n{info['n_usable']} usable records: the database matches the documented "
              "format.")
        return

    import pyro
    import pyro.distributions as dist
    import torch
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import ClippedAdam

    from bnn_model import (CHECKPOINT_FILE, X_MEAN_FILE, X_STD_FILE,
                           build_model_and_guide, predictive_moments, restore_params,
                           snapshot_params)

    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # -- Data: shuffle with the fixed seed, then drop unusable records ------
    perm = np.random.permutation(len(rrs))
    rrs, chl = rrs[perm, :], chl[perm]
    x_all, y_all, usable = to_log10(rrs, chl)
    x_all, y_all = x_all[usable], y_all[usable]

    # -- Split: [ test | train | validation ] of the shuffled rows ---------
    try:
        split = split_dataset(len(x_all), args.val_frac, args.test_frac)
    except DatasetError as exc:
        raise SystemExit(f"error: {exc}") from None
    x_test, y_test = x_all[split.test], y_all[split.test]
    x_train, y_train = x_all[split.train], y_all[split.train]
    x_val, y_val = x_all[split.validation], y_all[split.validation]

    # -- Standardisation (training statistics only) ------------------------
    x_mean, x_std = standardisation(x_train)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    np.save(os.path.join(args.out, X_MEAN_FILE), x_mean)
    np.save(os.path.join(args.out, X_STD_FILE), x_std)
    print(f"Samples: train {split.n_train}, validation {split.n_validation}, "
          f"test {split.n_test}")

    t = lambda a: torch.from_numpy(a).float()  # noqa: E731
    x_train_t, y_train_t = t(x_train), t(y_train)
    x_val_t, y_val_t = t(x_val), t(y_val)
    x_test_t = t(x_test)

    # -- Model and optimiser -----------------------------------------------
    pyro.clear_param_store()
    model, guide = build_model_and_guide()
    optimizer = ClippedAdam({"lr": 3e-3, "lrd": 0.99998,
                             "betas": (0.9, 0.999), "clip_norm": 2.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    val_predictive = Predictive(model, guide=guide, num_samples=args.val_samples,
                                return_sites=["mu", "ale_std"])

    # -- Training with early stopping on the validation NLL ----------------
    best_nll, best_step, best_params, waited = float("inf"), 0, None, 0
    history = []
    print(f"{'step':>8}  {'ELBO sum':>12}  {'val NLL':>9}")
    for step in range(args.max_steps):
        elbo_per = svi.step(x_train_t, y_train_t) / len(x_train)
        if step % args.check_every:
            continue

        mu, var_epi, var_ale = predictive_moments(val_predictive, x_val_t)
        val_nll = -dist.Normal(mu, (var_epi + var_ale).sqrt()).log_prob(y_val_t).mean().item()
        history.append((step, elbo_per, val_nll))

        if val_nll < best_nll - 1e-4:
            best_nll, best_step, waited = val_nll, step, 0
            best_params = snapshot_params()
            status = "best"
        else:
            waited += 1
            status = f"no improvement ({waited}/{args.patience})"
        print(f"{step:>8d}  {elbo_per:>12.4f}  {val_nll:>9.4f}  {status}")
        if waited >= args.patience:
            print("Early stopping.")
            break
    print(f"Selected step {best_step} (validation NLL {best_nll:.4f})")

    # -- Save the selected model -------------------------------------------
    restore_params(best_params)
    torch.save({"epoch": best_step, "val_nll": best_nll,
                "params": {k: v.detach().cpu().clone() for k, v in pyro.get_param_store().items()}},
               os.path.join(args.out, CHECKPOINT_FILE))
    with open(os.path.join(args.out, "training_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "elbo_per_sample", "val_nll"])
        w.writerows(history)

    # -- Evaluation on the test partition ----------------------------------
    test_predictive = Predictive(model, guide=guide, num_samples=args.test_samples,
                                 return_sites=["mu", "ale_std"])
    mu, var_epi, var_ale = (a.numpy() for a in predictive_moments(test_predictive, x_test_t))
    epi, ale, tot = np.sqrt(var_epi), np.sqrt(var_ale), np.sqrt(var_epi + var_ale)
    err = mu - y_test
    metrics = {
        "n_test": int(len(y_test)),
        "rmse_log10": float(np.sqrt(np.mean(err ** 2))),
        "bias_log10": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
        "coverage_95_percent": float(100 * np.mean(np.abs(err) <= 1.96 * tot)),
        "mean_epistemic_std": float(epi.mean()),
        "mean_aleatoric_std": float(ale.mean()),
        "mean_total_std": float(tot.mean()),
        "best_validation_nll": float(best_nll),
        "selected_step": int(best_step),
    }
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # test_predictions.csv is one row per test record: every column must be a
    # 1-D array of length n_test, otherwise csv.writer silently emits a single
    # row whose fields are whole arrays.
    columns = [("log10_chl_insitu", y_test), ("log10_chl_pred", mu),
               ("std_epistemic", epi), ("std_aleatoric", ale), ("std_total", tot)]
    for name, column in columns:
        column = np.asarray(column)
        if column.ndim != 1 or column.shape[0] != metrics["n_test"]:
            raise ValueError(f"internal error: the {name} column has shape {column.shape}, "
                             f"expected ({metrics['n_test']},)")
    with open(os.path.join(args.out, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name for name, _ in columns])
        w.writerows(zip(*(column for _, column in columns), strict=True))

    print("Test partition:")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
