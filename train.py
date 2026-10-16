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
  test_predictions.csv     per-sample test predictions and uncertainties

Example:
  python train.py --rrs data/rrs_in.npy --chl data/chl_out.npy --out model
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam

from bnn_model import (CHECKPOINT_FILE, X_MEAN_FILE, X_STD_FILE,
                       build_model_and_guide, sync_activations)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rrs", required=True,
                   help="in situ Rrs, .npy array (N, 4) in sr-1 at 443, 490, 510, 560 nm")
    p.add_argument("--chl", required=True,
                   help="in situ Chl-a, .npy array (N, 1) in mg m-3")
    p.add_argument("--out", default="model", help="output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--check-every", type=int, default=1_000,
                   help="SVI steps between validation-NLL evaluations")
    p.add_argument("--patience", type=int, default=10,
                   help="validation checks without improvement before stopping")
    p.add_argument("--val-samples", type=int, default=200,
                   help="posterior samples for the validation NLL")
    p.add_argument("--test-samples", type=int, default=500,
                   help="posterior samples for the test evaluation")
    return p.parse_args()


def predictive_moments(predictive, x_t):
    """Predictive mean and epistemic / aleatoric variances (law of total variance)."""
    with torch.no_grad():
        s = predictive(x_t)
    mu_s, sig_s = s["mu"], s["ale_std"]
    return mu_s.mean(0), mu_s.var(0), (sig_s ** 2).mean(0)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # -- Data --------------------------------------------------------------
    rrs = np.load(args.rrs)
    chl = np.load(args.chl).reshape(len(rrs), -1)
    perm = np.random.permutation(len(rrs))
    rrs, chl = rrs[perm, :], chl[perm, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        x_all = np.log10(rrs)
        y_all = np.log10(chl[:, 0])
    finite = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
    x_all, y_all = x_all[finite], y_all[finite]

    # -- Split: [ test | train | validation ] of the shuffled rows ---------
    n_all = len(x_all)
    n_val = int(args.val_frac * n_all)
    n_test = int(args.test_frac * n_all)
    n_train = n_all - n_val - n_test
    x_test, y_test = x_all[:n_test], y_all[:n_test]
    x_train, y_train = x_all[n_test:n_test + n_train], y_all[n_test:n_test + n_train]
    x_val, y_val = x_all[n_test + n_train:], y_all[n_test + n_train:]

    # -- Standardisation (training statistics only) ------------------------
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    np.save(os.path.join(args.out, X_MEAN_FILE), x_mean)
    np.save(os.path.join(args.out, X_STD_FILE), x_std)
    print(f"Samples: train {len(x_train)}, validation {len(x_val)}, test {len(x_test)}")

    t = lambda a: torch.from_numpy(a).float()
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
    print(f"{'step':>8}  {'ELBO/sample':>12}  {'val NLL':>9}")
    for step in range(args.max_steps):
        elbo_per = svi.step(x_train_t, y_train_t) / len(x_train)
        if step % args.check_every:
            continue

        mu, var_epi, var_ale = predictive_moments(val_predictive, x_val_t)
        val_nll = -dist.Normal(mu, (var_epi + var_ale).sqrt()).log_prob(y_val_t).mean().item()
        history.append((step, elbo_per, val_nll))

        if val_nll < best_nll - 1e-4:
            best_nll, best_step, waited = val_nll, step, 0
            best_params = {k: v.detach().clone()
                           for k, v in pyro.get_param_store().items()}
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
    pyro.get_param_store().clear()
    for k, v in best_params.items():
        pyro.param(k, v)
    sync_activations(model)
    torch.save({"epoch": best_step, "val_nll": best_nll,
                "params": {k: v.detach().cpu().clone() for k, v in best_params.items()}},
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
    with open(os.path.join(args.out, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log10_chl_insitu", "log10_chl_pred", "std_epistemic",
                    "std_aleatoric", "std_total"])
        w.writerows(zip(y_test, mu, epi, ale, tot))

    print("Test partition:")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
