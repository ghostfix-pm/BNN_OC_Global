"""Tests for train.py: argument validation, --check-data, the training run and its outputs."""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, make_database, train_tiny_model

import train


def run_main(argv):
    """Call train.main() with a given command line."""
    previous = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        train.main()
    finally:
        sys.argv = previous


# ---------------------------------------------------------------------------
# --check-data (NumPy only)
# ---------------------------------------------------------------------------
def test_check_data_prints_a_summary(capsys, database):
    rrs_path, chl_path = database
    run_main(["--rrs", str(rrs_path), "--chl", str(chl_path), "--check-data"])
    out = capsys.readouterr().out
    assert "60 records x 4 bands" in out
    assert "usable for training: 60 of 60" in out
    assert "Chl-a range" in out and "log10(Rrs) mean" in out


def test_check_data_needs_only_numpy(tmp_path):
    """--check-data must not drag in the deep-learning stack."""
    rrs, chl = make_database(20)
    np.save(tmp_path / "rrs.npy", rrs)
    np.save(tmp_path / "chl.npy", chl)
    code = (
        "import sys; sys.argv = ['train.py', '--rrs', sys.argv[1], '--chl', sys.argv[2], "
        "'--check-data']; import train; train.main(); "
        "missing = [m for m in ('torch', 'pyro') if m in sys.modules]; "
        "assert not missing, missing; print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "rrs.npy"),
                             str(tmp_path / "chl.npy")],
                            cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_check_data_rejects_a_broken_database(capsys, tmp_path):
    rrs, chl = make_database(20)
    np.save(tmp_path / "rrs.npy", rrs)
    np.save(tmp_path / "chl.npy", chl[:10])            # 10 Chl-a rows for 20 Rrs rows
    with pytest.raises(SystemExit) as exc:
        run_main(["--rrs", str(tmp_path / "rrs.npy"), "--chl", str(tmp_path / "chl.npy"),
                        "--check-data"])
    assert "different numbers of records" in str(exc.value)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag,value", [("--max-steps", "0"), ("--check-every", "0"),
                                        ("--val-samples", "0"), ("--test-samples", "0"),
                                        ("--patience", "0")])
def test_validate_args_rejects_non_positive_values(database, flag, value):
    rrs_path, chl_path = database
    with pytest.raises(SystemExit) as exc:
        run_main(["--rrs", str(rrs_path), "--chl", str(chl_path),
                        "--check-data", flag, value])
    assert "must be at least 1" in str(exc.value)


def test_training_refuses_split_fractions_without_training_data(database):
    rrs_path, chl_path = database
    with pytest.raises(SystemExit) as exc:
        run_main(["--rrs", str(rrs_path), "--chl", str(chl_path),
                        "--val-frac", "0.6", "--test-frac", "0.6", "--max-steps", "1"])
    assert "below 1" in str(exc.value)


# ---------------------------------------------------------------------------
# the training run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    directory = tmp_path_factory.mktemp("run")
    rrs, chl = make_database()
    np.save(directory / "rrs.npy", rrs)
    np.save(directory / "chl.npy", chl)
    out = directory / "model"
    train_tiny_model(out, directory / "rrs.npy", directory / "chl.npy", steps=3, samples=4)
    return directory, out


def test_training_writes_the_documented_outputs(trained):
    _, out = trained
    for name in ("bnn_best_checkpoint.pt", "x_train_mu.npy", "x_train_std.npy",
                 "training_curve.csv", "test_metrics.json", "test_predictions.csv"):
        assert (out / name).is_file(), name


def test_test_predictions_is_one_row_per_record(trained):
    """Regression: the columns used to be (1, n) arrays, so csv.writer wrote a
    single row whose fields were whole 9-element arrays."""
    _, out = trained
    metrics = json.loads((out / "test_metrics.json").read_text())
    with open(out / "test_predictions.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["log10_chl_insitu", "log10_chl_pred", "std_epistemic",
                       "std_aleatoric", "std_total"]
    assert len(rows) == metrics["n_test"] + 1
    assert all(len(row) == 5 for row in rows[1:])
    numbers = np.array([[float(value) for value in row] for row in rows[1:]])
    assert numbers.shape == (metrics["n_test"], 5)
    assert np.isfinite(numbers).all()
    # the columns must agree with the metrics file
    assert np.isclose(np.sqrt(((numbers[:, 1] - numbers[:, 0]) ** 2).mean()),
                      metrics["rmse_log10"], rtol=1e-6)
    assert np.allclose(numbers[:, 4], np.sqrt(numbers[:, 2] ** 2 + numbers[:, 3] ** 2), rtol=1e-6)


def test_test_metrics_are_within_reasonable_bounds(trained):
    _, out = trained
    metrics = json.loads((out / "test_metrics.json").read_text())
    assert metrics["n_test"] == 9
    assert metrics["selected_step"] == 2                # 3 steps: 0, 1, 2
    assert 0 < metrics["rmse_log10"] < 5
    assert -5 < metrics["bias_log10"] < 5
    assert metrics["r2"] <= 1.0
    assert 0 <= metrics["coverage_95_percent"] <= 100
    assert metrics["mean_aleatoric_std"] > 0 and metrics["mean_epistemic_std"] >= 0


def test_training_curve_has_one_row_per_check(trained):
    _, out = trained
    with open(out / "training_curve.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["step", "elbo_per_sample", "val_nll"]
    assert [int(row[0]) for row in rows[1:]] == [0, 1, 2]


def test_saved_model_loads_back_and_predicts(trained):
    _, out = trained
    from bnn_model import load_trained_model, predict
    predictive, x_mean, x_std = load_trained_model(str(out), num_samples=4)
    assert x_mean.shape == (4,) and x_std.shape == (4,)
    mu, epi, ale = predict(predictive, np.zeros((3, 4), dtype=np.float32),
                           np.zeros(3, dtype=bool), seed=42)
    assert np.isfinite(mu).all() and np.isfinite(epi).all() and (ale > 0).all()


def test_training_reproducible_with_the_same_seed(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    rrs, chl = make_database()
    paths = []
    for run in ("a", "b"):
        directory = tmp_path / run
        directory.mkdir()
        np.save(directory / "rrs.npy", rrs)
        np.save(directory / "chl.npy", chl)
        train_tiny_model(directory / "model", directory / "rrs.npy", directory / "chl.npy")
        paths.append(directory / "model")
    first = json.loads((paths[0] / "test_metrics.json").read_text())
    second = json.loads((paths[1] / "test_metrics.json").read_text())
    assert first == second


def test_training_creates_the_output_directory(database, tmp_path):
    rrs_path, chl_path = database
    out = tmp_path / "deep" / "model"
    train_tiny_model(out, rrs_path, chl_path, steps=1, samples=2)
    assert (out / "bnn_best_checkpoint.pt").is_file()
    assert os.path.isdir(out)
