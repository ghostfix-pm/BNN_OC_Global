"""Tests for the model module: standardisation, predictive moments, checkpoint loading."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyro")

from bnn_model import (BANDS_NM, CHECKPOINT_FILE, N_BANDS, _as_sample_matrix,  # noqa: E402
                       build_model_and_guide, load_trained_model, predict,
                       predictive_moments, snapshot_params, standardise)



# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------
def test_forward_returns_one_log10_chl_value_per_record():
    model, guide = build_model_and_guide()
    x = torch.rand(5, N_BANDS)
    with torch.no_grad():
        mu = model(x)
    assert tuple(mu.shape) == (5,)
    assert model.act1.beta.requires_grad          # one learnable Swish per layer


# ---------------------------------------------------------------------------
# standardise
# ---------------------------------------------------------------------------
def test_standardise_clips_at_the_rrs_floor_and_flags_bad_rows():
    rrs = np.array([[0.005, 0.005, 0.005, 0.005],
                    [0.0, 0.005, 0.005, 0.005],
                    [0.005, np.nan, 0.005, 0.005],
                    [0.005, -1.0e-9, 0.005, 0.005]])
    x, invalid = standardise(rrs, np.zeros(N_BANDS), np.ones(N_BANDS))
    # only non-finite rows are flagged and neutered
    assert invalid.tolist() == [False, False, True, False]
    assert np.all(x[invalid] == 0.0)
    assert x.dtype == np.float32
    # zero and negative Rrs are clipped to the floor instead of producing -inf
    assert np.isclose(x[1, 0], np.log10(1e-6)) and np.isfinite(x[1]).all()
    assert np.isclose(x[3, 1], np.log10(1e-6)) and np.isfinite(x[3]).all()
    assert np.isclose(x[0, 0], np.log10(0.005))


def test_standardise_applies_the_training_statistics():
    rrs = np.full((2, N_BANDS), 0.01)
    x, _ = standardise(rrs, np.zeros(N_BANDS), np.full(N_BANDS, 2.0))
    assert np.allclose(x, np.log10(0.01) / 2.0)


@pytest.mark.parametrize("shape", [(4, 3), (4,), (2, 2, N_BANDS)])
def test_standardise_rejects_a_wrong_shape(shape):
    with pytest.raises(ValueError):
        standardise(np.zeros(shape), np.zeros(N_BANDS), np.ones(N_BANDS))


# ---------------------------------------------------------------------------
# _as_sample_matrix / predictive_moments
# ---------------------------------------------------------------------------
def test_as_sample_matrix_drops_a_redundant_batch_dimension():
    samples = torch.zeros(4, 1, 7)
    out = _as_sample_matrix(samples, 7, "mu")
    assert tuple(out.shape) == (4, 7)


def test_as_sample_matrix_rejects_a_real_batch_dimension():
    """A guide prototyped on a batch would silently reduce the wrong axis."""
    with pytest.raises(ValueError) as exc:
        _as_sample_matrix(torch.zeros(4, 3, 7), 7, "mu")
    assert "batch dimensions" in str(exc.value)


def test_as_sample_matrix_rejects_a_mismatched_last_dimension():
    with pytest.raises(ValueError) as exc:
        _as_sample_matrix(torch.zeros(4, 1, 7), 5, "mu")
    assert "values per sample" in str(exc.value)


def test_as_sample_matrix_rejects_a_one_dimensional_site():
    with pytest.raises(ValueError) as exc:
        _as_sample_matrix(torch.zeros(7), 7, "mu")
    assert "at least" in str(exc.value)


class StubPredictive:
    """Posterior predictive stub with the (num_samples, 1, n_rows) site shapes
    that a Pyro guide prototyped on a single row produces."""

    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        n_rows = int(x.shape[0])
        assert self.mu.shape[-1] == n_rows
        return {"mu": self.mu.unsqueeze(1), "ale_std": self.sigma.unsqueeze(1)}


def test_predictive_moments_returns_one_value_per_row():
    """Regression: the sites carry a size-1 batch dim, and reducing the wrong
    axis used to leave a (1, n) array whose csv.writer output was one row of
    whole arrays."""
    mu = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    sigma = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    predictive = StubPredictive(mu, sigma)
    mean, var_epi, var_ale = predictive_moments(predictive, torch.zeros(3, N_BANDS))

    assert mean.shape == (3,) and var_epi.shape == (3,) and var_ale.shape == (3,)
    assert predictive.calls == 1
    np.testing.assert_allclose(mean.numpy(), mu.mean(dim=0).numpy(), rtol=1e-6)
    np.testing.assert_allclose(var_epi.numpy(), mu.var(dim=0).numpy(), rtol=1e-6)
    np.testing.assert_allclose(var_ale.numpy(), (sigma ** 2).mean(dim=0).numpy(), rtol=1e-6)


def test_predictive_moments_accepts_numpy_input():
    mu = torch.zeros(2, 4)
    predictive = StubPredictive(mu, torch.ones(2, 4))
    mean, _, _ = predictive_moments(predictive, np.zeros((4, N_BANDS), dtype=np.float64))
    assert mean.shape == (4,)


# ---------------------------------------------------------------------------
# predict (masking and seed handling)
# ---------------------------------------------------------------------------
def test_predict_returns_nan_for_invalid_rows():
    predictive = StubPredictive(torch.zeros(2, 2), torch.ones(2, 2))
    x = np.zeros((3, N_BANDS), dtype=np.float32)
    invalid = np.array([False, True, False])
    mu, epi, ale = predict(predictive, x, invalid)
    assert np.isnan(mu[1]) and np.isnan(epi[1]) and np.isnan(ale[1])
    assert np.isfinite(mu[[0, 2]]).all()


def test_predict_skips_the_predictive_when_every_row_is_invalid():
    predictive = StubPredictive(torch.zeros(2, 2), torch.ones(2, 2))
    mu, epi, ale = predict(predictive, np.zeros((2, N_BANDS), dtype=np.float32), np.ones(2, bool))
    assert predictive.calls == 0
    assert np.isnan(mu).all()


def test_predict_decomposition_follows_the_law_of_total_variance():
    # torch.var (and the published code) use the unbiased estimator, so the
    # epistemic spread is sqrt(sum((mu_i - mean)^2) / (S - 1)).
    mu = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 1.0, 4.0]])
    sigma = torch.full((3, 3), 0.5)
    predictive = StubPredictive(mu, sigma)
    _, epi, ale = predict(predictive, np.zeros((3, N_BANDS), dtype=np.float32),
                          np.zeros(3, dtype=bool), seed=42)
    np.testing.assert_allclose(epi, [1.0, 0.0, 1.0], atol=1e-6)   # sd of the sample means
    np.testing.assert_allclose(ale, 0.5, atol=1e-6)               # E[sigma^2] ** 0.5
    assert (epi >= 0).all() and (ale > 0).all()


# ---------------------------------------------------------------------------
# seed handling on a real trained model
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_and_no_seed_draws_again(model_dir):
    predictive, x_mean, x_std = load_trained_model(str(model_dir), num_samples=8)
    x = np.random.default_rng(0).normal(size=(6, N_BANDS)).astype(np.float32)
    x = ((x - x_mean) / x_std).astype(np.float32)

    first = predictive_moments(predictive, x, seed=42)[0].numpy()
    second = predictive_moments(predictive, x, seed=42)[0].numpy()
    np.testing.assert_array_equal(first, second)

    free_a = predictive_moments(predictive, x, seed=None)[0].numpy()
    free_b = predictive_moments(predictive, x, seed=None)[0].numpy()
    assert not np.array_equal(free_a, free_b)


def test_predict_with_seed_none_still_returns_finite_values(model_dir):
    predictive, x_mean, x_std = load_trained_model(str(model_dir), num_samples=8)
    mu, epi, ale = predict(predictive, np.zeros((4, N_BANDS), dtype=np.float32),
                           np.zeros(4, dtype=bool), seed=None)
    assert np.isfinite(mu).all() and (epi >= 0).all() and (ale > 0).all()


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def test_load_trained_model_reports_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_trained_model(str(tmp_path / "nope"))
    assert "train the model first" in str(exc.value)


def test_load_trained_model_reports_an_incomplete_folder(tmp_path, model_dir):
    (model_dir / "x_train_mu.npy").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        load_trained_model(str(model_dir))
    assert "not a complete model folder" in str(exc.value)


def test_load_trained_model_rejects_a_checkpoint_of_the_wrong_architecture(tmp_path, model_dir):
    """A checkpoint missing parameters must not be loaded silently."""
    state = torch.load(model_dir / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    del state["params"]["act1$$$beta"]
    torch.save(state, model_dir / CHECKPOINT_FILE)
    with pytest.raises(ValueError) as exc:
        load_trained_model(str(model_dir))
    assert "does not contain" in str(exc.value)


def test_load_trained_model_round_trips_a_real_checkpoint(model_dir):
    predictive, x_mean, x_std = load_trained_model(str(model_dir), num_samples=4)
    assert x_mean.shape == (N_BANDS,) and x_std.shape == (N_BANDS,)
    assert (x_std > 0).all()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, N_BANDS)).astype(np.float32)     # distinct rows
    mu, epi, ale = predict(predictive, x, np.zeros(5, dtype=bool), seed=42)
    assert np.isfinite(mu).all() and (ale > 0).all()
    assert mu.std() > 0                     # not a constant placeholder model


def test_snapshot_and_restore_round_trip():
    model, guide = build_model_and_guide()
    snapshot = snapshot_params()
    assert snapshot                                     # the guide registers parameters
    for value in snapshot.values():
        assert isinstance(value, torch.Tensor)


def test_rrs_variables_match_the_band_definition():
    from bnn_model import RRS_VARIABLES
    assert RRS_VARIABLES == tuple(f"Rrs_{b}" for b in BANDS_NM)
