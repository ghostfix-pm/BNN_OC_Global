"""Tests for the in situ database contract: loading, validation and splitting."""

import numpy as np
import pytest

from conftest import make_database

from data_utils import (BANDS_NM, DatasetError, N_BANDS, describe_dataset, load_dataset,
                        split_dataset, standardisation, to_log10)


def save(tmp_path, name, array):
    path = tmp_path / name
    np.save(path, array)
    return path


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
def test_load_dataset_accepts_the_documented_shapes(tmp_path):
    rrs, chl = make_database(12)
    r, c = load_dataset(save(tmp_path, "r.npy", rrs), save(tmp_path, "c.npy", chl))
    assert r.shape == (12, N_BANDS) and r.dtype == np.float64
    assert c.shape == (12,) and c.dtype == np.float64

    # (N, 1) and (N,) Chl-a are equivalent
    _, c_flat = load_dataset(save(tmp_path, "r.npy", rrs), save(tmp_path, "c2.npy", chl[:, 0]))
    assert np.array_equal(c, c_flat)


def test_load_dataset_rejects_mismatched_lengths(tmp_path):
    rrs, chl = make_database(10)
    chl_long = np.repeat(chl, 2, axis=0)            # 20 records for 10 Rrs records
    with pytest.raises(DatasetError) as exc:
        load_dataset(save(tmp_path, "r.npy", rrs), save(tmp_path, "c.npy", chl_long))
    assert "different numbers of records" in str(exc.value)
    assert "10" in str(exc.value) and "20" in str(exc.value)


def test_load_dataset_rejects_a_multi_column_chl(tmp_path):
    """A (N, k>1) Chl-a file used to be silently reduced to its first column."""
    rrs, chl = make_database(8)
    with pytest.raises(DatasetError) as exc:
        load_dataset(save(tmp_path, "r.npy", rrs),
                     save(tmp_path, "c.npy", np.repeat(chl, 3, axis=1)))
    assert "single Chl-a column" in str(exc.value)


def test_load_dataset_rejects_a_wrong_number_of_bands(tmp_path):
    rrs, chl = make_database(8)
    with pytest.raises(DatasetError) as exc:
        load_dataset(save(tmp_path, "r.npy", rrs[:, :3]), save(tmp_path, "c.npy", chl))
    assert "expected a 2-D array of shape (N, 4)" in str(exc.value)


def test_load_dataset_rejects_one_dimensional_rrs(tmp_path):
    rrs, chl = make_database(8)
    with pytest.raises(DatasetError):
        load_dataset(save(tmp_path, "r.npy", rrs[:, 0]), save(tmp_path, "c.npy", chl))


def test_load_dataset_rejects_empty_input(tmp_path):
    with pytest.raises(DatasetError) as exc:
        load_dataset(save(tmp_path, "r.npy", np.zeros((0, N_BANDS))),
                     save(tmp_path, "c.npy", np.zeros((0, 1))))
    assert "no records" in str(exc.value)


def test_load_dataset_names_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset(tmp_path / "absent.npy", tmp_path / "also_absent.npy")
    assert "absent.npy" in str(exc.value)


def test_load_dataset_rejects_a_non_numpy_file(tmp_path):
    path = tmp_path / "rrs.npy"
    path.write_text("this is not an array")
    with pytest.raises(DatasetError) as exc:
        load_dataset(path, path)
    assert "could not be read" in str(exc.value)


# ---------------------------------------------------------------------------
# to_log10 / standardisation
# ---------------------------------------------------------------------------
def test_to_log10_flags_non_positive_chl_and_non_finite_bands():
    rrs = np.full((4, N_BANDS), 0.005)
    rrs[1, 2] = np.nan
    rrs[2, 0] = np.inf
    chl = np.array([1.0, 2.0, 3.0, -1.0])
    x, y, usable = to_log10(rrs, chl)
    assert usable.tolist() == [True, False, False, False]
    assert np.isfinite(x[usable]).all() and np.isfinite(y[usable]).all()
    # log10(0.005) is exactly what the network sees before standardisation
    assert x[0, 0] == pytest.approx(np.log10(0.005))


def test_to_log10_requires_one_chl_value_per_record():
    rrs = np.full((3, N_BANDS), 0.005)
    with pytest.raises(DatasetError) as exc:
        to_log10(rrs, np.ones(2))
    assert "one value per Rrs record" in str(exc.value)


def test_standardisation_uses_the_training_statistics():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, std = standardisation(x)
    assert np.allclose(mean, [3.0, 4.0])
    assert np.all(std > 0)
    assert np.allclose(std, x.std(axis=0) + 1e-8)


def test_standardisation_keeps_a_constant_band_finite():
    mean, std = standardisation(np.ones((5, 2)))
    assert np.all(np.isfinite(std)) and np.all(std == 1e-8)


def test_standardisation_rejects_an_empty_partition():
    with pytest.raises(DatasetError):
        standardisation(np.zeros((0, N_BANDS)))


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------
def test_split_dataset_uses_the_documented_fractions():
    split = split_dataset(1000)
    assert (split.n_test, split.n_train, split.n_validation) == (150, 700, 150)
    assert split.n_total == 1000


def test_split_dataset_partitions_do_not_overlap():
    n = 37
    split = split_dataset(n, 0.2, 0.3)
    seen = []
    for part in (split.test, split.train, split.validation):
        seen.extend(range(*part.indices(n)))
    assert sorted(seen) == list(range(n))               # every record exactly once


def test_split_dataset_rejects_fractions_that_leave_no_training_data():
    with pytest.raises(DatasetError) as exc:
        split_dataset(100, 0.6, 0.6)
    assert "below 1" in str(exc.value)


@pytest.mark.parametrize("val_frac,test_frac", [(0.0, 0.15), (1.0, 0.0), (-0.1, 0.2), (0.5, 1.0)])
def test_split_dataset_rejects_fractions_outside_zero_one(val_frac, test_frac):
    with pytest.raises(DatasetError) as exc:
        split_dataset(100, val_frac, test_frac)
    assert "strictly between 0 and 1" in str(exc.value)


def test_split_dataset_rejects_an_empty_partition():
    with pytest.raises(DatasetError) as exc:
        split_dataset(6)                                 # int(0.15 * 6) == 0 records
    assert "would be empty" in str(exc.value)


def test_split_dataset_slices_are_contiguous():
    split = split_dataset(40, 0.25, 0.25)
    assert split.test == slice(0, 10)
    assert split.train == slice(10, 30)
    assert split.validation == slice(30, None)


# ---------------------------------------------------------------------------
# describe_dataset
# ---------------------------------------------------------------------------
def test_describe_dataset_summarises_the_database():
    rrs, chl = make_database(20)
    rrs[0, 0] = np.nan
    chl[1, 0] = -5.0
    info = describe_dataset(rrs, chl)
    assert info["n_records"] == 20 and info["n_bands"] == N_BANDS
    assert info["n_usable"] == 18 and info["n_dropped"] == 2
    assert info["n_invalid_chl"] == 1 and info["n_nonfinite_rrs"] == 1
    assert info["bands_nm"] == list(BANDS_NM)
    assert info["chl_min"] > 0 and info["chl_max"] >= info["chl_min"]
    assert len(info["log10_rrs_mean"]) == N_BANDS
