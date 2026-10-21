"""
Loading, validation and splitting of the in situ Rrs / Chl-a database.

NumPy only: this module deliberately imports nothing from PyTorch or Pyro so
that the data contract (shapes, units, partitions) can be checked and tested
without the full stack.

The input records are Rrs (sr-1) at ``BANDS_NM`` and the matching Chl-a
concentration (mg m-3) for the same N stations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Input definition
# ---------------------------------------------------------------------------
BANDS_NM = (443, 490, 510, 560)                 # Rrs input bands (nm)
RRS_VARIABLES = tuple(f"Rrs_{b}" for b in BANDS_NM)
N_BANDS = len(BANDS_NM)
RRS_FLOOR = 1e-6                                # Rrs (sr-1) is clipped here before log10
STD_FLOOR = 1e-8                                # keeps the standardisation finite for constant bands

DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15


class DatasetError(ValueError):
    """Raised when the in situ database does not match the documented format."""


def _load_array(path, what):
    """Read one .npy file, turning loader errors into a message naming the file."""
    try:
        array = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{what} file not found: {path}") from None
    except Exception as exc:                    # corrupt/pickled/empty file
        raise DatasetError(f"{what} file {path} could not be read as a .npy array: {exc}") from None
    return np.asarray(array, dtype=np.float64)


def load_dataset(rrs_path, chl_path, n_bands=N_BANDS):
    """
    Load and validate the in situ training database.

    ``rrs_path`` must hold an (N, n_bands) array of Rrs in sr-1 and
    ``chl_path`` an (N,) or (N, 1) array of Chl-a in mg m-3 for the same
    records; the two files are matched record by record, so their lengths must
    agree. Returns ``(rrs, chl)`` as float64, with ``chl`` always 1-D.

    Unlike a bare ``reshape(len(rrs), -1)``, this refuses mismatched inputs
    instead of silently pairing every Rrs record with the wrong Chl-a value.
    """
    rrs = _load_array(rrs_path, "Rrs")
    chl = _load_array(chl_path, "Chl-a")

    if rrs.ndim != 2 or rrs.shape[1] != n_bands:
        raise DatasetError(
            f"Rrs file {rrs_path}: expected a 2-D array of shape (N, {n_bands}) "
            f"with the bands {', '.join(str(b) for b in BANDS_NM)} nm, got shape {rrs.shape}")

    n_records = rrs.shape[0]
    if chl.ndim == 2:
        if chl.shape[1] != 1:
            raise DatasetError(
                f"Chl-a file {chl_path}: expected a single Chl-a column of shape (N,) "
                f"or (N, 1), got shape {chl.shape}")
        chl = chl[:, 0]
    elif chl.ndim != 1:
        raise DatasetError(
            f"Chl-a file {chl_path}: expected shape (N,) or (N, 1), got shape {chl.shape}")

    if chl.shape[0] != n_records:
        raise DatasetError(
            f"the two files describe different numbers of records: {rrs_path} has "
            f"{n_records}, {chl_path} has {chl.shape[0]}. Rrs and Chl-a are paired "
            "record by record, so they must be the same length.")
    if n_records == 0:
        raise DatasetError(f"{rrs_path} contains no records")

    return rrs, chl


def to_log10(rrs, chl):
    """
    log10 of both the Rrs and the Chl-a records, plus the mask of the records
    that are usable for training (all bands finite and Chl-a positive).

    Returns ``(log10_rrs, log10_chl, usable)``.
    """
    rrs = np.asarray(rrs, dtype=np.float64)
    chl = np.asarray(chl, dtype=np.float64)
    if rrs.ndim != 2:
        raise DatasetError(f"Rrs must be 2-D (N, {N_BANDS}), got shape {rrs.shape}")
    if chl.shape != (rrs.shape[0],):
        raise DatasetError(
            f"Chl-a must be 1-D with one value per Rrs record, got shape {chl.shape} "
            f"for {rrs.shape[0]} records")
    with np.errstate(divide="ignore", invalid="ignore"):
        x_all = np.log10(rrs)
        y_all = np.log10(chl)
    usable = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
    return x_all, y_all, usable


def standardisation(x_train):
    """
    Mean and standard deviation of the training partition, used to standardise
    every partition. ``STD_FLOOR`` keeps the scaling finite for a constant band.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise DatasetError(f"the training partition must be a non-empty 2-D array, got {x_train.shape}")
    return x_train.mean(axis=0), x_train.std(axis=0) + STD_FLOOR


@dataclass(frozen=True)
class Split:
    """
    Positional partitions of the shuffled record index, matching the layout
    ``[ test | train | validation ]``.
    """

    test: slice
    train: slice
    validation: slice
    n_test: int
    n_train: int
    n_validation: int

    @property
    def n_total(self):
        return self.n_test + self.n_train + self.n_validation


def split_dataset(n_samples, val_frac=DEFAULT_VAL_FRAC, test_frac=DEFAULT_TEST_FRAC):
    """
    Split ``n_samples`` shuffled records into contiguous test / train /
    validation partitions.

    Refuses fractions that would leave a partition empty or make the
    partitions overlap (the upstream ``int(frac * n)`` arithmetic silently
    produced a negative training size and overlapping slices).
    """
    for name, frac in (("--val-frac", val_frac), ("--test-frac", test_frac)):
        if not np.isfinite(frac) or not 0.0 < frac < 1.0:
            raise DatasetError(f"{name} must be strictly between 0 and 1, got {frac}")
    if val_frac + test_frac >= 1.0:
        raise DatasetError(
            f"--val-frac ({val_frac}) + --test-frac ({test_frac}) must be below 1: "
            "no records would be left for training")

    n_test = int(test_frac * n_samples)
    n_val = int(val_frac * n_samples)
    n_train = n_samples - n_test - n_val
    for name, n in (("test", n_test), ("validation", n_val), ("training", n_train)):
        if n < 1:
            raise DatasetError(
                f"the {name} partition would be empty ({n} of {n_samples} records): "
                "use smaller --val-frac / --test-frac values or supply more records")

    return Split(
        test=slice(0, n_test),
        train=slice(n_test, n_test + n_train),
        validation=slice(n_test + n_train, None),
        n_test=n_test, n_train=n_train, n_validation=n_val,
    )


def describe_dataset(rrs, chl):
    """Summary of the in situ database, for ``train.py --check-data``."""
    chl = np.asarray(chl, dtype=np.float64)
    if chl.ndim == 2 and chl.shape[1] == 1:      # the documented (N, 1) form
        chl = chl[:, 0]
    x_all, y_all, usable = to_log10(rrs, chl)
    valid_chl = np.isfinite(chl) & (chl > 0)
    info = {
        "n_records": int(rrs.shape[0]),
        "n_bands": int(rrs.shape[1]),
        "n_usable": int(usable.sum()),
        "n_dropped": int((~usable).sum()),
        "n_invalid_chl": int((~valid_chl).sum()),
        "n_nonfinite_rrs": int((~np.isfinite(rrs).all(axis=1)).sum()),
        "bands_nm": list(BANDS_NM),
    }
    if valid_chl.any():
        info["chl_min"] = float(chl[valid_chl].min())
        info["chl_max"] = float(chl[valid_chl].max())
    if usable.any():
        info["log10_rrs_min"] = float(x_all[usable].min())
        info["log10_rrs_max"] = float(x_all[usable].max())
        info["log10_rrs_mean"] = [round(float(v), 4) for v in x_all[usable].mean(axis=0)]
        info["log10_rrs_std"] = [round(float(v), 4) for v in x_all[usable].std(axis=0)]
    return info
