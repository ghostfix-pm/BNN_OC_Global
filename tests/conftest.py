"""
Shared fixtures: a synthetic in situ database, a small trained model and a
synthetic OC-CCI input directory.

The repository is a flat set of scripts (``train.py`` imports ``bnn_model``),
so the repository root is put on ``sys.path`` before the modules are imported.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BANDS = (443, 490, 510, 560)


def make_database(n=60, seed=3):
    """Synthetic (N, 4) Rrs / (N, 1) Chl-a pair with a learnable relation."""
    rng = np.random.default_rng(seed)
    rrs = 10.0 ** rng.normal(-2.3, 0.25, size=(n, len(BANDS)))
    # A monotone relation so a few SVI steps already fit something.
    signal = rrs[:, 1] / rrs.mean() - 1.0
    chl = 0.6 * np.exp(3.0 * signal) * np.exp(rng.normal(0, 0.12, size=n))
    return rrs, chl.reshape(-1, 1)


@pytest.fixture
def database(tmp_path):
    """Paths of the synthetic database, as (rrs_path, chl_path)."""
    rrs, chl = make_database()
    rrs_path, chl_path = tmp_path / "rrs_in.npy", tmp_path / "chl_out.npy"
    np.save(rrs_path, rrs)
    np.save(chl_path, chl)
    return rrs_path, chl_path


def train_tiny_model(out_dir, rrs_path, chl_path, steps=3, samples=4):
    """Run train.py's main() for a handful of SVI steps."""
    import train

    argv = ["train.py", "--rrs", str(rrs_path), "--chl", str(chl_path),
            "--out", str(out_dir), "--max-steps", str(steps), "--check-every", "1",
            "--val-samples", str(samples), "--test-samples", str(samples)]
    previous = sys.argv
    sys.argv = argv
    try:
        train.main()
    finally:
        sys.argv = previous
    return out_dir


@pytest.fixture
def model_dir(tmp_path, database):
    """A model folder produced by a real (very short) training run."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    rrs_path, chl_path = database
    return train_tiny_model(tmp_path / "model", rrs_path, chl_path)


def write_daily_file(path, lat, lon, rng, nan_fraction=0.10, fill_hole=None):
    """One synthetic OC-CCI v6.0 style daily file."""
    import netCDF4 as nc

    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", None)
    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))
    t = ds.createVariable("time", "i4", ("time",))
    t.units, t.calendar = "days since 1970-01-01 00:00:00", "standard"
    t[:] = 0
    la = ds.createVariable("lat", "f4", ("lat",))
    la.units = "degrees_north"
    la[:] = lat
    lo = ds.createVariable("lon", "f4", ("lon",))
    lo.units = "degrees_east"
    lo[:] = lon
    n_lat, n_lon = len(lat), len(lon)
    base = {443: 0.004, 490: 0.005, 510: 0.006, 560: 0.007}
    hole = rng.random((1, n_lat, n_lon)) < nan_fraction
    if fill_hole is not None:
        # a whole block without a single valid pixel, to test --min-valid
        i0, j0, size = fill_hole
        hole[:, i0:i0 + size, j0:j0 + size] = True
    for band in BANDS:
        var = ds.createVariable(f"Rrs_{band}", "f4", ("time", "lat", "lon"),
                                fill_value=np.float32(9.96921e+36), zlib=True)
        var.units = "sr-1"
        field = base[band] * (1.0 + 0.25 * rng.standard_normal((1, n_lat, n_lon)))
        field = np.clip(field, 1e-5, None)
        var[:] = np.ma.masked_array(field, mask=hole)
    ds.close()
    return path


@pytest.fixture
def occi_dir(tmp_path):
    """Directory with three synthetic daily files on a 48 x 48 grid."""
    import netCDF4 as nc  # noqa: F401  (skip the fixture when the module is missing)
    from datetime import datetime

    pytest.importorskip("netCDF4")
    out = tmp_path / "occci"
    out.mkdir()
    rng = np.random.default_rng(7)
    lat = np.arange(60.0, 12.0 - 1e-9, -1.0)[:48]        # north -> south
    lon = np.arange(-30.0, 18.0 - 1e-9, 1.0)[:48]        # west -> east
    for day in (datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2022, 3, 4)):
        name = f"ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-{day:%Y%m%d}-fv6.0.nc"
        write_daily_file(out / name, lat, lon, rng)
    return out
