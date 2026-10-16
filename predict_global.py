"""
Generate the global daily Chl-a dataset from OC-CCI v6.0 Rrs.

Each daily OC-CCI Rrs file (4 km, 1/24 degree) is read, the four input bands
are averaged in linear space over non-overlapping BIN x BIN blocks of valid
pixels (a pixel is valid when all four bands are valid), and the trained BNN
is applied to the block-mean spectrum. With --bin 6 (default) the output grid
is 0.25 degree; with --bin 1 the BNN is applied at the native 4 km resolution.

One NetCDF file is written per year, with one time record per day:
  chl_mean        posterior predictive mean of log10(Chl-a)   [log10(mg m-3)]
  unc_epistemic   epistemic 1-sigma uncertainty               [log10(mg m-3)]
  unc_aleatoric   aleatoric 1-sigma uncertainty               [log10(mg m-3)]
  n_valid         number of valid 4 km pixels averaged in the cell

Re-running the script skips days that are already in the yearly file.

Example:
  python predict_global.py --input-dir /data/occci_v6.0_rrs_daily \
      --output-dir output --years 2000-2025
"""

import argparse
import glob
import os
import re
from datetime import datetime, timedelta, timezone

import netCDF4 as nc
import numpy as np
import torch
from tqdm import tqdm

from bnn_model import RRS_VARIABLES, load_trained_model, predict, standardise

FILL = np.float32(9.96921e+36)
EPOCH = datetime(1970, 1, 1)
DATE_IN_NAME = re.compile(r"-(\d{8})-fv")       # e.g. ..._RRS-20210101-fv6.0.nc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="folder with OC-CCI v6.0 daily Rrs NetCDF files (searched recursively)")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model-dir", default="model",
                   help="folder with the trained model files written by train.py")
    p.add_argument("--years", nargs="+", required=True,
                   help="years to process, e.g. 2021 2022 or 2000-2025")
    p.add_argument("--bin", type=int, default=6,
                   help="block size in 4 km pixels (6 = 0.25 degree, 1 = native 4 km)")
    p.add_argument("--min-valid", type=int, default=1,
                   help="minimum valid 4 km pixels for a cell to be retrieved")
    p.add_argument("--mc-samples", type=int, default=100,
                   help="posterior samples per retrieval")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    return p.parse_args()


def parse_years(tokens):
    years = []
    for tok in tokens:
        if "-" in tok:
            a, b = (int(v) for v in tok.split("-"))
            years.extend(range(a, b + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def file_date(path):
    """Date of a daily file, from its name or, failing that, its time variable."""
    m = DATE_IN_NAME.search(os.path.basename(path))
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    with nc.Dataset(path) as ds:
        t = ds["time"]
        d = nc.num2date(t[0], t.units, getattr(t, "calendar", "standard"))
    return datetime(d.year, d.month, d.day)


def find_daily_files(input_dir):
    files = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)):
        date = file_date(path)
        if date in files:
            raise ValueError(f"two files for {date:%Y-%m-%d}: {files[date]} and {path}")
        files[date] = path
    return files


def read_rrs(path):
    """Return the four input bands as a (lat, lon, 4) float32 array, NaN = missing."""
    with nc.Dataset(path) as ds:
        bands = []
        for name in RRS_VARIABLES:
            a = ds[name][:]
            if a.ndim == 3:            # (time, lat, lon) with a single time step
                a = a[0]
            bands.append(np.ma.filled(a.astype(np.float32), np.nan))
    return np.stack(bands, axis=-1)


def read_grid(path):
    with nc.Dataset(path) as ds:
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
    return lat, lon


def bin_rrs(rrs, b, min_valid):
    """
    Average Rrs in linear space over the valid pixels of each b x b block.
    Returns the block-mean spectra (lat_c, lon_c, 4), NaN where fewer than
    `min_valid` pixels are valid, and the number of valid pixels per block.
    """
    nlat, nlon, nband = rrs.shape
    valid = np.isfinite(rrs).all(axis=-1)
    n = valid.reshape(nlat // b, b, nlon // b, b).sum(axis=(1, 3))
    ok = n >= max(min_valid, 1)
    denom = np.where(n > 0, n, 1).astype(np.float64)

    out = np.full((nlat // b, nlon // b, nband), np.nan)
    for k in range(nband):
        band = np.where(valid, rrs[..., k], 0.0).astype(np.float64)
        mean = band.reshape(nlat // b, b, nlon // b, b).sum(axis=(1, 3)) / denom
        mean[~ok] = np.nan
        out[..., k] = mean
    return out, n


def create_year_file(path, lat_c, lon_c, args, res_label):
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.title = "Global daily chlorophyll-a with per-pixel uncertainty from a Bayesian neural network"
    ds.institution = "KAUST / SUSTech"
    ds.source = ("Heteroscedastic Bayesian neural network applied to ESA OC-CCI v6.0 "
                 "merged remote sensing reflectance (443, 490, 510, 560 nm)")
    ds.references = "<paper DOI>"
    ds.license = "CC-BY-4.0"
    ds.Conventions = "CF-1.8"
    ds.date_created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ds.spatial_resolution = res_label
    ds.bin_factor = np.int32(args.bin)
    ds.min_valid_pixels = np.int32(max(args.min_valid, 1))
    ds.mc_samples = np.int32(args.mc_samples)
    ds.processing = (
        f"OC-CCI v6.0 Rrs averaged in linear space over the valid 4 km pixels of each "
        f"non-overlapping {args.bin}x{args.bin} block (a pixel is valid when all four bands "
        "are valid); the BNN is applied to the block-mean spectrum with "
        f"{args.mc_samples} posterior samples. Uncertainties are 1-sigma in log10 space; "
        "total uncertainty = sqrt(unc_epistemic**2 + unc_aleatoric**2).")

    ds.createDimension("time", None)
    ds.createDimension("lat", len(lat_c))
    ds.createDimension("lon", len(lon_c))

    v = ds.createVariable("time", "i4", ("time",))
    v.units, v.calendar = "days since 1970-01-01 00:00:00", "standard"
    v.standard_name, v.long_name, v.axis = "time", "time", "T"

    v = ds.createVariable("lat", "f4", ("lat",))
    v.units, v.standard_name, v.long_name, v.axis = "degrees_north", "latitude", "latitude (cell centre)", "Y"
    v[:] = lat_c

    v = ds.createVariable("lon", "f4", ("lon",))
    v.units, v.standard_name, v.long_name, v.axis = "degrees_east", "longitude", "longitude (cell centre)", "X"
    v[:] = lon_c

    dims, chunks = ("time", "lat", "lon"), (1, len(lat_c), len(lon_c))
    for name, long_name, vmin, vmax in (
            ("chl_mean", "Posterior predictive mean of log10 chlorophyll-a concentration", -3.0, 3.0),
            ("unc_epistemic", "Epistemic uncertainty of log10 chlorophyll-a (1-sigma)", 0.0, 5.0),
            ("unc_aleatoric", "Aleatoric uncertainty of log10 chlorophyll-a (1-sigma)", 0.0, 5.0)):
        v = ds.createVariable(name, "f4", dims, fill_value=FILL,
                              zlib=True, complevel=4, chunksizes=chunks)
        v.units, v.long_name = "log10(mg m-3)", long_name
        v.valid_min, v.valid_max = np.float32(vmin), np.float32(vmax)

    n_type = "i1" if args.bin ** 2 <= 127 else "i2"
    v = ds.createVariable("n_valid", n_type, dims, zlib=True, complevel=4, chunksizes=chunks)
    v.units, v.long_name = "1", "Number of valid 4 km pixels averaged in the cell"
    v.valid_min, v.valid_max = np.array([0, args.bin ** 2], dtype=n_type)
    return ds


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)

    predictive, x_mean, x_std = load_trained_model(args.model_dir, args.mc_samples)
    files = find_daily_files(args.input_dir)
    if not files:
        raise SystemExit(f"no NetCDF files found in {args.input_dir}")

    lat, lon = read_grid(next(iter(files.values())))
    # OC-CCI stores latitude north to south; outputs are written south to north
    flip_lat, flip_lon = lat[0] > lat[-1], lon[0] > lon[-1]
    lat = lat[::-1] if flip_lat else lat
    lon = lon[::-1] if flip_lon else lon
    b = args.bin
    if len(lat) % b or len(lon) % b:
        raise SystemExit(f"grid {len(lat)} x {len(lon)} is not divisible by --bin {b}")
    lat_c = lat.reshape(-1, b).mean(axis=1)
    lon_c = lon.reshape(-1, b).mean(axis=1)
    res_label = "4km" if b == 1 else f"{b / 24:g}deg"

    for year in parse_years(args.years):
        days = sorted(d for d in files if d.year == year)
        if not days:
            print(f"{year}: no input files, skipped")
            continue

        out_path = os.path.join(args.output_dir, f"BNN_OC-CCI_chl_uncertainty_{res_label}_{year}.nc")
        if os.path.exists(out_path):
            ds = nc.Dataset(out_path, "a")
            if ds.dimensions["lat"].size != len(lat_c) or ds.dimensions["lon"].size != len(lon_c):
                raise SystemExit(f"{out_path} exists with a different grid")
        else:
            ds = create_year_file(out_path, lat_c, lon_c, args, res_label)

        try:
            done = set(int(t) for t in np.asarray(ds["time"][:]))
            for date in tqdm(days, desc=str(year), unit="day"):
                tval = (date - EPOCH).days
                if tval in done:
                    continue

                rrs = read_rrs(files[date])
                rrs = rrs[::-1] if flip_lat else rrs
                rrs = rrs[:, ::-1] if flip_lon else rrs
                rrs_c, n_valid = bin_rrs(rrs, b, args.min_valid)
                x, invalid = standardise(rrs_c.reshape(-1, rrs_c.shape[-1]), x_mean, x_std)
                mu, epi, ale = predict(predictive, x, invalid)

                k = ds.dimensions["time"].size
                ds["time"][k] = tval
                for name, a in (("chl_mean", mu), ("unc_epistemic", epi), ("unc_aleatoric", ale)):
                    a = a.reshape(len(lat_c), len(lon_c))
                    ds[name][k] = np.where(np.isnan(a), FILL, a)
                ds["n_valid"][k] = n_valid
                ds.sync()
                done.add(tval)

            t_all = np.asarray(ds["time"][:])
            ds.time_coverage_start = (EPOCH + timedelta(days=int(t_all.min()))).strftime("%Y-%m-%d")
            ds.time_coverage_end = (EPOCH + timedelta(days=int(t_all.max()))).strftime("%Y-%m-%d")
        finally:
            ds.close()
        print(f"{year}: {len(done)} days -> {out_path}")


if __name__ == "__main__":
    main()
