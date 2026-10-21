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

Re-running the script skips days that are already in the yearly file; pass
--overwrite to recompute them. --list-days prints the plan (files, grid, days
already written) and exits without reading the model.

Example:
  python predict_global.py --input-dir /data/occci_v6.0_rrs_daily \\
      --output-dir output --years 2000-2025
  python predict_global.py --input-dir /data/occci --output-dir out \\
      --start-date 2021-01-01 --end-date 2021-12-31 --list-days
"""

import argparse
import glob
import os
from datetime import datetime, timedelta, timezone

import netCDF4 as nc
import numpy as np
import torch
from tqdm import tqdm

from bnn_model import RRS_VARIABLES, load_trained_model, predict, standardise
from oc_cci import (OUTPUT_STEM, block_mean, cell_centres, date_from_name, flip_flags,
                    is_output_file, orient, orient_rrs, parse_date, parse_years,
                    resolution_label, select_days)

FILL = np.float32(9.96921e+36)
EPOCH = datetime(1970, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="folder with OC-CCI v6.0 daily Rrs NetCDF files (searched recursively)")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model-dir", default="model",
                   help="folder with the trained model files written by train.py")
    p.add_argument("--years", nargs="+",
                   help="years to process, e.g. 2021 2022 or 2000-2025; may be omitted when "
                        "--start-date/--end-date are given")
    p.add_argument("--start-date", default=None,
                   help="first day to process, YYYY-MM-DD (inclusive)")
    p.add_argument("--end-date", default=None,
                   help="last day to process, YYYY-MM-DD (inclusive)")
    p.add_argument("--bin", type=int, default=6,
                   help="block size in 4 km pixels (6 = 0.25 degree, 1 = native 4 km)")
    p.add_argument("--min-valid", type=int, default=1,
                   help="minimum valid 4 km pixels for a cell to be retrieved")
    p.add_argument("--mc-samples", type=int, default=100,
                   help="posterior samples per retrieval")
    p.add_argument("--mc-seed", type=int, default=42,
                   help="Monte Carlo seed used for the posterior samples")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute the days already present in the yearly files "
                        "(by default they are kept)")
    p.add_argument("--list-days", action="store_true",
                   help="print the files, grid and days that would be processed, then exit "
                        "(the trained model is not read)")
    return p.parse_args()


def validate_args(args):
    """Reject argument values that would fail deep inside the run."""
    if args.bin < 1:
        raise SystemExit(f"error: --bin must be at least 1, got {args.bin}")
    if args.min_valid < 1:
        raise SystemExit(f"error: --min-valid must be at least 1, got {args.min_valid}")
    if args.mc_samples < 1:
        raise SystemExit(f"error: --mc-samples must be at least 1, got {args.mc_samples}")
    if args.mc_seed is not None and args.mc_seed < 0:
        raise SystemExit(f"error: --mc-seed must not be negative, got {args.mc_seed}")
    if args.threads is not None and args.threads < 1:
        raise SystemExit(f"error: --threads must be at least 1, got {args.threads}")
    if not args.years and not args.start_date and not args.end_date:
        raise SystemExit("error: give --years and/or --start-date / --end-date")


def parse_window(args):
    """
    Years to process plus the inclusive date window.

    ``--start-date`` / ``--end-date`` narrow ``--years``; when ``--years`` is
    omitted they define it (from the first year of the start bound to the last
    year of the end bound).
    """
    start = parse_date(args.start_date, "--start-date") if args.start_date else None
    end = parse_date(args.end_date, "--end-date") if args.end_date else None
    if start is not None and end is not None and end < start:
        raise ValueError(f"--end-date ({end:%Y-%m-%d}) is earlier than "
                         f"--start-date ({start:%Y-%m-%d})")
    if args.years:
        years = parse_years(args.years)
    else:
        first_bound = start if start is not None else end
        last_bound = end if end is not None else start
        if first_bound is None or last_bound is None:
            raise ValueError("give --years and/or --start-date / --end-date")
        years = list(range(first_bound.year, last_bound.year + 1))
    return years, start, end


def file_date(path):
    """Date of a daily file, from its name or, failing that, its time variable."""
    date = date_from_name(path)
    if date is not None:
        return date
    with nc.Dataset(path) as ds:
        if "time" not in ds.variables:
            raise ValueError(f"{path} carries neither a date in its name nor a time variable")
        t = ds["time"]
        d = nc.num2date(t[0], t.units, getattr(t, "calendar", "standard"))
    return datetime(d.year, d.month, d.day)


def find_daily_files(input_dir):
    """
    Map the date of every daily Rrs file found under ``input_dir`` to its path.

    The yearly files written by this script are skipped, so an output directory
    placed inside the input directory cannot be mistaken for input; two input
    files for the same day are an error rather than a silent coin toss.
    """
    files, skipped = {}, 0
    for path in sorted(glob.glob(os.path.join(input_dir, "**", "*.nc"), recursive=True)):
        if is_output_file(path):
            skipped += 1
            continue
        date = file_date(path)
        if date in files:
            raise ValueError(f"two input files for {date:%Y-%m-%d}: "
                             f"{files[date]} and {path}")
        files[date] = path
    if skipped:
        print(f"skipped {skipped} file(s) named {OUTPUT_STEM}* in {input_dir} "
              "(output of this script)")
    return files


def read_rrs(path):
    """Return the four input bands as a (lat, lon, 4) float32 array, NaN = missing."""
    with nc.Dataset(path) as ds:
        missing = [name for name in RRS_VARIABLES if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} does not contain {missing}: is it an OC-CCI v6.0 "
                             "Rrs file?")
        bands = []
        for name in RRS_VARIABLES:
            band = ds[name][:]
            if band.ndim == 3:          # (time, lat, lon) with a single time step
                band = band[0]
            if band.ndim != 2:
                raise ValueError(f"{path}: {name} has shape {tuple(band.shape)}, expected "
                                 "(lat, lon) or (time=1, lat, lon)")
            bands.append(np.ma.filled(band.astype(np.float32), np.nan))
    return np.stack(bands, axis=-1)


def read_grid(path):
    with nc.Dataset(path) as ds:
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
    return lat, lon


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
    ds.min_valid_pixels = np.int32(args.min_valid)
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


def check_existing_file(ds, path, args, res_label, n_lat, n_lon):
    """
    Refuse to add days to a yearly file that was written with other settings,
    so that one file never mixes two processing chains.
    """
    if any(name not in ds.dimensions for name in ("time", "lat", "lon")):
        raise ValueError(
            f"{path} does not look like a yearly file written by this script: the time / lat / "
            "lon dimensions are not all present. Use another --output-dir or --overwrite")
    if ds.dimensions["lat"].size != n_lat or ds.dimensions["lon"].size != n_lon:
        raise ValueError(
            f"{path} holds a {ds.dimensions['lat'].size} x {ds.dimensions['lon'].size} grid "
            f"but {n_lat} x {n_lon} is expected: use another --output-dir or --overwrite")
    missing = [name for name in ("time", "lat", "lon", "chl_mean", "unc_epistemic",
                                 "unc_aleatoric", "n_valid") if name not in ds.variables]
    if missing:
        raise ValueError(
            f"{path} does not look like a yearly file written by this script: {missing} "
            "is/are missing. Use another --output-dir or --overwrite")
    expected = {"spatial_resolution": res_label, "bin_factor": int(args.bin),
                "min_valid_pixels": int(args.min_valid), "mc_samples": int(args.mc_samples)}
    problems = []
    for name, wanted in expected.items():
        if name not in ds.ncattrs():
            problems.append(f"{name} is not recorded")
            continue
        on_disk = ds.getncattr(name)
        if isinstance(wanted, int):
            if int(on_disk) != wanted:
                problems.append(f"{name}={int(on_disk)} on disk, {wanted} requested")
        elif str(on_disk) != wanted:
            problems.append(f"{name}={str(on_disk)!r} on disk, {wanted!r} requested")
    if problems:
        raise ValueError(
            f"{path} was written with different settings ({'; '.join(problems)}): pass "
            "--overwrite to recompute it or use another --output-dir")


def describe_plan(files, years, days_by_year, output_dir, res_label, args, flip_lat, lat_c, lon_c):
    """Printable summary of what a run would do (--list-days)."""
    all_days = [day for year in years for day in days_by_year.get(year, [])]
    lines = [
        f"input       : {args.input_dir}  ({len(files)} daily files, "
        f"{min(files):%Y-%m-%d} .. {max(files):%Y-%m-%d})",
        f"grid        : {len(lat_c) * args.bin} x {len(lon_c) * args.bin} -> "
        f"{len(lat_c)} x {len(lon_c)} cells of {res_label} (bin {args.bin}, "
        f"at least {args.min_valid} valid pixel(s))",
        f"orientation : input latitude {'north to south' if flip_lat else 'south to north'}"
        " (flipped), output written south to north",
        f"model       : {args.model_dir} ({args.mc_samples} posterior samples, "
        f"seed {args.mc_seed})",
    ]
    pending = 0
    for year in years:
        days = days_by_year.get(year, [])
        if not days:
            lines.append(f"{year}: no input files")
            continue
        out_path = os.path.join(output_dir, f"{OUTPUT_STEM}{res_label}_{year}.nc")
        done = 0
        if os.path.exists(out_path):
            with nc.Dataset(out_path) as ds:
                done = int(ds.dimensions["time"].size)
        state = "to be created" if not done else f"{done}/{len(days)} days already written"
        if args.overwrite and done:
            pending += len(days)
            state += " (recomputed: --overwrite)"
        else:
            pending += len(days) - done
        lines.append(f"{year}: {len(days)} days ({days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}) -> "
                     f"{out_path} [{state}]")
    lines.append(f"total: {len(all_days)} days in {len(years)} year(s), {pending} to process")
    return "\n".join(lines)


def main():
    args = parse_args()
    validate_args(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        years, start, end = parse_window(args)
        files = find_daily_files(args.input_dir)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    if not files:
        raise SystemExit(f"error: no NetCDF files found in {args.input_dir}")

    days_by_year = {year: select_days(sorted(d for d in files if d.year == year), start, end)
                    for year in years}
    if not any(days_by_year.values()):
        raise SystemExit(f"error: no input file falls in the requested period "
                         f"({years[0]} .. {years[-1]}"
                         f"{'' if start is None else f' from {start:%Y-%m-%d}'}"
                         f"{'' if end is None else f' to {end:%Y-%m-%d}'})")

    first_path = files[min(files)]
    lat, lon = read_grid(first_path)
    flip_lat, flip_lon = flip_flags(lat, lon)
    # OC-CCI stores latitude north to south; outputs are written south to north
    lat, lon = orient(lat, flip_lat), orient(lon, flip_lon)
    b = args.bin
    try:
        lat_c, lon_c = cell_centres(lat, b), cell_centres(lon, b)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    res_label = resolution_label(b)
    n_lat, n_lon = len(lat_c), len(lon_c)

    if args.list_days:
        print(describe_plan(files, years, days_by_year, args.output_dir, res_label, args,
                            flip_lat, lat_c, lon_c))
        return

    try:
        predictive, x_mean, x_std = load_trained_model(args.model_dir, args.mc_samples)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    for year in years:
        days = days_by_year[year]
        if not days:
            print(f"{year}: no input files, skipped")
            continue

        out_path = os.path.join(args.output_dir, f"{OUTPUT_STEM}{res_label}_{year}.nc")
        if args.overwrite and os.path.exists(out_path):
            os.remove(out_path)
            print(f"{year}: --overwrite, recomputing {out_path}")
        if os.path.exists(out_path):
            ds = nc.Dataset(out_path, "a")
            try:
                check_existing_file(ds, out_path, args, res_label, n_lat, n_lon)
            except ValueError as exc:
                ds.close()
                raise SystemExit(f"error: {exc}") from None
        else:
            ds = create_year_file(out_path, lat_c, lon_c, args, res_label)

        try:
            done = set(int(t) for t in np.asarray(ds["time"][:]))
            for date in tqdm(days, desc=str(year), unit="day"):
                tval = (date - EPOCH).days
                if tval in done:
                    continue

                rrs = read_rrs(files[date])
                if rrs.shape[:2] != (len(lat), len(lon)):
                    raise ValueError(
                        f"{files[date]}: grid {rrs.shape[0]} x {rrs.shape[1]} differs from the "
                        f"{len(lat)} x {len(lon)} of {first_path}")
                rrs = orient_rrs(rrs, flip_lat, flip_lon)
                rrs_c, n_valid = block_mean(rrs, b, args.min_valid)
                x, invalid = standardise(rrs_c.reshape(-1, rrs_c.shape[-1]), x_mean, x_std)
                mu, epi, ale = predict(predictive, x, invalid, seed=args.mc_seed)

                k = ds.dimensions["time"].size
                ds["time"][k] = tval
                for name, a in (("chl_mean", mu), ("unc_epistemic", epi), ("unc_aleatoric", ale)):
                    a = a.reshape(n_lat, n_lon)
                    ds[name][k] = np.where(np.isnan(a), FILL, a)
                ds["n_valid"][k] = n_valid
                ds.sync()
                done.add(tval)

            t_all = np.asarray(ds["time"][:])
            if t_all.size:
                ds.time_coverage_start = (EPOCH + timedelta(
                    days=int(t_all.min()))).strftime("%Y-%m-%d")
                ds.time_coverage_end = (EPOCH + timedelta(
                    days=int(t_all.max()))).strftime("%Y-%m-%d")
        except ValueError as exc:
            # the finally clause below closes the file exactly once
            raise SystemExit(f"error: {exc}") from None
        finally:
            ds.close()
        print(f"{year}: {len(done)} days -> {out_path}")


if __name__ == "__main__":
    main()
