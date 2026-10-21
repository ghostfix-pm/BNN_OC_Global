"""
Helpers for the OC-CCI v6.0 daily Rrs input of ``predict_global.py``.

Date parsing, year selection, grid orientation and the block averaging are
kept here (NumPy + standard library only) so that they can be exercised
without reading a NetCDF file.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np

# Daily files are named e.g. ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210101-fv6.0.nc
DATE_IN_NAME = re.compile(r"-(\d{8})-fv")
YEAR_TOKEN = re.compile(r"^(\d{4})$")
YEAR_RANGE_TOKEN = re.compile(r"^(\d{4})-(\d{4})$")
DATE_TOKEN = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")

# The yearly files written by predict_global.py must not be mistaken for input
OUTPUT_STEM = "BNN_OC-CCI_chl_uncertainty_"


def date_from_name(path):
    """Date encoded in a daily file name, or None when the name carries none."""
    match = DATE_IN_NAME.search(os.path.basename(path))
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d")


def is_output_file(path):
    """True for the yearly files written by predict_global.py itself."""
    return os.path.basename(path).startswith(OUTPUT_STEM)


def parse_years(tokens):
    """
    Expand year tokens (``2021`` or ``2000-2025``) into a sorted list of years.

    A typo must not turn into a silently empty run: malformed tokens and
    reversed ranges raise ``ValueError`` instead of being ignored.
    """
    if not tokens:
        raise ValueError("no years given: pass at least one year or a year range")
    years = set()
    for token in tokens:
        text = str(token).strip()
        match = YEAR_TOKEN.match(text)
        if match is not None:
            years.add(int(match.group(1)))
            continue
        match = YEAR_RANGE_TOKEN.match(text)
        if match is not None:
            first, last = int(match.group(1)), int(match.group(2))
            if last < first:
                raise ValueError(
                    f"year range {text!r} runs backwards: the first year must not be "
                    f"later than the last ({first} > {last})")
            years.update(range(first, last + 1))
            continue
        raise ValueError(
            f"cannot parse year token {text!r}: expected a year such as 2021 or a range "
            "such as 2000-2025")
    return sorted(years)


def parse_date(value, option="--start-date/--end-date"):
    """Parse ``YYYY-MM-DD`` (or ``YYYYMMDD``) into a datetime."""
    text = str(value).strip()
    match = DATE_TOKEN.match(text)
    if match is None:
        raise ValueError(f"cannot parse {option} value {text!r}: expected YYYY-MM-DD")
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day)
    except ValueError as exc:
        raise ValueError(f"invalid {option} value {text!r}: {exc}") from None


def select_days(days, start=None, end=None):
    """Keep the days inside the inclusive ``[start, end]`` window."""
    if start is not None and end is not None and end < start:
        raise ValueError(
            f"--end-date ({end:%Y-%m-%d}) is earlier than --start-date ({start:%Y-%m-%d})")
    selected = []
    for day in days:
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        selected.append(day)
    return selected


def flip_flags(lat, lon):
    """
    Which axes run the other way round. OC-CCI stores latitude north to south
    and longitude west to east; the outputs are written south to north.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    return bool(lat.size > 1 and lat[0] > lat[-1]), bool(lon.size > 1 and lon[0] > lon[-1])


def orient(values, flip):
    """Reverse an axis so that it runs in the output direction."""
    return np.asarray(values)[::-1] if flip else values


def orient_rrs(rrs, flip_lat, flip_lon):
    """
    Bring a (lat, lon, band) cube into the same orientation as the oriented
    latitude/longitude axes. One helper so that the grid and the data can never
    disagree about their flips.
    """
    rrs = np.asarray(rrs)
    if rrs.ndim != 3:
        raise ValueError(f"the Rrs cube must be 3-D (lat, lon, band), got shape {rrs.shape}")
    if flip_lat:
        rrs = rrs[::-1]
    if flip_lon:
        rrs = rrs[:, ::-1]
    return rrs


def cell_centres(axis_values, b):
    """Centres of the ``b``-wide blocks of an axis (mean of each block)."""
    axis = np.asarray(axis_values, dtype=np.float64)
    b = int(b)
    if b < 1:
        raise ValueError(f"the block size must be at least 1, got {b}")
    if axis.ndim != 1:
        raise ValueError(f"a grid axis must be 1-D, got shape {axis.shape}")
    if axis.size % b:
        raise ValueError(
            f"axis of {axis.size} points is not divisible by the block size {b}")
    return axis.reshape(-1, b).mean(axis=1)


def resolution_label(b):
    """``4km`` at the native resolution, otherwise the block size in degrees."""
    b = int(b)
    if b < 1:
        raise ValueError(f"the block size must be at least 1, got {b}")
    return "4km" if b == 1 else f"{b / 24:g}deg"


def block_mean(rrs, b, min_valid=1):
    """
    Average Rrs in linear space over the valid pixels of each b x b block.

    A pixel is valid when all bands are finite (NaN marks missing data); a
    block is retrieved when at least ``min_valid`` of its pixels are valid.
    Returns the block-mean spectra (lat_c, lon_c, band) with NaN where a block
    is not retrieved, and the number of valid pixels per block.
    """
    rrs = np.asarray(rrs, dtype=np.float64)
    b = int(b)
    if rrs.ndim != 3:
        raise ValueError(f"the Rrs cube must be 3-D (lat, lon, band), got shape {rrs.shape}")
    if b < 1:
        raise ValueError(f"the block size must be at least 1, got {b}")
    min_valid = int(min_valid)
    if min_valid < 1:
        raise ValueError(f"the minimum number of valid pixels must be at least 1, got {min_valid}")

    n_lat, n_lon, n_band = rrs.shape
    if n_lat % b or n_lon % b:
        raise ValueError(
            f"grid {n_lat} x {n_lon} is not divisible by the block size {b}")

    valid = np.isfinite(rrs).all(axis=-1)
    n_valid = valid.reshape(n_lat // b, b, n_lon // b, b).sum(axis=(1, 3))
    retrieved = n_valid >= min_valid
    denominator = np.where(n_valid > 0, n_valid, 1).astype(np.float64)

    means = np.full((n_lat // b, n_lon // b, n_band), np.nan)
    for band in range(n_band):
        values = np.where(valid, rrs[..., band], 0.0)
        mean = values.reshape(n_lat // b, b, n_lon // b, b).sum(axis=(1, 3)) / denominator
        mean[~retrieved] = np.nan
        means[..., band] = mean
    return means, n_valid
