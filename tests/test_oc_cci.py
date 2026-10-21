"""Tests for the OC-CCI helpers: dates, year selection, orientation, block means."""

from datetime import datetime

import numpy as np
import pytest

from oc_cci import (block_mean, cell_centres, date_from_name, flip_flags, is_output_file,
                    orient, orient_rrs, parse_date, parse_years, resolution_label, select_days)


# ---------------------------------------------------------------------------
# dates and file names
# ---------------------------------------------------------------------------
def test_date_from_name_reads_the_oc_cci_convention():
    name = "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210615-fv6.0.nc"
    assert date_from_name("/data/" + name) == datetime(2021, 6, 15)
    assert date_from_name("/data/anything_else.nc") is None


def test_is_output_file_recognises_our_own_yearly_files():
    assert is_output_file("BNN_OC-CCI_chl_uncertainty_0.25deg_2021.nc")
    assert not is_output_file("ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km-20210101-fv6.0.nc")


@pytest.mark.parametrize("value,expected", [
    ("2021-01-02", datetime(2021, 1, 2)),
    ("20210102", datetime(2021, 1, 2)),
    (" 2020-12-31 ", datetime(2020, 12, 31)),
])
def test_parse_date_accepts_the_documented_forms(value, expected):
    assert parse_date(value) == expected


@pytest.mark.parametrize("value", ["2021/01/02", "2021-13-01", "2021-02-30", "yesterday", ""])
def test_parse_date_rejects_bad_values(value):
    with pytest.raises(ValueError) as exc:
        parse_date(value)
    assert "--start-date/--end-date" in str(exc.value)


# ---------------------------------------------------------------------------
# year selection
# ---------------------------------------------------------------------------
def test_parse_years_expands_single_years_and_ranges():
    assert parse_years(["2021"]) == [2021]
    assert parse_years(["2000-2002"]) == [2000, 2001, 2002]
    assert parse_years(["2022", "2000-2001", "2022"]) == [2000, 2001, 2022]   # sorted, unique


@pytest.mark.parametrize("tokens", [["2025-2000"], ["2021-"], ["-2021"], ["twenty"], ["20211"],
                                    ["2021-2022-2023"], ["2021 2022"]])
def test_parse_years_rejects_malformed_and_reversed_tokens(tokens):
    """A typo must not become a silent no-op run."""
    with pytest.raises(ValueError):
        parse_years(tokens)


def test_parse_years_says_which_way_a_range_is_wrong():
    with pytest.raises(ValueError) as exc:
        parse_years(["2025-2000"])
    assert "runs backwards" in str(exc.value)


def test_parse_years_needs_at_least_one_token():
    with pytest.raises(ValueError) as exc:
        parse_years([])
    assert "no years given" in str(exc.value)


def test_select_days_applies_an_inclusive_window():
    days = [datetime(2021, 1, 1), datetime(2021, 6, 15), datetime(2021, 12, 31)]
    assert select_days(days, start=datetime(2021, 1, 1), end=datetime(2021, 12, 31)) == days
    assert select_days(days, start=datetime(2021, 2, 1)) == days[1:]
    assert select_days(days, end=datetime(2021, 6, 15)) == days[:2]
    assert select_days(days, start=datetime(2021, 6, 16)) == [days[2]]
    assert select_days(days, start=datetime(2022, 1, 1)) == []


def test_select_days_rejects_a_backwards_window():
    with pytest.raises(ValueError) as exc:
        select_days([], start=datetime(2021, 2, 1), end=datetime(2021, 1, 1))
    assert "earlier than" in str(exc.value)


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------
def test_flip_flags_detect_the_oc_cci_latitude_order():
    lat_north_first = np.array([60.0, 40.0, 20.0])
    lat_south_first = np.array([20.0, 40.0, 60.0])
    assert flip_flags(lat_north_first, np.array([-10.0, 0.0]))[0] is True
    assert flip_flags(lat_south_first, np.array([-10.0, 0.0]))[0] is False
    assert flip_flags(np.array([20.0]), np.array([0.0, 10.0]))[0] is False


def test_orient_reverses_only_when_asked():
    values = np.arange(4)
    assert np.array_equal(orient(values, False), values)
    assert np.array_equal(orient(values, True), values[::-1])


def test_orient_rrs_moves_the_data_and_the_axes_the_same_way():
    rrs = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    lat, lon = np.array([60.0, 50.0]), np.array([10.0, 20.0, 30.0])
    flip_lat, flip_lon = flip_flags(lat, lon)
    oriented = orient_rrs(rrs, flip_lat, flip_lon)
    assert np.array_equal(oriented, rrs[::-1])          # latitude only
    assert np.array_equal(oriented[0], rrs[1])          # the first output row is the south row
    assert np.array_equal(orient_rrs(rrs, False, True), rrs[:, ::-1])
    assert np.array_equal(orient_rrs(rrs, False, False), rrs)


def test_orient_rrs_rejects_a_non_cube():
    with pytest.raises(ValueError):
        orient_rrs(np.zeros((2, 2)), True, False)


# ---------------------------------------------------------------------------
# grid geometry
# ---------------------------------------------------------------------------
def test_cell_centres_average_each_block():
    axis = np.arange(0.0, 8.0)
    assert np.array_equal(cell_centres(axis, 2), [0.5, 2.5, 4.5, 6.5])
    assert np.array_equal(cell_centres(axis, 8), [3.5])


@pytest.mark.parametrize("b", [0, -3])
def test_cell_centres_reject_a_bad_block_size(b):
    with pytest.raises(ValueError) as exc:
        cell_centres(np.arange(4.0), b)
    assert "at least 1" in str(exc.value)


def test_cell_centres_reject_an_indivisible_axis():
    with pytest.raises(ValueError) as exc:
        cell_centres(np.arange(7.0), 4)
    assert "not divisible" in str(exc.value)


def test_resolution_label():
    assert resolution_label(1) == "4km"
    assert resolution_label(6) == "0.25deg"
    with pytest.raises(ValueError):
        resolution_label(0)


# ---------------------------------------------------------------------------
# block_mean
# ---------------------------------------------------------------------------
def brute_force_block_mean(rrs, b, min_valid):
    """Independent reference implementation, written as plain loops."""
    n_lat, n_lon, n_band = rrs.shape
    means = np.full((n_lat // b, n_lon // b, n_band), np.nan)
    counts = np.zeros((n_lat // b, n_lon // b), dtype=int)
    for i in range(n_lat // b):
        for j in range(n_lon // b):
            block = rrs[i * b:(i + 1) * b, j * b:(j + 1) * b, :]
            valid = np.isfinite(block).all(axis=-1)
            counts[i, j] = int(valid.sum())
            if counts[i, j] >= min_valid:
                means[i, j] = block[valid].mean(axis=0)
    return means, counts


def test_block_mean_matches_a_brute_force_reference():
    rng = np.random.default_rng(11)
    rrs = rng.uniform(1e-3, 1e-2, size=(12, 16, 4))
    mask = rng.random(rrs.shape[:-1]) < 0.25
    rrs[mask] = np.nan
    for b in (1, 2, 4):
        for min_valid in (1, 2, 5):
            got, counts = block_mean(rrs, b, min_valid)
            expected, expected_counts = brute_force_block_mean(rrs, b, min_valid)
            assert np.array_equal(counts, expected_counts)
            np.testing.assert_allclose(got, expected, equal_nan=True)


def test_block_mean_marks_blocks_below_min_valid():
    rrs = np.full((4, 4, 4), 0.005)
    rrs[0, 0] = np.nan                      # one invalid pixel in the first block
    means, counts = block_mean(rrs, 2, 1)
    assert counts[0, 0] == 3 and np.isfinite(means[0, 0]).all()
    means, _ = block_mean(rrs, 2, 4)
    assert np.isnan(means[0, 0]).all()       # only 3 of 4 pixels valid
    assert np.isfinite(means[1, 1]).all()


def test_block_mean_requires_all_bands_to_be_valid():
    rrs = np.full((2, 2, 4), 0.005)
    rrs[0, 0, 2] = np.nan
    _, counts = block_mean(rrs, 2, 1)
    assert counts[0, 0] == 3                 # the pixel is dropped, not averaged per band


def test_block_mean_is_nan_free_where_a_block_is_empty():
    rrs = np.full((2, 2, 4), np.nan)
    means, counts = block_mean(rrs, 2, 1)
    assert counts[0, 0] == 0
    assert np.isnan(means).all()             # never a division by zero


def test_block_mean_keeps_the_band_axis_and_shape():
    rrs = np.ones((6, 6, 4))
    means, counts = block_mean(rrs, 3, 1)
    assert means.shape == (2, 2, 4) and counts.shape == (2, 2)
    assert np.allclose(means, 1.0) and np.all(counts == 9)


@pytest.mark.parametrize("b,min_valid", [(0, 1), (-2, 1), (3, 1), (2, 0), (2, -1)])
def test_block_mean_rejects_bad_arguments(b, min_valid):
    # b=3 does not divide the 4 x 4 grid; b<=0 / min_valid<1 are invalid outright
    with pytest.raises(ValueError):
        block_mean(np.ones((4, 4, 4)), b, min_valid)


def test_block_mean_rejects_a_non_cube():
    with pytest.raises(ValueError):
        block_mean(np.ones((4, 4)), 2, 1)
