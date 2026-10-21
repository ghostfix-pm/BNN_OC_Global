"""Tests for predict_global.py: the plan, the yearly output files and the CLI guards."""

import sys
from datetime import datetime

import numpy as np
import pytest

from conftest import write_daily_file

import predict_global

netCDF4 = pytest.importorskip("netCDF4")
nc = netCDF4


def run_main(argv):
    """Call predict_global.main() with a given command line."""
    previous = sys.argv
    sys.argv = ["predict_global.py", *argv]
    try:
        predict_global.main()
    finally:
        sys.argv = previous


BASE = ["--input-dir", None, "--output-dir", None, "--model-dir", None,
        "--years", "2021", "--mc-samples", "4"]


def args_for(occi_dir, output_dir, model_dir, extra=(), years="2021"):
    return ["--input-dir", str(occi_dir), "--output-dir", str(output_dir),
            "--model-dir", str(model_dir), "--years", years, "--mc-samples", "4", *extra]


def output_file(output_dir, year=2021, res="0.25deg"):
    return output_dir / f"BNN_OC-CCI_chl_uncertainty_{res}_{year}.nc"


@pytest.fixture
def run(tmp_path, occi_dir, model_dir):
    """First run over the synthetic input, returning (output_dir, path)."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    output_dir = tmp_path / "out"
    run_main(args_for(occi_dir, output_dir, model_dir))
    return output_dir, output_file(output_dir)


# ---------------------------------------------------------------------------
# argument guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag,value,message", [
    ("--bin", "0", "--bin must be at least 1"),
    ("--min-valid", "0", "--min-valid must be at least 1"),
    ("--mc-samples", "0", "--mc-samples must be at least 1"),
    ("--threads", "0", "--threads must be at least 1"),
])
def test_validate_args_rejects_bad_values(flag, value, message):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", "/nowhere", "--years", "2021", flag, value])
    assert message in str(exc.value)


def test_needs_years_or_a_date_window():
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", "/nowhere"])
    assert "give --years" in str(exc.value)


def test_rejects_a_backwards_date_window(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(tmp_path), "--start-date", "2021-06-01",
                  "--end-date", "2021-01-01"])
    assert "earlier than" in str(exc.value)


def test_rejects_a_malformed_year(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(tmp_path), "--years", "2021-"])
    assert "cannot parse year token" in str(exc.value)


def test_reversed_year_range_is_not_a_silent_no_op(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(tmp_path), "--years", "2025-2000"])
    assert "runs backwards" in str(exc.value)


def test_reports_a_missing_input_directory(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(tmp_path / "absent"), "--years", "2021"])
    assert "no NetCDF files found" in str(exc.value)


def test_reports_a_period_without_input_files(tmp_path, occi_dir):
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(occi_dir), "--output-dir", str(tmp_path / "o"),
                  "--years", "1999"])
    assert "no input file falls in the requested period" in str(exc.value)


def test_reports_a_missing_model_before_writing_anything(tmp_path, occi_dir):
    output_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, output_dir, tmp_path / "no-model"))
    assert "train the model first" in str(exc.value)
    assert not output_dir.exists() or not list(output_dir.glob("*.nc"))


# ---------------------------------------------------------------------------
# --list-days
# ---------------------------------------------------------------------------
def test_list_days_prints_the_plan_without_the_model(tmp_path, occi_dir):
    output_dir = tmp_path / "out"
    run_main(args_for(occi_dir, output_dir, tmp_path / "does-not-exist", extra=["--list-days"]))
    assert not list(output_dir.glob("*.nc"))


def test_list_days_reports_the_grid_and_the_days(tmp_path, occi_dir, capsys):
    run_main(args_for(occi_dir, tmp_path / "out", tmp_path / "model", extra=["--list-days"]))
    out = capsys.readouterr().out
    assert "48 x 48 -> 8 x 8 cells of 0.25deg" in out
    assert "north to south" in out                       # the input order is detected
    assert "2021: 2 days (2021-01-01 .. 2021-01-02)" in out
    assert "total: 2 days in 1 year(s), 2 to process" in out


def test_list_days_ignores_the_other_years(tmp_path, occi_dir, capsys):
    run_main(args_for(occi_dir, tmp_path / "out", tmp_path / "model",
                      extra=["--list-days"], years="2022"))
    out = capsys.readouterr().out
    assert "2022: 1 days (2022-03-04 .. 2022-03-04)" in out


# ---------------------------------------------------------------------------
# the yearly output file
# ---------------------------------------------------------------------------
def test_output_file_has_the_documented_structure(run, model_dir):
    _, path = run
    with nc.Dataset(path) as ds:
        assert set(ds.variables) == {"time", "lat", "lon", "chl_mean", "unc_epistemic",
                                     "unc_aleatoric", "n_valid"}
        assert ds.dimensions["time"].size == 2
        assert ds.dimensions["lat"].size == 8 and ds.dimensions["lon"].size == 8
        assert ds.spatial_resolution == "0.25deg"
        assert int(ds.bin_factor) == 6 and int(ds.min_valid_pixels) == 1
        assert int(ds.mc_samples) == 4
        assert ds["time"].units == "days since 1970-01-01 00:00:00"
        # days since the epoch, in the order of the input files
        assert list(ds["time"][:]) == [(datetime(2021, 1, 1) - datetime(1970, 1, 1)).days,
                                       (datetime(2021, 1, 2) - datetime(1970, 1, 1)).days]
        assert ds.time_coverage_start == "2021-01-01"
        assert ds.time_coverage_end == "2021-01-02"


def test_output_grid_is_written_south_to_north_with_cell_centres(run):
    _, path = run
    with nc.Dataset(path) as ds:
        lat = ds["lat"][:]
        lon = ds["lon"][:]
    assert np.all(np.diff(lat) > 0)                    # south -> north after the flip
    assert np.all(np.diff(lon) > 0)
    assert lat[0] == pytest.approx(15.5) and lat[-1] == pytest.approx(57.5)
    assert lon[0] == pytest.approx(-27.5)


def test_retrieval_values_are_physical_and_uncertainties_positive(run):
    _, path = run
    with nc.Dataset(path) as ds:
        chl = ds["chl_mean"][:]
        epi = ds["unc_epistemic"][:]
        ale = ds["unc_aleatoric"][:]
        n_valid = ds["n_valid"][:]
    # every block of a 6 x 6 tile with 10 % missing pixels has valid data
    assert np.isfinite(chl).all()
    assert chl.min() > -3.0 and chl.max() < 3.0       # the documented valid range
    assert (epi >= 0).all() and (ale > 0).all()
    assert n_valid.min() >= 1 and n_valid.max() <= 36
    # the concentrations implied by these synthetic spectra are ocean-like
    assert 0.1 < 10 ** chl.mean() < 30


def test_missing_pixels_produce_the_fill_value(tmp_path, occi_dir, model_dir):
    """A block with no valid pixel must be fill, not zero and not NaN."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    from conftest import write_daily_file as writer
    rng = np.random.default_rng(1)
    lat = np.arange(60.0, 12.0 - 1e-9, -1.0)[:48]
    lon = np.arange(-30.0, 18.0 - 1e-9, 1.0)[:48]
    hollow = occi_dir / "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210103-fv6.0.nc"
    writer(hollow, lat, lon, rng, nan_fraction=0.0, fill_hole=(0, 0, 6))
    output_dir = tmp_path / "out"
    run_main(args_for(occi_dir, output_dir, model_dir))
    with nc.Dataset(output_file(output_dir)) as ds:
        chl = np.ma.getdata(ds["chl_mean"][:])
        n_valid = ds["n_valid"][:]
        epoch_day = (datetime(2021, 1, 3) - datetime(1970, 1, 1)).days
        k = list(ds["time"][:]).index(epoch_day)          # the record of the hollow file
        # the hole sits at the north-west corner of the input, and the output
        # runs south to north, so it lands in the last output row, first column
        assert n_valid[k, 7, 0] == 0
        assert np.isclose(chl[k, 7, 0], 9.96921e36)
        assert not np.isclose(chl[k, 7, 1], 9.96921e36)
        assert n_valid[k, 7, 1] > 0


# ---------------------------------------------------------------------------
# resume / overwrite / consistency
# ---------------------------------------------------------------------------
def test_rerunning_keeps_the_days_already_written(run, occi_dir, model_dir, tmp_path):
    output_dir, path = run
    with nc.Dataset(path) as ds:
        before = ds["chl_mean"][:].copy()
    # a different seed would change the values if the days were recomputed
    run_main(args_for(occi_dir, output_dir, model_dir, extra=["--mc-seed", "7"]))
    with nc.Dataset(path) as ds:
        assert ds.dimensions["time"].size == 2
        np.testing.assert_array_equal(ds["chl_mean"][:], before)


def test_overwrite_recomputes_the_days(run, occi_dir, model_dir):
    output_dir, path = run
    with nc.Dataset(path) as ds:
        before = ds["chl_mean"][:].copy()
    run_main(args_for(occi_dir, output_dir, model_dir,
                      extra=["--mc-seed", "7", "--overwrite"]))
    with nc.Dataset(path) as ds:
        assert ds.dimensions["time"].size == 2
        assert not np.array_equal(ds["chl_mean"][:], before)


def test_refuses_to_mix_days_processed_with_other_settings(run, occi_dir, model_dir):
    output_dir, _ = run
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, output_dir, model_dir, extra=["--min-valid", "5"]))
    assert "was written with different settings" in str(exc.value)
    assert "min_valid_pixels=1 on disk, 5 requested" in str(exc.value)


def test_refuses_a_file_with_a_different_grid(run, occi_dir, model_dir):
    output_dir, path = run
    coarse = output_file(output_dir, res="0.5deg")
    with nc.Dataset(coarse, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 8)                  # bin 12 would expect 4 x 4 cells
        ds.createDimension("lon", 8)
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, output_dir, model_dir, extra=["--bin", "12"]))
    assert "holds a 8 x 8 grid" in str(exc.value)


def test_refuses_a_file_that_is_not_a_yearly_output(run, occi_dir, model_dir):
    output_dir, _ = run
    stray = output_file(output_dir, res="0.5deg")
    with nc.Dataset(stray, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 4)
        ds.createDimension("lon", 4)
        ds.createVariable("time", "i4", ("time",))
        ds.createVariable("lat", "f4", ("lat",))
        ds.createVariable("lon", "f4", ("lon",))
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, output_dir, model_dir, extra=["--bin", "12"]))
    assert "does not look like a yearly file" in str(exc.value)


def test_continues_an_existing_file_with_more_days(tmp_path, occi_dir, model_dir):
    """Days added later must be appended, not rewrite the file."""
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    from conftest import write_daily_file as writer
    rng = np.random.default_rng(2)
    lat = np.arange(60.0, 12.0 - 1e-9, -1.0)[:48]
    lon = np.arange(-30.0, 18.0 - 1e-9, 1.0)[:48]
    writer(occi_dir / "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20211224-fv6.0.nc",
           lat, lon, rng)
    output_dir = tmp_path / "out"
    run_main(args_for(occi_dir, output_dir, model_dir,
                      extra=["--end-date", "2021-01-01"]))
    with nc.Dataset(output_file(output_dir)) as ds:
        assert ds.dimensions["time"].size == 1
    run_main(args_for(occi_dir, output_dir, model_dir))
    with nc.Dataset(output_file(output_dir)) as ds:
        assert ds.dimensions["time"].size == 3
        assert ds.time_coverage_end == "2021-12-24"


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def test_date_window_selects_a_subset(tmp_path, occi_dir, model_dir):
    output_dir = tmp_path / "out"
    run_main(args_for(occi_dir, output_dir, model_dir,
                      extra=["--start-date", "2021-01-02", "--end-date", "2021-01-02"]))
    with nc.Dataset(output_file(output_dir)) as ds:
        assert list(ds["time"][:]) == [(datetime(2021, 1, 2) - datetime(1970, 1, 1)).days]


def test_dates_alone_define_the_years(tmp_path, occi_dir, model_dir):
    output_dir = tmp_path / "out"
    run_main(["--input-dir", str(occi_dir), "--output-dir", str(output_dir),
              "--model-dir", str(model_dir), "--mc-samples", "4",
              "--start-date", "2022-03-01", "--end-date", "2022-03-31"])
    assert output_file(output_dir, year=2022).is_file()
    assert not output_file(output_dir, year=2021).exists()


def test_output_files_inside_the_input_directory_are_skipped(tmp_path, occi_dir, model_dir, capsys):
    output_dir = occi_dir / "yearly"                   # inside the input tree
    run_main(args_for(occi_dir, output_dir, model_dir))
    assert output_file(output_dir).is_file()
    capsys.readouterr()
    run_main(args_for(occi_dir, output_dir, model_dir))        # must not see its own file
    with nc.Dataset(output_file(output_dir)) as ds:
        assert ds.dimensions["time"].size == 2


def test_duplicate_days_are_an_error(tmp_path, occi_dir):
    import shutil
    shutil.copy(occi_dir / "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210101-fv6.0.nc",
                occi_dir / "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210101-fv6.0-copy.nc")
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(occi_dir), "--years", "2021"])
    assert "two input files" in str(exc.value)


def test_missing_bands_are_reported(tmp_path, model_dir):
    with nc.Dataset(tmp_path / "broken-20210101-fv6.0.nc", "w", format="NETCDF4") as ds:
        ds.createDimension("lat", 6)
        ds.createDimension("lon", 6)
        ds.createVariable("lat", "f4", ("lat",))
        ds.createVariable("lon", "f4", ("lon",))
        ds.createVariable("Rrs_443", "f4", ("lat", "lon"))
    with pytest.raises(SystemExit) as exc:
        run_main(["--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                  "--model-dir", str(model_dir), "--years", "2021"])
    assert "does not contain" in str(exc.value)
    assert "Rrs_490" in str(exc.value)


def test_grid_changes_between_days_are_reported(tmp_path, occi_dir, model_dir):
    rng = np.random.default_rng(5)
    write_daily_file(occi_dir / "ESACCI-OC-L3S-RRS-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210104-fv6.0.nc",
                     np.arange(30.0, 6.0 - 1e-9, -1.0)[:24], np.arange(0.0, 24.0), rng)
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, tmp_path / "out", model_dir))
    assert "differs from the" in str(exc.value)


def test_block_size_must_divide_the_grid(tmp_path, occi_dir, model_dir):
    with pytest.raises(SystemExit) as exc:
        run_main(args_for(occi_dir, tmp_path / "out", model_dir, extra=["--bin", "7"]))
    assert "not divisible by the block size" in str(exc.value)
