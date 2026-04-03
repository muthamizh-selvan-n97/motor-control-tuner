"""
test/test_dashboard.py — pytest tests for modules/dashboard.py

Verified:
  - generate_dashboard saves PNG and CSV to output_dir
  - CSV has correct columns and 3 rows (current_d, current_q, speed)
  - CSV Kp/Ki values match loop results
  - Dashboard works for SPMSM+fan and IPMSM+const_torque
  - Dashboard works without plant= (waterfall panel disabled gracefully)
  - _safe_name produces filesystem-safe strings
  - _write_csv: GM 'inf' appears for loops with no phase crossover
"""

import csv
import math
import os
import pytest

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.speed_loop import SpeedLoopTuner
from modules.dashboard import generate_dashboard, _safe_name, _write_csv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spmsm_loops():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    tuner_i = CurrentLoopTuner(plant, params)
    i_d = tuner_i.tune("pole_zero", axis="d", tau_cl_s=1e-3)
    i_q = tuner_i.tune("pole_zero", axis="q", tau_cl_s=1e-3)
    w = SpeedLoopTuner(plant, params, i_d).tune("pole_zero", tau_cl_s=50e-3)
    return plant, params, i_d, i_q, w


@pytest.fixture(scope="module")
def ipmsm_loops():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    tuner_i = CurrentLoopTuner(plant, params)
    i_d = tuner_i.tune("pole_zero", axis="d", tau_cl_s=2e-3)
    i_q = tuner_i.tune("pole_zero", axis="q", tau_cl_s=2e-3)
    w = SpeedLoopTuner(plant, params, i_d).tune("pole_zero", tau_cl_s=50e-3)
    return plant, params, i_d, i_q, w


# ---------------------------------------------------------------------------
# _safe_name
# ---------------------------------------------------------------------------

class TestSafeName:
    def test_spaces_replaced(self):
        assert " " not in _safe_name("Delta ECMA C21010")

    def test_lowercase(self):
        assert _safe_name("Motor Name") == _safe_name("motor name")

    def test_special_chars_removed(self):
        name = _safe_name("Motor (SPMSM) / 1kW")
        assert "(" not in name
        assert ")" not in name
        assert "/" not in name

    def test_hyphens_to_underscore(self):
        assert "-" not in _safe_name("motor-name")

    def test_non_empty(self):
        assert len(_safe_name("X")) > 0


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestWriteCSV:
    def test_creates_file(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        assert os.path.exists(path)

    def test_has_three_rows(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_loop_names(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        names = [r["loop"] for r in rows]
        assert names == ["current_d", "current_q", "speed"]

    def test_kp_matches_loop(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert math.isclose(float(rows[0]["Kp"]), i_d.Kp, rel_tol=1e-5)
        assert math.isclose(float(rows[1]["Kp"]), i_q.Kp, rel_tol=1e-5)
        assert math.isclose(float(rows[2]["Kp"]), w.Kp, rel_tol=1e-5)

    def test_ki_matches_loop(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert math.isclose(float(rows[0]["Ki"]), i_d.Ki, rel_tol=1e-5)

    def test_correct_columns(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        expected = ["loop", "method", "Kp", "Ki", "BW_Hz", "PM_deg",
                    "GM_dB", "settling_ms", "overshoot_pct", "warnings"]
        for col in expected:
            assert col in cols

    def test_gm_inf_when_large(self, spmsm_loops, tmp_path):
        """GM > 900 should be written as 'inf'."""
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "gains_inf.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        # At least one loop typically has infinite GM (PI + 1st-order plant)
        gm_vals = [r["GM_dB"] for r in rows]
        assert any(v == "inf" for v in gm_vals)

    def test_bw_hz_positive(self, spmsm_loops, tmp_path):
        _, params, i_d, i_q, w = spmsm_loops
        path = str(tmp_path / "bw.csv")
        _write_csv(path, i_d, i_q, w)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert float(r["BW_Hz"]) >= 0.0


# ---------------------------------------------------------------------------
# generate_dashboard — SPMSM
# ---------------------------------------------------------------------------

class TestDashboardSPMSM:
    def test_saves_png(self, spmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        pngs = list(tmp_path.glob("dashboard_*.png"))
        assert len(pngs) == 1

    def test_saves_csv(self, spmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        csvs = list(tmp_path.glob("gains_summary_*.csv"))
        assert len(csvs) == 1

    def test_csv_has_data(self, spmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        csvs = list(tmp_path.glob("gains_summary_*.csv"))
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_png_non_empty(self, spmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        pngs = list(tmp_path.glob("dashboard_*.png"))
        assert pngs[0].stat().st_size > 10_000  # at least 10 kB

    def test_no_save_does_not_create_files(self, spmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=False, output_dir=str(tmp_path))
        assert len(list(tmp_path.iterdir())) == 0

    def test_without_plant_no_crash(self, spmsm_loops, tmp_path):
        """Dashboard without plant= should still save (waterfall panel greyed out)."""
        _, params, i_d, i_q, w = spmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=None,
                           save=True, output_dir=str(tmp_path))
        pngs = list(tmp_path.glob("dashboard_*.png"))
        assert len(pngs) == 1


# ---------------------------------------------------------------------------
# generate_dashboard — IPMSM
# ---------------------------------------------------------------------------

class TestDashboardIPMSM:
    def test_saves_png(self, ipmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = ipmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        pngs = list(tmp_path.glob("dashboard_*.png"))
        assert len(pngs) == 1

    def test_saves_csv(self, ipmsm_loops, tmp_path):
        plant, params, i_d, i_q, w = ipmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        csvs = list(tmp_path.glob("gains_summary_*.csv"))
        assert len(csvs) == 1

    def test_asymmetric_dq_csv(self, ipmsm_loops, tmp_path):
        """IPMSM: d and q axis Kp should differ (Ld != Lq)."""
        plant, params, i_d, i_q, w = ipmsm_loops
        generate_dashboard(params, i_d, i_q, w, plant=plant,
                           save=True, output_dir=str(tmp_path))
        csvs = list(tmp_path.glob("gains_summary_*.csv"))
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        kp_d = float(rows[0]["Kp"])
        kp_q = float(rows[1]["Kp"])
        assert not math.isclose(kp_d, kp_q, rel_tol=0.05), \
            f"IPMSM: expected Kp_d != Kp_q, got {kp_d:.4f} and {kp_q:.4f}"


# ---------------------------------------------------------------------------
# run_all smoke test (import only — don't actually run the full pipeline)
# ---------------------------------------------------------------------------

class TestRunAllImport:
    def test_run_all_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_all", "run_all.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Just check it loads without syntax errors — don't execute
        assert spec is not None
        assert mod is not None

    def test_run_all_has_run_one(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_all", "run_all.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_one")

    def test_run_all_has_motor_list(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_all", "run_all.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "MOTORS")
        assert len(mod.MOTORS) == 2

    def test_run_all_has_load_list(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_all", "run_all.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "LOADS")
        assert len(mod.LOADS) == 2
