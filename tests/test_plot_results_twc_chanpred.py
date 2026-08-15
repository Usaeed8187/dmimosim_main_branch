import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "results" / "plot_results_twc_chanpred.py"
_SPEC = importlib.util.spec_from_file_location(
    "plot_results_twc_chanpred", _MODULE_PATH
)
plotter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plotter
_SPEC.loader.exec_module(plotter)


def _write_result(path, readout_mode, throughput=1.0):
    metrics = {
        "per_link_predictor_phases": {
            "0:0": {"readout_objective": readout_mode}
        }
    }
    np.savez(
        path,
        throughput=np.asarray(throughput),
        uncoded_ber_list=np.asarray([0.1]),
        ldpc_ber_list=np.asarray([0.01]),
        predictor_complexity_raw_json=np.asarray(json.dumps(metrics)),
    )


class TwcChannelPredictionPlotTest(unittest.TestCase):
    def test_sync_throughput_sweeps_fix_the_other_error_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            prefix = (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_kalman_filter_pmi_quantization_True_workers_8"
            )
            _write_result(
                folder / f"{prefix}_sync_errors_False_phase_std_deg_0_"
                "timing_std_samples_0.npz",
                "unused",
                10.0,
            )
            _write_result(
                folder / f"{prefix}_sync_errors_True_phase_std_deg_0_"
                "timing_std_samples_0p2.npz",
                "unused",
                8.0,
            )
            _write_result(
                folder / f"{prefix}_sync_errors_True_phase_std_deg_3p6_"
                "timing_std_samples_0.npz",
                "unused",
                7.0,
            )
            scenario = next(
                scenario
                for scenario in plotter._default_scenarios(link_adapt=True)
                if scenario.prediction_method == "kalman_filter"
            )
            cfg = plotter.PlotConfig(
                base_dir=str(base), mobility="higher_mobility", drops=[1],
                rx_ues=[4], tx_ues=[8], modulation_orders=[4], code_rates=[0.5],
                ber_modulation_order=4, ber_code_rate=0.5,
                fixed_rx_for_tx_sweep=4, fixed_tx_for_rx_sweep=8,
                output_dir=tmp, link_adapt=True, scenarios=[scenario],
                channelmamba_seen_drops=[], channelmamba_all_drops=[],
            )

            timing = plotter.sync_throughput_series(
                plotter.ResultLoader, cfg, [scenario], [0.0, 0.2], "timing"
            )
            phase = plotter.sync_throughput_series(
                plotter.ResultLoader, cfg, [scenario], [0.0, 3.6], "phase"
            )

            np.testing.assert_allclose(timing[0][1], [10.0, 8.0])
            np.testing.assert_allclose(phase[0][1], [10.0, 7.0])

    def test_balanced_wesn_loader_finds_time_split_worker_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            result = folder / (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_configured_wesn_balanced_pmi_quantization_True_"
                "time_split_workers_8.npz"
            )
            _write_result(result, "unused")
            scenario = next(
                scenario
                for scenario in plotter._default_scenarios(link_adapt=True)
                if scenario.prediction_method == "configured_wesn_balanced"
            )
            cfg = plotter.PlotConfig(
                base_dir=str(base), mobility="higher_mobility", drops=[1],
                rx_ues=[4], tx_ues=[8], modulation_orders=[4], code_rates=[0.5],
                ber_modulation_order=4, ber_code_rate=0.5,
                fixed_rx_for_tx_sweep=4, fixed_tx_for_rx_sweep=8,
                output_dir=tmp, link_adapt=True, scenarios=[scenario],
                channelmamba_seen_drops=[], channelmamba_all_drops=[],
            )
            selected = plotter.ResultLoader(cfg)._find_file(
                1, 4, 8, 4, 0.5, scenario
            )
            self.assertEqual(Path(selected), result)

    def test_balanced_lite_loader_finds_time_split_worker_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            result = folder / (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_configured_wesn_balanced_lite_"
                "pmi_quantization_True_time_split_workers_8.npz"
            )
            _write_result(result, "unused")
            scenario = next(
                scenario
                for scenario in plotter._default_scenarios(link_adapt=True)
                if scenario.prediction_method
                == "configured_wesn_balanced_lite"
            )
            cfg = plotter.PlotConfig(
                base_dir=str(base), mobility="higher_mobility", drops=[1],
                rx_ues=[4], tx_ues=[8], modulation_orders=[4], code_rates=[0.5],
                ber_modulation_order=4, ber_code_rate=0.5,
                fixed_rx_for_tx_sweep=4, fixed_tx_for_rx_sweep=8,
                output_dir=tmp, link_adapt=True, scenarios=[scenario],
                channelmamba_seen_drops=[], channelmamba_all_drops=[],
            )
            selected = plotter.ResultLoader(cfg)._find_file(
                1, 4, 8, 4, 0.5, scenario
            )
            self.assertEqual(Path(selected), result)

    def test_wesn_lite_loader_rejects_stale_matched_ridge_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            prefix = (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_wesn_lite_pmi_quantization_True_time_split"
            )
            matched = folder / f"{prefix}.npz"
            centered = folder / f"{prefix}_workers_8.npz"
            _write_result(matched, "matched_ridge")
            _write_result(centered, "centered_ridge")

            scenario = plotter.Scenario(
                perfect_csi=False,
                prediction=True,
                quantization=True,
                label="Low-Rank Configured WESN",
                link_adapt=True,
                prediction_method="wesn_lite",
            )
            cfg = plotter.PlotConfig(
                base_dir=str(base),
                mobility="higher_mobility",
                drops=[1],
                rx_ues=[4],
                tx_ues=[8],
                modulation_orders=[4],
                code_rates=[0.5],
                ber_modulation_order=4,
                ber_code_rate=0.5,
                fixed_rx_for_tx_sweep=4,
                fixed_tx_for_rx_sweep=8,
                output_dir=tmp,
                link_adapt=True,
                scenarios=[scenario],
                channelmamba_seen_drops=[],
                channelmamba_all_drops=[],
                wesn_lite_readout_mode="centered_ridge",
            )
            selected = plotter.ResultLoader(cfg)._find_file(
                1, 4, 8, 4, 0.5, scenario
            )

            self.assertEqual(Path(selected), centered)

    def test_kalman_loaders_prefer_new_worker_suffixed_results(self):
        for method in ("kalman_filter", "steady_state_kalman_filter"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                folder = base / "channels_higher_mobility_1"
                folder.mkdir()
                prefix = (
                    "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                    f"prediction_{method}_pmi_quantization_True"
                )
                legacy = folder / f"{prefix}.npz"
                current = folder / f"{prefix}_workers_8.npz"
                _write_result(legacy, "unused")
                _write_result(current, "unused")
                os.utime(legacy, (1, 1))
                os.utime(current, (2, 2))

                scenario = next(
                    scenario
                    for scenario in plotter._default_scenarios(link_adapt=True)
                    if scenario.prediction_method == method
                )
                cfg = plotter.PlotConfig(
                    base_dir=str(base),
                    mobility="higher_mobility",
                    drops=[1],
                    rx_ues=[4],
                    tx_ues=[8],
                    modulation_orders=[4],
                    code_rates=[0.5],
                    ber_modulation_order=4,
                    ber_code_rate=0.5,
                    fixed_rx_for_tx_sweep=4,
                    fixed_tx_for_rx_sweep=8,
                    output_dir=tmp,
                    link_adapt=True,
                    scenarios=[scenario],
                    channelmamba_seen_drops=[],
                    channelmamba_all_drops=[],
                )

                selected = plotter.ResultLoader(cfg)._find_file(
                    1, 4, 8, 4, 0.5, scenario
                )
                self.assertEqual(Path(selected), current)


if __name__ == "__main__":
    unittest.main()
