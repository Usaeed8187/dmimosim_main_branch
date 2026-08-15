import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "results" / "plot_results_twc_chanpred_w_p1.py"
_SPEC = importlib.util.spec_from_file_location(
    "plot_results_twc_chanpred_w_p1", _MODULE_PATH
)
plotter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plotter
_SPEC.loader.exec_module(plotter)


def _write_result(path: Path, throughput: float) -> None:
    np.savez(
        path,
        throughput=np.asarray(throughput),
        uncoded_ber_list=np.asarray([0.1]),
        ldpc_ber_list=np.asarray([0.01]),
    )


class Phase1ComparisonPlotTest(unittest.TestCase):
    def _config(self, base: Path, scenarios):
        return plotter.phase2_plotter.PlotConfig(
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
            output_dir=str(base),
            link_adapt=True,
            scenarios=scenarios,
            channelmamba_seen_drops=[],
            channelmamba_all_drops=[],
        )

    def test_loader_pairs_phase2_only_and_phase1_enabled_names(self):
        scenarios = plotter._comparison_scenarios(["kalman_filter"], True)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            perfect = folder / (
                "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_kalman_filter_pmi_quantization_True.npz"
            )
            imperfect = folder / (
                "mu_mimo_results_p1_True_p3_False_link_adapt_rx_UE_4_tx_UE_8_"
                "prediction_kalman_filter_pmi_quantization_True.npz"
            )
            _write_result(perfect, 10.0)
            _write_result(imperfect, 8.0)
            loader = plotter.Phase1ResultLoader(self._config(base, scenarios))

            self.assertEqual(
                Path(loader._find_file(1, 4, 8, 0, 0, scenarios[0])), perfect
            )
            self.assertEqual(
                Path(loader._find_file(1, 4, 8, 0, 0, scenarios[1])), imperfect
            )
            self.assertEqual(scenarios[0].phase_1_label, "perfect phase 1")
            self.assertEqual(scenarios[1].phase_1_label, "imperfect phase 1")

    def test_phase1_sync_sweep_uses_shared_fixed_rx_tx_aggregation(self):
        scenarios = plotter._comparison_scenarios(["kalman_filter"], True)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            for scenario, throughput in zip(scenarios, (10.0, 8.0)):
                phase_prefix = (
                    "p1_True_p3_False_" if scenario.imperfect_phase_1 else ""
                )
                result = folder / (
                    f"mu_mimo_results_{phase_prefix}link_adapt_rx_UE_4_tx_UE_8_"
                    "prediction_kalman_filter_pmi_quantization_True_workers_8_"
                    "sync_errors_True_phase_std_deg_0_timing_std_samples_0p2.npz"
                )
                _write_result(result, throughput)
            cfg = self._config(base, scenarios)

            series = plotter.phase2_plotter.sync_throughput_series(
                plotter.Phase1ResultLoader,
                cfg,
                scenarios,
                [0.2],
                "timing",
            )

            self.assertEqual([scenario for scenario, _ in series], scenarios)
            np.testing.assert_allclose(
                [values[0] for _, values in series], [10.0, 8.0]
            )

    def test_phase1_sync_sweep_loads_zero_from_tx_sweep(self):
        scenarios = plotter._comparison_scenarios(["kalman_filter"], True)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            for scenario, throughput in zip(scenarios, (10.0, 8.0)):
                phase_prefix = (
                    "p1_True_p3_False_" if scenario.imperfect_phase_1 else ""
                )
                result = folder / (
                    f"mu_mimo_results_{phase_prefix}link_adapt_rx_UE_4_tx_UE_8_"
                    "prediction_kalman_filter_pmi_quantization_True_workers_8_"
                    "sync_errors_False_phase_std_deg_0_timing_std_samples_0.npz"
                )
                _write_result(result, throughput)

            series = plotter.phase2_plotter.sync_throughput_series(
                plotter.Phase1ResultLoader,
                self._config(base, scenarios),
                scenarios,
                [0.0],
                "phase",
            )

            np.testing.assert_allclose(
                [values[0] for _, values in series], [10.0, 8.0]
            )

    def test_each_method_has_both_phase1_curves(self):
        scenarios = plotter._comparison_scenarios(plotter.METHODS, True)
        for method in plotter.METHODS:
            matching = [s for s in scenarios if s.prediction_method == method]
            self.assertEqual(len(matching), 2)
            self.assertEqual(
                {scenario.phase_1_label for scenario in matching},
                {"perfect phase 1", "imperfect phase 1"},
            )

    def test_balanced_method_is_available_with_distinct_style(self):
        self.assertIn("configured_wesn_balanced", plotter.METHODS)
        scenarios = plotter._comparison_scenarios(
            ["configured_wesn_balanced"], True
        )
        self.assertEqual(len(scenarios), 2)
        self.assertTrue(
            all("Balanced Configured WESN" in scenario.curve_label for scenario in scenarios)
        )
        self.assertIn("configured_wesn_balanced_lite", plotter.METHODS)

    def test_kalman_loaders_prefer_new_worker_suffixed_results(self):
        for method in ("kalman_filter", "steady_state_kalman_filter"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                folder = base / "channels_higher_mobility_1"
                folder.mkdir()
                scenarios = plotter._comparison_scenarios([method], True)
                scenario = next(s for s in scenarios if s.imperfect_phase_1)
                prefix = (
                    "mu_mimo_results_p1_True_p3_False_link_adapt_rx_UE_4_tx_UE_8_"
                    f"prediction_{method}_pmi_quantization_True"
                )
                legacy = folder / f"{prefix}.npz"
                current = folder / f"{prefix}_workers_8.npz"
                _write_result(legacy, 10.0)
                _write_result(current, 20.0)
                os.utime(legacy, (1, 1))
                os.utime(current, (2, 2))

                loader = plotter.Phase1ResultLoader(
                    self._config(base, scenarios)
                )
                selected = loader._find_file(1, 4, 8, 0, 0, scenario)
                self.assertEqual(Path(selected), current)


if __name__ == "__main__":
    unittest.main()
