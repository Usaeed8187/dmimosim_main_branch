import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "results" / "plot_chan_pred_nmse_cdf.py"
_SPEC = importlib.util.spec_from_file_location(
    "plot_chan_pred_nmse_cdf", _MODULE_PATH
)
plotter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = plotter
_SPEC.loader.exec_module(plotter)


class ChannelPredictionNmseCdfTest(unittest.TestCase):
    def test_latest_worker_result_is_used_for_both_kalman_filters(self):
        prefix = "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8"

        for method in ("kalman_filter", "steady_state_kalman_filter"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                folder = base / "channels_higher_mobility_1"
                folder.mkdir()
                scenario = plotter.Scenario(
                    label=method,
                    prediction=True,
                    prediction_method=method,
                )
                stem = f"{prefix}_prediction_{method}_pmi_quantization_True"
                legacy = folder / f"{stem}.npz"
                current = folder / f"{stem}_workers_8.npz"
                np.savez(legacy, chan_pred_nmse=np.asarray([0.1]))
                np.savez(current, chan_pred_nmse=np.asarray([0.2]))
                os.utime(legacy, (1_000, 1_000))
                os.utime(current, (2_000, 2_000))

                collected = plotter._collect_nmse(
                    base_dir=base,
                    mobility="higher_mobility",
                    drops=[1],
                    prefixes=[prefix],
                    quantization=True,
                    scenarios=[scenario],
                    wesn_lite_readout_mode="centered_ridge",
                )

                np.testing.assert_array_equal(
                    collected[method], np.asarray([0.2])
                )

    def test_balanced_wesn_time_split_worker_result_is_loaded(self):
        scenario = plotter.Scenario(
            label="Balanced Configured WESN",
            prediction=True,
            prediction_method="configured_wesn_balanced",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            prefix = "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8"
            result = folder / (
                f"{prefix}_prediction_configured_wesn_balanced_"
                "pmi_quantization_True_time_split_workers_8.npz"
            )
            expected = np.asarray([0.2, 0.1, 0.05])
            np.savez(result, chan_pred_nmse=expected)

            collected = plotter._collect_nmse(
                base_dir=base,
                mobility="higher_mobility",
                drops=[1],
                prefixes=[prefix],
                quantization=True,
                scenarios=[scenario],
                wesn_lite_readout_mode="centered_ridge",
            )

            np.testing.assert_array_equal(
                collected["Balanced Configured WESN"], expected
            )

    def test_balanced_lite_time_split_worker_result_is_loaded(self):
        scenario = plotter.Scenario(
            label="Balanced Configured WESN-Lite",
            prediction=True,
            prediction_method="configured_wesn_balanced_lite",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "channels_higher_mobility_1"
            folder.mkdir()
            prefix = "mu_mimo_results_link_adapt_rx_UE_4_tx_UE_8"
            result = folder / (
                f"{prefix}_prediction_configured_wesn_balanced_lite_"
                "pmi_quantization_True_time_split_workers_8.npz"
            )
            expected = np.asarray([0.15, 0.08])
            np.savez(result, chan_pred_nmse=expected)
            collected = plotter._collect_nmse(
                base_dir=base,
                mobility="higher_mobility",
                drops=[1],
                prefixes=[prefix],
                quantization=True,
                scenarios=[scenario],
                wesn_lite_readout_mode="centered_ridge",
            )
            np.testing.assert_array_equal(
                collected["Balanced Configured WESN-Lite"], expected
            )


if __name__ == "__main__":
    unittest.main()
