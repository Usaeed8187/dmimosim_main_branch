import copy
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
import sys
import types

import numpy as np

# Load only the predictor modules. Importing the top-level dmimo package also
# imports the TensorFlow simulation stack, which this standalone test does not
# need.
_ROOT = Path(__file__).resolve().parents[1]
if "dmimo" not in sys.modules:
    _dmimo = types.ModuleType("dmimo")
    _dmimo.__path__ = [str(_ROOT / "dmimo")]
    sys.modules["dmimo"] = _dmimo
if "dmimo.channel" not in sys.modules:
    _channel = types.ModuleType("dmimo.channel")
    _channel.__path__ = [str(_ROOT / "dmimo" / "channel")]
    sys.modules["dmimo.channel"] = _channel

from dmimo.channel.steady_state_kalman_filter_pred import (
    steady_state_kalman_filter_pred,
)
from dmimo.channel.kalman_filter_pred import kalman_filter_pred
from dmimo.channel.configured_wesn_pred import (
    build_configured_predictors_simple,
    configured_wesn_pred,
    predict_all_links_with_configured,
    predict_all_links_with_configured_simple,
    square_root_balanced_truncate,
)
from dmimo.channel.complexity_instrumentation import measure_phase, phase_summary
from tests.test_ns3_channels_prediction import (
    LowRankConfiguredWeightsESN,
    centered_ridge_readout_train_predict_next,
    collect_low_rank_esn_states_per_tile,
)


def _history(values):
    """Create [T, batch, 1, Nr, 1, Nt, sym, sc] scalar CSI history."""
    values = np.asarray(values, dtype=np.complex128)
    return values.reshape(-1, 1, 1, 1, 1, 1, 1, 1)


def _configured_scalar_predictor(max_iters=5000, tol=1e-12):
    predictor = steady_state_kalman_filter_pred(
        ar_order=1,
        max_riccati_iters=max_iters,
        riccati_tol=tol,
    )
    predictor.p = 1
    predictor.d = 1
    predictor.f_aug = np.array([[0.8]], dtype=np.complex128)
    predictor.q_aug = np.array([[0.1]], dtype=np.complex128)
    predictor.r_diag = np.array([0.2])
    predictor.h_mat, predictor.k_gain = predictor._solve_steady_state_gain(
        predictor.f_aug, predictor.q_aug, predictor.r_diag
    )
    predictor.is_configured = True
    predictor.reset_state()
    return predictor


class SteadyStateKalmanPredictorTest(unittest.TestCase):
    def test_full_kalman_batched_tiles_match_scalar_updates(self):
        rng = np.random.default_rng(19)
        predictor = kalman_filter_pred(ar_order=2)
        num_history = 6
        num_tiles = 9
        d = 3
        p = 2

        history = (
            rng.normal(size=(num_history, num_tiles, d))
            + 1j * rng.normal(size=(num_history, num_tiles, d))
        )
        r_diag = 0.02 + rng.random((num_tiles, d))
        a_blocks = [
            0.15 * np.eye(d, dtype=np.complex128),
            -0.05 * np.eye(d, dtype=np.complex128),
        ]
        q_seed = (
            rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        )
        q_proc = 0.01 * (q_seed @ q_seed.conj().T)
        f_aug, q_aug = predictor._build_augmented_system(a_blocks, q_proc)

        batched_z, batched_p = predictor._initialize_recursive_prior_batched(
            history, r_diag, f_aug, q_aug
        )
        scalar_priors = [
            predictor._initialize_recursive_prior(
                history[:, tile_idx], r_diag[tile_idx], f_aug, q_aug
            )
            for tile_idx in range(num_tiles)
        ]
        scalar_z = np.stack([prior[0] for prior in scalar_priors])
        scalar_p = np.stack([prior[1] for prior in scalar_priors])
        np.testing.assert_allclose(batched_z, scalar_z, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(batched_p, scalar_p, rtol=1e-12, atol=1e-12)

        new_observation = (
            rng.normal(size=(num_tiles, d))
            + 1j * rng.normal(size=(num_tiles, d))
        )
        batched_next_z, batched_next_p = (
            predictor._update_and_predict_prior_batched(
                batched_z,
                batched_p,
                new_observation,
                r_diag,
                f_aug,
                q_aug,
            )
        )
        scalar_next = [
            predictor._update_and_predict_prior(
                scalar_z[tile_idx],
                scalar_p[tile_idx],
                new_observation[tile_idx],
                r_diag[tile_idx],
                f_aug,
                q_aug,
            )
            for tile_idx in range(num_tiles)
        ]
        np.testing.assert_allclose(
            batched_next_z,
            np.stack([prior[0] for prior in scalar_next]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            batched_next_p,
            np.stack([prior[1] for prior in scalar_next]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_full_kalman_prediction_is_independent_of_tile_batch_size(self):
        rng = np.random.default_rng(23)
        history = (
            rng.normal(size=(5, 1, 1, 1, 1, 1, 2, 7))
            + 1j * rng.normal(size=(5, 1, 1, 1, 1, 1, 2, 7))
        )
        error_variance = 0.01 + rng.random(history.shape)

        single_tile = kalman_filter_pred(ar_order=2, tile_batch_size=1)
        batched_tiles = kalman_filter_pred(ar_order=2, tile_batch_size=5)
        for predictor in (single_tile, batched_tiles):
            predictor.num_bs_ant = 1
            predictor.num_ue_ant = 1

        expected = single_tile.predict(history, error_variance)
        actual = batched_tiles.predict(history, error_variance)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_balanced_lite_retains_smallest_order_above_hankel_energy_target(self):
        a = np.diag([0.8, 0.4, 0.1]).astype(np.complex128)
        b = np.diag([1.0, 0.2, 0.02]).astype(np.complex128)
        c = np.eye(3, dtype=np.complex128)
        d = np.zeros((3, 3), dtype=np.complex128)
        a_red, _, _, _, hsv, _ = square_root_balanced_truncate(
            a, b, c, d, energy_threshold=0.90
        )
        energy = hsv**2
        cumulative = np.cumsum(energy) / np.sum(energy)
        expected_order = int(np.searchsorted(cumulative, 0.90)) + 1

        self.assertEqual(a_red.shape[0], expected_order)
        self.assertGreaterEqual(cumulative[expected_order - 1], 0.90)
        if expected_order > 1:
            self.assertLess(cumulative[expected_order - 2], 0.90)

    def test_full_kalman_persists_state_and_freezes_drop_model(self):
        predictor = kalman_filter_pred(ar_order=2)
        predictor.num_bs_ant = 1
        predictor.num_ue_ant = 1

        values = np.arange(1, 15, dtype=np.float64).astype(np.complex128)
        history = _history(values)
        error_variance = np.full(history.shape, 0.05, dtype=np.float64)

        estimate_calls = 0
        original_estimate = predictor._estimate_ar_p_q_joint

        def counted_estimate(*args, **kwargs):
            nonlocal estimate_calls
            estimate_calls += 1
            return original_estimate(*args, **kwargs)

        predictor._estimate_ar_p_q_joint = counted_estimate

        first_prediction = predictor.predict(
            history[0:4], error_variance[0:4]
        )
        self.assertEqual(predictor.num_filter_updates_last_predict, 2)
        self.assertEqual(predictor.num_model_reconfigurations, 1)
        self.assertEqual(estimate_calls, 1)
        model = predictor._model_cache[(0, 0, 0)]
        expected_first = predictor._kalman_predict_one_step_ar_p(
            history[0:4].reshape(4, 1),
            model["r_diag_tiles"][0],
            model["f_aug"],
            model["q_aug"],
        )
        np.testing.assert_allclose(
            first_prediction.reshape(-1), expected_first.reshape(-1)
        )

        for start in range(1, 11):
            predictor.predict(
                history[start : start + 4],
                error_variance[start : start + 4],
            )
            self.assertEqual(predictor.num_filter_updates_last_predict, 1)
            self.assertEqual(estimate_calls, 1)

        self.assertEqual(predictor.num_model_reconfigurations, 1)

    def test_full_kalman_rebuilds_state_for_discontinuous_history(self):
        predictor = kalman_filter_pred(ar_order=2)
        predictor.num_bs_ant = 1
        predictor.num_ue_ant = 1

        history = _history(np.arange(1, 7, dtype=np.float64))
        error_variance = np.full(history.shape, 0.05, dtype=np.float64)

        predictor.predict(history[0:4], error_variance[0:4])
        predictor.predict(history[1:5], error_variance[1:5])
        self.assertEqual(predictor.num_filter_updates_last_predict, 1)

        predictor.predict(history[0:4], error_variance[0:4])
        self.assertEqual(predictor.num_filter_updates_last_predict, 2)

    def test_adaptive_full_kalman_refits_f_q_r_every_prediction(self):
        predictor = kalman_filter_pred(
            ar_order=2,
            reconfiguration_interval=1,
        )
        predictor.num_bs_ant = 1
        predictor.num_ue_ant = 1

        history = _history([1.0, 2.0, 4.0, 3.0, 7.0])
        first_error_variance = np.full(history[0:4].shape, 0.05)
        second_error_variance = np.full(history[1:5].shape, 0.20)

        estimate_calls = 0
        original_estimate = predictor._estimate_ar_p_q_joint

        def counted_estimate(*args, **kwargs):
            nonlocal estimate_calls
            estimate_calls += 1
            return original_estimate(*args, **kwargs)

        predictor._estimate_ar_p_q_joint = counted_estimate
        predictor.predict(history[0:4], first_error_variance)
        first_model = predictor._model_cache[(0, 0, 0)]
        predictor.predict(history[1:5], second_error_variance)
        second_model = predictor._model_cache[(0, 0, 0)]

        self.assertEqual(estimate_calls, 2)
        self.assertEqual(predictor.num_model_reconfigurations, 2)
        self.assertEqual(predictor.num_filter_updates_last_predict, 2)
        self.assertIsNot(first_model, second_model)
        np.testing.assert_allclose(second_model["r_diag_tiles"], 0.20)
        self.assertFalse(
            np.allclose(first_model["f_aug"], second_model["f_aug"])
        )

    def test_configured_wesn_process_workers_match_simple_and_persist_updates(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.8,
            input_scale=0.2,
            window_length=2,
            regularization=1e-3,
            enable_window=True,
            enable_kalman_weight_config=False,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="identity",
            esn_ls_reg=1e-3,
            enable_skip_connections=True,
        )
        ns3cfg = types.SimpleNamespace(num_txue_sel=1, num_rxue_sel=1)
        values = np.arange(1, 1 + 4 * 2 * 2, dtype=np.float32)
        history = values.reshape(4, 1, 1, 2, 1, 2, 1, 1).astype(np.complex64)
        predictors = build_configured_predictors_simple(
            history,
            rc_config,
            ns3cfg,
            num_bs_ant=1,
            num_ue_ant=1,
        )
        parallel_predictors = copy.deepcopy(predictors)

        expected = predict_all_links_with_configured_simple(
            history,
            predictors,
            ns3cfg,
            num_bs_ant=1,
            num_ue_ant=1,
        )
        actual = predict_all_links_with_configured(
            history,
            parallel_predictors,
            ns3cfg,
            num_bs_ant=1,
            num_ue_ant=1,
            max_workers=2,
        )
        persistent_predictors = copy.deepcopy(predictors)
        with ThreadPoolExecutor(max_workers=2) as executor:
            persistent_actual = predict_all_links_with_configured(
                history,
                persistent_predictors,
                ns3cfg,
                num_bs_ant=1,
                num_ue_ant=1,
                max_workers=2,
                executor=executor,
            )

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(persistent_actual, expected)
        for link, predictor in predictors.items():
            np.testing.assert_allclose(
                parallel_predictors[link].W_out,
                predictor.W_out,
            )
            self.assertIn(
                "online_update_batch_ridge",
                parallel_predictors[link].predictor_complexity_metrics["phases"],
            )

    def test_standalone_wesn_lite_realizes_truncated_residue(self):
        poles = np.array([[0.5]], dtype=np.complex128)
        residue = np.diag([4.0, 1.0]).astype(np.complex128)
        residues = residue.T.reshape(1, 4, 1)
        predictor = LowRankConfiguredWeightsESN(
            poles=poles,
            residues=residues,
            d_out=2,
            d_in=2,
            energy_threshold=0.90,
            activation="identity",
        )

        self.assertEqual(predictor.state_dim, 1)
        np.testing.assert_allclose(predictor.W_in, [[4.0, 0.0]])
        np.testing.assert_allclose(predictor.W_out_reference, [[1.0], [0.0]])

        inputs = np.array(
            [[[2.0, 3.0]], [[1.0, -2.0]]], dtype=np.complex128
        )
        states = collect_low_rank_esn_states_per_tile(predictor, inputs)
        np.testing.assert_allclose(states[:, 0, 0], [8.0, 8.0])

    def test_standalone_wesn_lite_centered_readout(self):
        features = np.array(
            [[[1.0]], [[2.0]], [[-1.0]]], dtype=np.complex128
        )
        targets = np.array(
            [[[1.0]], [[3.0]], [[-2.0]]], dtype=np.complex128
        )
        reference = np.array([[2.0]], dtype=np.complex128)
        prediction = centered_ridge_readout_train_predict_next(
            features, targets, reference, reg=0.5
        )

        z_train = features[:-1].reshape(-1, 1)
        y_train = targets[1:].reshape(-1, 1)
        normal_reg = z_train.shape[0] * 0.5
        fitted = np.linalg.solve(
            z_train.conj().T @ z_train + normal_reg * np.eye(1),
            z_train.conj().T @ y_train + normal_reg * reference.T,
        )
        np.testing.assert_allclose(prediction, features[-1] @ fitted)

    def test_residue_rank_uses_95_percent_singular_value_energy(self):
        rank, retained = configured_wesn_pred._rank_for_energy([5.0, 1.0], 0.95)
        self.assertEqual(rank, 1)
        self.assertGreaterEqual(retained, 0.95)

        rank, retained = configured_wesn_pred._rank_for_energy([3.0, 1.0], 0.95)
        self.assertEqual(rank, 2)
        self.assertEqual(retained, 1.0)

    def test_low_rank_residue_mapping_sets_adaptive_reservoir_size(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex128
        predictor.input_dim = 4
        predictor.output_dim = 2
        predictor.enable_skip_connections = True
        predictor.residue_energy_threshold = 0.95
        predictor.W_out_reference = None
        predictor.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}

        poles = np.array([[0.8, 0.6]], dtype=np.complex128)
        residue_matrices = [
            np.diag([5.0, 1.0]),
            np.diag([3.0, 1.0]),
        ]
        residues = np.zeros((1, 4, 2), dtype=np.complex128)
        for pole_idx, matrix in enumerate(residue_matrices):
            residues[0, :, pole_idx] = matrix.T.reshape(-1)

        predictor._configure_low_rank_reservoir(poles, residues, d=2)

        self.assertEqual(predictor.residue_ranks, [1, 2])
        self.assertEqual(predictor.state_dim, 3)
        self.assertEqual(predictor.feature_dim, 7)
        np.testing.assert_allclose(predictor.W_res, [0.8, 0.6, 0.6])
        np.testing.assert_allclose(predictor.W_in[:, :2], 0.0)
        np.testing.assert_allclose(
            predictor.W_in[:, -2:],
            [[5.0, 0.0], [3.0, 0.0], [0.0, 1.0]],
        )
        np.testing.assert_allclose(
            predictor.W_out_reference[:, : predictor.state_dim],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(
            predictor.W_out_reference[:, predictor.state_dim :], 0.0
        )
        np.testing.assert_allclose(predictor.W_out, predictor.W_out_reference)
        self.assertEqual(
            predictor.predictor_complexity_metrics["residue_rank_histogram"],
            {"1": 1, "2": 1},
        )

    def test_wesn_lite_uses_full_batch_ridge_readout_update(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.8,
            input_scale=0.2,
            window_length=2,
            regularization=1e-3,
            enable_window=True,
            enable_kalman_weight_config=False,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="identity",
            esn_ls_reg=1e-3,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=True,
            residue_energy_threshold=0.95,
            reservoir_readout_regularization=1e-2,
            skip_readout_regularization=1e-4,
        )
        predictor = configured_wesn_pred(rc_config, 2, 1, 1)
        values = np.arange(1, 9, dtype=np.float32).reshape(4, 1, 1, 1, 1, 1, 1, 2)
        history = values.astype(np.complex64)

        predictor.fit_offline(history)
        initial_readout = predictor.W_out.copy()
        prediction = predictor.predict_online(history)

        self.assertEqual(prediction.shape, history[0].shape)
        self.assertFalse(np.array_equal(initial_readout, predictor.W_out))
        self.assertEqual(
            len(predictor.predictor_complexity_metrics["phases"]["online_update_batch_ridge"]),
            1,
        )

    def test_wesn_lite_averages_to_resource_blocks_and_expands(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.wesn_lite_subcarriers_per_rb = 12

        # Two OFDM symbols and 25 subcarriers produce three RBs. The final RB
        # contains only subcarrier 24, as it should not be padded before averaging.
        grid = np.stack(
            [np.arange(25), np.arange(25) + 100], axis=0
        ).reshape(1, 1, 1, 1, 1, 1, 2, 25).astype(np.complex64)
        reduced = predictor._average_resource_grid_for_wesn_lite(grid)

        self.assertEqual(reduced.shape[-2:], (1, 3))
        np.testing.assert_allclose(
            reduced.reshape(-1),
            [55.5, 67.5, 74.0],
        )

        expanded = predictor._expand_wesn_lite_resource_grid(
            reduced,
            num_ofdm_symbols=2,
            num_subcarriers=25,
        )
        self.assertEqual(expanded.shape[-2:], (2, 25))
        np.testing.assert_allclose(expanded[..., :12], 55.5)
        np.testing.assert_allclose(expanded[..., 12:24], 67.5)
        np.testing.assert_allclose(expanded[..., 24:], 74.0)

    def test_wesn_lite_trains_on_rb_grid_and_returns_subcarrier_grid(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.8,
            input_scale=0.2,
            window_length=2,
            regularization=1e-3,
            enable_window=True,
            enable_kalman_weight_config=False,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="identity",
            esn_ls_reg=1e-3,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=True,
            residue_energy_threshold=0.95,
            reservoir_readout_regularization=1e-2,
            skip_readout_regularization=1e-4,
            wesn_lite_subcarriers_per_rb=12,
        )
        predictor = configured_wesn_pred(rc_config, 25, 1, 1)
        history = np.arange(
            4 * 14 * 25, dtype=np.float32
        ).reshape(4, 1, 1, 1, 1, 1, 14, 25).astype(np.complex64)
        predictor.fit_offline(history)

        captured = {}
        original_fit_readout = predictor._fit_readout

        def capture_fit_readout(states, inputs, targets):
            captured["states_shape"] = states.shape
            return original_fit_readout(states, inputs, targets)

        predictor._fit_readout = capture_fit_readout
        prediction = predictor.predict_online(history)

        # 25 subcarriers form RBs of widths 12, 12, and 1. Three temporal
        # transitions therefore give 3*3=9 readout-training pairs.
        self.assertEqual(captured["states_shape"][:2], (3, 3))
        self.assertEqual(prediction.shape, history[0].shape)
        self.assertEqual(
            predictor.predictor_complexity_metrics["model_num_resource_blocks"],
            3,
        )
        np.testing.assert_allclose(prediction[..., 0, :], prediction[..., 1, :])
        np.testing.assert_allclose(
            prediction[..., :12],
            np.repeat(prediction[..., :1], 12, axis=-1),
        )
        np.testing.assert_allclose(
            prediction[..., 12:24],
            np.repeat(prediction[..., 12:13], 12, axis=-1),
        )

    def test_wesn_lite_persists_state_across_overlapping_histories(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.2,
            input_scale=0.2,
            window_length=2,
            regularization=1e-3,
            enable_window=True,
            enable_kalman_weight_config=False,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="tanh",
            esn_ls_reg=1e-3,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=True,
            residue_energy_threshold=0.95,
            reservoir_readout_regularization=1e-2,
            skip_readout_regularization=1e-4,
            wesn_lite_subcarriers_per_rb=12,
        )
        predictor = configured_wesn_pred(rc_config, 25, 1, 1)
        samples = np.arange(
            5 * 2 * 25, dtype=np.float32
        ).reshape(5, 1, 1, 1, 1, 1, 2, 25).astype(np.complex64) * 1e-3
        predictor.fit_offline(samples[:4])

        self.assertEqual(
            predictor.predictor_complexity_metrics["model_num_ofdm_symbols"],
            1,
        )
        self.assertEqual(
            predictor.predictor_complexity_metrics["model_num_resource_blocks"],
            3,
        )

        predictor.predict_online(samples[:4])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 4)
        first_cached_state = predictor._online_state.copy()

        predictor.predict_online(samples[1:5])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 1)
        self.assertFalse(np.array_equal(first_cached_state, predictor._online_state))
        self.assertEqual(predictor._online_states.shape[0], 4)
        self.assertEqual(predictor._online_inputs.shape[0], 4)
        self.assertEqual(
            predictor.predictor_complexity_metrics["online_feature_cache_length"],
            4,
        )

        # Rewinding to a non-overlapping history resets and replays the window.
        predictor.predict_online(samples[:4])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 4)

    def test_full_configured_wesn_persists_state_across_overlapping_histories(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.2,
            input_scale=0.2,
            window_length=2,
            regularization=1e-3,
            enable_window=True,
            enable_kalman_weight_config=False,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="tanh",
            esn_ls_reg=1e-3,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=False,
        )
        predictor = configured_wesn_pred(rc_config, 3, 1, 1)
        samples = np.arange(
            5 * 2 * 3, dtype=np.float32
        ).reshape(5, 1, 1, 1, 1, 1, 2, 3).astype(np.complex64) * 1e-3
        predictor.fit_offline(samples[:4])

        self.assertEqual(
            predictor.predictor_complexity_metrics["model_num_resource_blocks"],
            1,
        )

        predictor.predict_online(samples[:4])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 4)
        first_cached_state = predictor._online_state.copy()

        predictor.predict_online(samples[1:5])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 1)
        self.assertFalse(np.array_equal(first_cached_state, predictor._online_state))
        self.assertEqual(predictor._online_states.shape[0], 4)
        self.assertEqual(predictor._online_inputs.shape[0], 4)
        self.assertTrue(
            predictor.predictor_complexity_metrics["persistent_online_state"]
        )

        # A discontinuity still performs a safe zero-state replay.
        predictor.predict_online(samples[:4])
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 4)

    def test_balanced_configured_wesn_fits_and_advances_online(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=1.0,
            input_scale=1.0,
            window_length=2,
            regularization=1e-4,
            enable_window=True,
            enable_kalman_weight_config=True,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=16,
            esn_activation="identity",
            esn_ls_reg=1e-4,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=False,
            enable_balanced_truncation=True,
        )
        rng = np.random.default_rng(19)
        samples = np.zeros((9, 1, 1, 1, 1, 1, 2, 8), dtype=np.complex64)
        samples[0] = rng.normal(size=samples[0].shape) + 1j * rng.normal(
            size=samples[0].shape
        )
        for time_idx in range(1, samples.shape[0]):
            innovation = rng.normal(size=samples[time_idx].shape) + 1j * rng.normal(
                size=samples[time_idx].shape
            )
            samples[time_idx] = 0.75 * samples[time_idx - 1] + 0.1 * innovation
        error_variance = np.full(samples.shape, 0.02, dtype=np.float32)

        predictor = configured_wesn_pred(rc_config, 8, 1, 1)
        predictor.fit_offline(samples[:8], error_variance[:8])
        first = predictor.predict_online(samples[:8])
        second = predictor.predict_online(samples[1:9])

        self.assertEqual(
            predictor.predictor_complexity_metrics["method"],
            "configured_wesn_balanced",
        )
        self.assertEqual(predictor.state_dim, 1)
        self.assertEqual(
            predictor.predictor_complexity_metrics["balanced_order_per_mode"],
            1,
        )
        self.assertEqual(
            predictor.predictor_complexity_metrics["model_num_resource_blocks"],
            1,
        )
        self.assertEqual(predictor.num_reservoir_updates_last_predict, 1)
        self.assertEqual(first.shape, samples[0].shape)
        self.assertEqual(second.shape, samples[0].shape)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(np.isfinite(second)))

    def test_balanced_configured_wesn_preserves_activation_and_input_scale(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=0.2,
            input_scale=0.15,
            window_length=2,
            regularization=1e-4,
            enable_window=True,
            enable_kalman_weight_config=True,
            esn_m=1,
            esn_k=1,
            esn_num_freqs=8,
            esn_activation="tanh",
            esn_ls_reg=1e-4,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=False,
            enable_balanced_truncation=True,
        )
        predictor = configured_wesn_pred(rc_config, 1, 1, 1)
        predictor.W_res = np.asarray([0.0], dtype=np.complex64)
        predictor.W_in = np.asarray([[1.0]], dtype=np.complex64)
        predictor.state_dim = 1
        state = predictor._state_step(
            np.zeros((1, 1), dtype=np.complex64),
            np.ones((1, 1), dtype=np.complex64),
        )

        self.assertEqual(predictor.esn_activation, "tanh")
        self.assertEqual(predictor.input_scale, 0.15)
        np.testing.assert_allclose(state, np.tanh(0.15), rtol=1e-6)

    def test_balanced_lite_uses_hankel_threshold_and_rb_grid(self):
        rc_config = types.SimpleNamespace(
            W_tran_sparsity=0.5,
            W_tran_radius=1.0,
            input_scale=0.15,
            window_length=2,
            regularization=1e-4,
            enable_window=True,
            enable_kalman_weight_config=True,
            esn_m=1,
            esn_k=2,
            esn_num_freqs=16,
            esn_activation="tanh",
            esn_ls_reg=1e-4,
            esn_diagnostics=False,
            enable_skip_connections=True,
            wesn_online_update="batch_ridge",
            enable_residue_low_rank=False,
            enable_balanced_truncation=True,
            enable_balanced_hankel_truncation=True,
            balanced_hankel_energy_threshold=0.90,
            wesn_lite_subcarriers_per_rb=12,
        )
        rng = np.random.default_rng(23)
        samples = np.zeros((9, 1, 1, 1, 1, 1, 2, 25), dtype=np.complex64)
        samples[0] = rng.normal(size=samples[0].shape) + 1j * rng.normal(
            size=samples[0].shape
        )
        for time_idx in range(1, samples.shape[0]):
            noise = rng.normal(size=samples[time_idx].shape) + 1j * rng.normal(
                size=samples[time_idx].shape
            )
            samples[time_idx] = 0.8 * samples[time_idx - 1] + 0.05 * noise
        error_variance = np.full(samples.shape, 0.02, dtype=np.float32)

        predictor = configured_wesn_pred(rc_config, 25, 1, 1)
        predictor.fit_offline(samples[:8], error_variance[:8])
        prediction = predictor.predict_online(samples[:8])
        metrics = predictor.predictor_complexity_metrics

        self.assertEqual(metrics["method"], "configured_wesn_balanced_lite")
        self.assertEqual(metrics["model_num_resource_blocks"], 3)
        self.assertEqual(prediction.shape, samples[0].shape)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(
            all(
                retained >= 0.90
                for retained in metrics[
                    "balanced_retained_hankel_energy_per_mode"
                ]
            )
        )

    def test_low_rank_readout_uses_reference_centered_ridge_closed_form(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex128
        predictor.enable_skip_connections = True
        predictor.enable_residue_low_rank = True
        predictor.state_dim = 1
        predictor.input_dim = 1
        predictor.feature_dim = 2
        predictor.output_dim = 1
        predictor.reservoir_readout_reg = 0.5
        predictor.skip_readout_reg = 0.25
        predictor.low_rank_readout_mode = "centered_ridge"
        predictor.W_out_reference = np.array([[2.0 + 0.5j, 0.0]])
        predictor.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}

        states = np.array([[[1.0]], [[2.0]], [[-1.0]]], dtype=np.complex128)
        inputs = np.array([[[0.5j]], [[1.0]], [[2.0j]]], dtype=np.complex128)
        targets = np.array([[[1.0]], [[3.0j]], [[-2.0]]], dtype=np.complex128)

        features = np.concatenate([states, inputs], axis=-1).reshape(-1, 2)
        target_matrix = targets.reshape(-1, 1)
        regularization = len(features) * np.array([0.5, 0.25])
        expected_t = np.linalg.solve(
            features.conj().T @ features + np.diag(regularization),
            features.conj().T @ target_matrix
            + regularization[:, None] * predictor.W_out_reference.T,
        )

        predictor._fit_readout(states, inputs, targets)

        np.testing.assert_allclose(predictor.W_out, expected_t.T)
        self.assertIn("readout_residual_fro_norm", predictor.predictor_complexity_metrics)

    def test_low_rank_matched_readout_uses_configured_wesn_ridge(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex128
        predictor.enable_skip_connections = True
        predictor.enable_residue_low_rank = True
        predictor.low_rank_readout_mode = "matched_ridge"
        predictor.state_dim = 1
        predictor.input_dim = 1
        predictor.feature_dim = 2
        predictor.output_dim = 1
        predictor.esn_ls_reg = 0.125
        predictor.W_out_reference = np.array([[20.0, -10.0]])
        predictor.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}

        states = np.array([[[1.0]], [[2.0]], [[-1.0]]], dtype=np.complex128)
        inputs = np.array([[[0.5j]], [[1.0]], [[2.0j]]], dtype=np.complex128)
        targets = np.array([[[1.0]], [[3.0j]], [[-2.0]]], dtype=np.complex128)
        features = np.concatenate([states, inputs], axis=-1).reshape(-1, 2)
        target_matrix = targets.reshape(-1, 1)
        expected = (
            np.linalg.pinv(
                features.conj().T @ features
                + predictor.esn_ls_reg * np.eye(2)
            )
            @ features.conj().T
            @ target_matrix
        ).T

        predictor._fit_readout(states, inputs, targets)

        np.testing.assert_allclose(predictor.W_out, expected)
        self.assertNotIn(
            "readout_residual_fro_norm", predictor.predictor_complexity_metrics
        )

    def test_matched_ridge_accumulates_ill_conditioned_system_in_complex128(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex64
        predictor.enable_skip_connections = True
        predictor.enable_residue_low_rank = True
        predictor.low_rank_readout_mode = "matched_ridge"
        predictor.state_dim = 1
        predictor.input_dim = 1
        predictor.feature_dim = 2
        predictor.output_dim = 1
        predictor.esn_ls_reg = 1e-4
        predictor.esn_diagnostics = True
        predictor.W_out_reference = np.zeros((1, 2), dtype=np.complex64)
        predictor.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}

        states = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.complex64).reshape(4, 1, 1)
        inputs = np.array(
            [1000.0, 1000.001, 999.999, 1000.002], dtype=np.complex64
        ).reshape(4, 1, 1)
        targets = np.array([1.0, 1.2, 0.8, 1.4], dtype=np.complex64).reshape(4, 1, 1)

        features_128 = np.concatenate([states, inputs], axis=-1).reshape(-1, 2).astype(np.complex128)
        targets_128 = targets.reshape(-1, 1).astype(np.complex128)
        expected = np.linalg.solve(
            features_128.conj().T @ features_128
            + predictor.esn_ls_reg * np.eye(2, dtype=np.complex128),
            features_128.conj().T @ targets_128,
        ).T

        predictor._fit_readout(states, inputs, targets)

        np.testing.assert_allclose(predictor.W_out, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(
            predictor.predictor_complexity_metrics["readout_solve_dtype"],
            "complex128",
        )
        self.assertGreater(
            predictor.predictor_complexity_metrics["readout_gram_condition_first"],
            1e6,
        )

    def test_low_rank_rollout_applies_configured_tanh_activation(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex128
        predictor.state_dim = 1
        predictor.input_scale = 0.15
        predictor.esn_activation = "tanh"
        predictor.enable_residue_low_rank = True
        predictor.W_in = np.array([[3.0]], dtype=np.complex128)
        predictor.W_res = np.array([0.5], dtype=np.complex128)
        inputs = np.array([[[2.0]], [[1.0]]], dtype=np.complex128)

        states = predictor._state_rollout(inputs)

        first = np.tanh(6.0)
        second = np.tanh(3.0 + 0.5 * first)
        np.testing.assert_allclose(states[:, 0, 0], [first, second])

    def test_complexity_instrumentation_reports_latency_quantiles(self):
        owner = types.SimpleNamespace()
        with measure_phase(owner, "inference_system", method="wesn_lite", workers=1):
            np.zeros((8,), dtype=np.float64)
        summary = phase_summary(owner.predictor_complexity_metrics)

        inference = summary["phases"]["inference_system"]
        self.assertEqual(inference["samples"], 1)
        self.assertGreaterEqual(inference["p99_seconds"], 0.0)
        self.assertIn("max_process_peak_rss_bytes", inference)

    def test_diagonal_wesn_recurrence_matches_dense_product(self):
        predictor = object.__new__(configured_wesn_pred)
        predictor.dtype = np.complex128
        predictor.state_dim = 2
        predictor.input_scale = 1.0
        predictor.esn_activation = "identity"
        predictor.W_in = np.array(
            [[1.0, 0.2], [-0.1, 0.8]], dtype=np.complex128
        )
        predictor.W_res = np.array([0.7, -0.3], dtype=np.complex128)
        inputs = np.array(
            [
                [[1.0, 2.0]],
                [[-0.5, 0.4]],
                [[0.2, -0.1]],
            ],
            dtype=np.complex128,
        )

        actual = predictor._state_rollout(inputs)
        dense_reservoir = np.diag(predictor.W_res)
        state = np.zeros((1, 2), dtype=np.complex128)
        expected = []
        for input_t in inputs:
            state = input_t @ predictor.W_in.T + state @ dense_reservoir.T
            expected.append(state.copy())

        np.testing.assert_allclose(actual, np.asarray(expected))

    def test_structured_companion_transition_matches_dense_product(self):
        predictor = steady_state_kalman_filter_pred(ar_order=2)
        predictor.p = 2
        predictor.d = 2
        a1 = np.array([[0.7, 0.1], [-0.2, 0.6]], dtype=np.complex128)
        a2 = np.array([[0.1, -0.1], [0.05, 0.2]], dtype=np.complex128)
        predictor.f_aug = np.block(
            [[a1, a2], [np.eye(2, dtype=np.complex128), np.zeros((2, 2))]]
        )
        state = np.array(
            [[1.0 + 0.2j, 2.0 - 0.1j, 3.0 + 0.4j, 4.0 - 0.3j]],
            dtype=np.complex128,
        )

        dense = state @ predictor.f_aug.T
        structured = predictor._companion_predict_state(state)

        np.testing.assert_allclose(structured, dense)
        np.testing.assert_allclose(
            predictor._companion_predict_observation(state), dense[:, :2]
        )

    def test_riccati_convergence_is_recorded(self):
        predictor = _configured_scalar_predictor()

        self.assertTrue(predictor.riccati_converged)
        self.assertLess(predictor.riccati_iterations, predictor.max_riccati_iters)
        self.assertLessEqual(
            predictor.riccati_relative_change, predictor.riccati_tol
        )

    def test_riccati_nonconvergence_raises(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            _configured_scalar_predictor(max_iters=1, tol=1e-15)

    def test_filter_state_is_preserved_across_prediction_events(self):
        predictor = _configured_scalar_predictor()

        predictor.predict(_history([1.0, 2.0, 3.0]))
        self.assertEqual(predictor.num_state_updates_last_predict, 2)
        prior = predictor._state.copy()

        newest_observation = np.array([[4.0]], dtype=np.complex128)
        innovation = newest_observation - prior[:, : predictor.d]
        expected_posterior = prior + innovation @ predictor.k_gain.T
        expected_state = predictor._companion_predict_state(expected_posterior)

        actual = predictor.predict(_history([2.0, 3.0, 4.0]))

        self.assertEqual(predictor.num_state_updates_last_predict, 1)
        np.testing.assert_allclose(predictor._state, expected_state)
        np.testing.assert_allclose(
            actual.reshape(-1), expected_state[:, : predictor.d].reshape(-1)
        )

        predictor.reset_state()
        predictor.predict(_history([2.0, 3.0, 4.0]))
        self.assertEqual(predictor.num_state_updates_last_predict, 2)


if __name__ == "__main__":
    unittest.main()
