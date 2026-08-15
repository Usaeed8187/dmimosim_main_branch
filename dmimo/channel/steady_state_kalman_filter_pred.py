"""Drop-configured steady-state Kalman channel predictor.

The transition, process covariance, measurement covariance, and Kalman gain
are estimated once from an offline segment.  Online prediction keeps these
quantities fixed and performs only fixed-gain state updates and prediction.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from dmimo.channel.kalman_filter_pred import kalman_filter_pred
from dmimo.channel.complexity_instrumentation import measure_phase


class steady_state_kalman_filter_pred:
    def __init__(
        self,
        ar_order=2,
        lam=1e-3,
        eps=1e-8,
        max_riccati_iters=5000,
        riccati_tol=1e-8,
    ):
        self.ar_order = int(ar_order)
        self.lam = float(lam)
        self.eps = float(eps)
        self.max_riccati_iters = int(max_riccati_iters)
        self.riccati_tol = float(riccati_tol)
        self.predictor_complexity_metrics = {"schema_version": 1, "phases": {}}
        self.is_configured = False
        self.reset_state()

    def reset_state(self):
        """Discard the online filter state without changing configuration."""
        self._state = None
        self._num_tiles = None
        self.num_state_updates_last_predict = 0

    def _align_error_variance(self, h_hist, e_hist):
        if h_hist.shape == e_hist.shape:
            return e_hist
        sc_diff = h_hist.shape[-1] - e_hist.shape[-1]
        if sc_diff < 0:
            raise ValueError("Error-variance history has more subcarriers than CSI history.")
        left = np.repeat(e_hist[..., :1], sc_diff // 2, axis=-1)
        right = np.repeat(e_hist[..., -1:], sc_diff - sc_diff // 2, axis=-1)
        aligned = np.concatenate([left, e_hist, right], axis=-1)
        if aligned.shape != h_hist.shape:
            raise ValueError("CSI and error-variance histories could not be aligned.")
        return aligned

    @staticmethod
    def _link_tiles(history):
        # [T, batch, 1, Nr, 1, Nt, sym, sc] -> [T, batch*sym*sc, Nr*Nt]
        t_len, num_batches, _, num_rx, _, num_tx, num_syms, num_sc = history.shape
        tiles = history[:, :, 0, :, 0, :, :, :]
        tiles = tiles.transpose(0, 1, 4, 5, 2, 3).reshape(
            t_len, num_batches * num_syms * num_sc, num_rx * num_tx
        )
        return tiles

    def _companion_predict_state(self, state):
        """Apply the AR companion transition without a dense (PD)x(PD) product."""
        newest_block = state @ self.f_aug[: self.d, :].T
        if self.p == 1:
            return newest_block
        return np.concatenate([newest_block, state[:, : (self.p - 1) * self.d]], axis=1)

    def _companion_predict_observation(self, state):
        """Return H F state using only the dense AR block row."""
        return state @ self.f_aug[: self.d, :].T

    def _solve_steady_state_gain(self, f_aug, q_aug, r_diag):
        d = r_diag.size
        n_state = f_aug.shape[0]
        p = n_state // d
        h_mat = np.zeros((d, n_state), dtype=np.complex128)
        h_mat[:, :d] = np.eye(d, dtype=np.complex128)
        r_mat = np.diag(np.maximum(r_diag, self.eps)).astype(np.complex128)
        covariance = np.diag(np.tile(np.maximum(r_diag, self.eps), p)).astype(
            np.complex128
        )
        eye_state = np.eye(n_state, dtype=np.complex128)
        gain = np.zeros((n_state, d), dtype=np.complex128)

        converged = False
        relative_change = float("inf")
        for iteration in range(self.max_riccati_iters):
            pred_cov = f_aug @ covariance @ f_aug.conj().T + q_aug
            innovation_cov = h_mat @ pred_cov @ h_mat.conj().T + r_mat
            gain = pred_cov @ h_mat.conj().T @ np.linalg.pinv(innovation_cov)
            next_cov = (eye_state - gain @ h_mat) @ pred_cov
            next_cov = 0.5 * (next_cov + next_cov.conj().T)
            scale = max(float(np.linalg.norm(covariance, ord="fro")), self.eps)
            relative_change = float(
                np.linalg.norm(next_cov - covariance, ord="fro") / scale
            )
            covariance = next_cov
            if relative_change <= self.riccati_tol:
                converged = True
                break

        if not np.all(np.isfinite(gain)):
            raise FloatingPointError("Steady-state Riccati iteration produced a non-finite gain.")
        self.riccati_iterations = iteration + 1
        self.riccati_converged = converged
        self.riccati_relative_change = relative_change
        if not converged:
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(f_aug))))
            raise RuntimeError(
                "Steady-state Riccati iteration did not converge: "
                f"iterations={self.riccati_iterations}, "
                f"relative_change={relative_change:.3e}, "
                f"tolerance={self.riccati_tol:.3e}, "
                f"spectral_radius(F)={spectral_radius:.6f}."
            )
        return h_mat, gain

    def fit_offline(self, h_freq_csi_history, err_var_history):
        h_hist = np.asarray(h_freq_csi_history)
        e_hist = np.asarray(err_var_history)
        if h_hist.ndim != 8:
            raise ValueError("Steady-state KF expects an 8D CSI history tensor.")
        e_hist = self._align_error_variance(h_hist, e_hist)

        y_tiles = self._link_tiles(h_hist).astype(np.complex128)
        e_tiles = np.real(self._link_tiles(e_hist))
        p = min(self.ar_order, y_tiles.shape[0] - 1)
        if p < 1:
            raise ValueError("Steady-state KF configuration requires at least two samples.")

        helper = kalman_filter_pred(
            lam=self.lam,
            eps=self.eps,
            ar_order=p,
            debug=False,
        )
        with measure_phase(self, "configuration_ar"):
            a_blocks, q_proc = helper._estimate_ar_p_q_joint(y_tiles, p)
        # Match the coefficient convention used by the existing full KF.
        selected_blocks = [a_block.conj() for a_block in a_blocks]
        self.f_aug, self.q_aug = helper._build_augmented_system(
            selected_blocks, q_proc
        )
        self.r_diag = np.maximum(np.mean(e_tiles, axis=(0, 1)), self.eps)
        with measure_phase(self, "configuration_riccati"):
            self.h_mat, self.k_gain = self._solve_steady_state_gain(
                self.f_aug, self.q_aug, self.r_diag
            )
        self.p = p
        self.d = y_tiles.shape[-1]
        self.predictor_complexity_metrics.update(
            {
                "method": "steady_state_kalman_filter",
                "ar_order": self.p,
                "channel_dimension": self.d,
                "state_dimension": self.p * self.d,
                "riccati_iterations": self.riccati_iterations,
                "riccati_converged": self.riccati_converged,
            }
        )
        self.is_configured = True
        self.reset_state()
        return self

    def predict(self, h_freq_csi_history, err_var_history=None):
        if not self.is_configured:
            raise ValueError("Steady-state KF must be configured before prediction.")
        h_hist = np.asarray(h_freq_csi_history)
        if h_hist.ndim != 8:
            raise ValueError("Steady-state KF expects an 8D CSI history tensor.")

        t_len, num_batches, _, num_rx, _, num_tx, num_syms, num_sc = h_hist.shape
        y_tiles = self._link_tiles(h_hist).astype(np.complex128)
        if y_tiles.shape[-1] != self.d:
            raise ValueError("Online CSI dimensions differ from configured dimensions.")

        num_tiles = y_tiles.shape[1]
        if self._state is not None and self._num_tiles != num_tiles:
            raise ValueError(
                "Online CSI tile count changed while preserving steady-state KF state. "
                "Call reset_state() before changing the batch/resource-grid dimensions."
            )

        self.num_state_updates_last_predict = 0
        if t_len <= self.p:
            y_next = y_tiles[-1]
            self.reset_state()
        elif self._state is None:
            # Initialize once, then retain the posterior at the newest observation
            # for the next prediction event. Consecutive simulation events advance
            # by one CSI-history interval, so only their newest observation is new.
            state = np.concatenate(
                [y_tiles[self.p - 1 - lag] for lag in range(self.p)], axis=1
            )
            for t_idx in range(self.p, t_len):
                state_pred = self._companion_predict_state(state)
                innovation = y_tiles[t_idx] - state_pred[:, : self.d]
                state = state_pred + innovation @ self.k_gain.T
                self.num_state_updates_last_predict += 1
            # Preserve the prior for the next observation. This makes each
            # subsequent event one fixed-gain update plus one AR prediction.
            self._state = self._companion_predict_state(state)
            self._num_tiles = num_tiles
            y_next = self._state[:, : self.d]
        else:
            newest_observation = y_tiles[-1]
            innovation = newest_observation - self._state[:, : self.d]
            posterior_state = self._state + innovation @ self.k_gain.T
            self._state = self._companion_predict_state(posterior_state)
            self.num_state_updates_last_predict = 1
            y_next = self._state[:, : self.d]

        block = y_next.reshape(
            num_batches, num_syms, num_sc, num_rx, num_tx
        ).transpose(0, 3, 4, 1, 2)
        pred = np.zeros_like(h_hist[0])
        pred[:, 0, :, 0, :, :, :] = block.astype(h_hist.dtype, copy=False)
        return pred


def build_steady_state_kalman_predictors_simple(
    h_freq_csi_history,
    err_var_csi_history,
    rc_config,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
):
    predictors = {}
    for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
            tx_ant_idx = np.arange(0, num_bs_ant) if tx_node_idx == 0 else np.arange(
                num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
                num_bs_ant + tx_node_idx * num_ue_ant,
            )
            rx_ant_idx = np.arange(0, num_bs_ant) if rx_node_idx == 0 else np.arange(
                num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
                num_bs_ant + rx_node_idx * num_ue_ant,
            )
            curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]
            curr_e = err_var_csi_history[:, :, :, rx_ant_idx, :, ...]
            curr_e = curr_e[:, :, :, :, :, tx_ant_idx, ...]

            predictor = steady_state_kalman_filter_pred(
                ar_order=rc_config.window_length,
                max_riccati_iters=int(
                    getattr(rc_config, "steady_state_kf_max_iters", 5000)
                ),
                riccati_tol=float(
                    getattr(rc_config, "steady_state_kf_tolerance", 1e-8)
                ),
            )
            predictor.fit_offline(curr_h, curr_e)
            predictors[(tx_node_idx, rx_node_idx)] = predictor
    return predictors


def predict_all_links_with_steady_state_kalman_simple(
    h_freq_csi_history,
    predictors,
    ns3cfg,
    num_bs_ant=4,
    num_ue_ant=2,
    max_workers=1,
):
    prediction = np.zeros_like(h_freq_csi_history[0])

    def predict_link(tx_node_idx, rx_node_idx):
        tx_ant_idx = np.arange(0, num_bs_ant) if tx_node_idx == 0 else np.arange(
            num_bs_ant + (tx_node_idx - 1) * num_ue_ant,
            num_bs_ant + tx_node_idx * num_ue_ant,
        )
        rx_ant_idx = np.arange(0, num_bs_ant) if rx_node_idx == 0 else np.arange(
            num_bs_ant + (rx_node_idx - 1) * num_ue_ant,
            num_bs_ant + rx_node_idx * num_ue_ant,
        )
        curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
        curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]
        with measure_phase(predictors[(tx_node_idx, rx_node_idx)], "inference_fixed_gain"):
            tmp = predictors[(tx_node_idx, rx_node_idx)].predict(curr_h)
        return rx_ant_idx, tx_ant_idx, tmp

    links = [
        (tx_node_idx, rx_node_idx)
        for tx_node_idx in range(ns3cfg.num_txue_sel + 1)
        for rx_node_idx in range(ns3cfg.num_rxue_sel + 1)
    ]
    if int(max_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            results = list(executor.map(lambda pair: predict_link(*pair), links))
    else:
        results = [predict_link(*pair) for pair in links]

    for rx_ant_idx, tx_ant_idx, tmp in results:
        rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
        prediction[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(
            2, 4, 0, 1, 3, 5, 6
        )
    return prediction
