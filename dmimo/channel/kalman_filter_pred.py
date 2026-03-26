import numpy as np


class kalman_filter_pred:

    def __init__(self, lam=1e-3, eps=1e-8, ar_order=4, debug=False):
        self.lam = lam
        self.eps = eps
        self.ar_order = ar_order
        self.debug = debug
        self.num_bs_ant = 4
        self.num_ue_ant = 2
    
    def _debug_print(self, msg):
        if self.debug:
            print(msg)

    def _debug_pass_fail(self, check_name, passed, details=""):
        status = "PASS" if passed else "FAIL"
        detail_msg = f" | {details}" if details else ""
        self._debug_print(f"[DEBUG][{status}] {check_name}{detail_msg}")


    def _estimate_ar_p_q_joint(self, y_hist_tiles, p):
        """Estimate shared AR(p) coefficients and process covariance across tiles.

        y_hist_tiles: [T, Ntiles, D] complex.
        Returns:
            a_blocks: list of p matrices [D,D], where
                x_t ≈ sum_{k=1..p} A_k x_{t-k}
            q_proc: process covariance for x_t, shape [D,D]

        """
        t_len, _, d = y_hist_tiles.shape
        if p < 1:
            raise ValueError("AR order p must be >= 1")
        if t_len <= p:
            ident = np.eye(d, dtype=np.complex128)
            return [ident] + [np.zeros((d, d), dtype=np.complex128) for _ in range(p - 1)], self.eps * ident

        x_target = y_hist_tiles[p:].reshape(-1, d)  # [N, D]
        x_lags = []
        for k in range(1, p + 1):
            x_lags.append(y_hist_tiles[p - k:t_len - k].reshape(-1, d))
        phi = np.concatenate(x_lags, axis=1)  # [N, pD]

        n_samples = max(phi.shape[0], 1)
        r0 = (phi.conj().T @ phi) / n_samples
        r1 = (x_target.conj().T @ phi) / n_samples
        theta = r1 @ np.linalg.pinv(
            r0 + self.lam * np.eye(p * d, dtype=np.complex128)
        )  # [D, pD]

        residual = x_target - phi @ theta.conj().T
        q_proc = (residual.conj().T @ residual) / max(residual.shape[0], 1)
        q_proc = 0.5 * (q_proc + q_proc.conj().T)
        q_proc += self.eps * np.eye(d, dtype=np.complex128)

        a_blocks = [theta[:, k * d:(k + 1) * d] for k in range(p)]
        return a_blocks, q_proc

    def _build_augmented_system(self, a_blocks, q_proc):
        """Build AR(p) companion-form state-space matrices.

        Expects column-state transition blocks: x_t(col) = sum_k A_k(col) x_{t-k}(col).
        If coefficients were learned in row-regression form
        x_t(row) ≈ sum_k x_{t-k}(row) A_k(row)^H, convert before calling.
        """
        p = len(a_blocks)
        d = a_blocks[0].shape[0]
        pd = p * d

        f_aug = np.zeros((pd, pd), dtype=np.complex128)
        f_aug[:d, :] = np.concatenate(a_blocks, axis=1)
        for row in range(1, p):
            f_aug[row * d:(row + 1) * d, (row - 1) * d:row * d] = np.eye(d, dtype=np.complex128)

        q_aug = np.zeros((pd, pd), dtype=np.complex128)
        q_aug[:d, :d] = q_proc
        q_aug += self.eps * np.eye(pd, dtype=np.complex128)
        return f_aug, q_aug

    def _kalman_predict_one_step_ar_p(self, y_hist, r_diag, f_aug, q_aug, return_debug_stats=False):
        """Run Kalman filter for AR(p) augmented state and predict next x."""

        t_len, d = y_hist.shape
        p = f_aug.shape[0] // d
        if t_len <= p:
            return y_hist[-1].astype(np.complex128)

        r_diag = np.maximum(np.asarray(r_diag, dtype=np.float64), self.eps)
        r_mat = np.diag(r_diag.astype(np.complex128))

        p0_diag = np.tile(r_diag, p)

        h_mat = np.zeros((d, p * d), dtype=np.complex128)
        h_mat[:, :d] = np.eye(d, dtype=np.complex128)

        state_stack = [y_hist[p - 1 - k].astype(np.complex128) for k in range(p)]
        z_hat = np.concatenate(state_stack, axis=0)
        p_hat = np.diag(p0_diag + self.eps).astype(np.complex128)
        eye_pd = np.eye(p * d, dtype=np.complex128)

        innovation_power_list = []
        nis_list = []
        kh_diag_list = []

        for t_idx in range(p, t_len):
            z_pred = f_aug @ z_hat
            p_pred = f_aug @ p_hat @ f_aug.conj().T + q_aug

            y_obs = y_hist[t_idx].astype(np.complex128)
            innovation = y_obs - h_mat @ z_pred
            s_mat = h_mat @ p_pred @ h_mat.conj().T + r_mat
            k_gain = p_pred @ h_mat.conj().T @ np.linalg.pinv(s_mat)

            if return_debug_stats:
                innovation_power = float(np.real((innovation.conj().T @ innovation) / max(d, 1)))
                innovation_power_list.append(innovation_power)
                nis_val = float(np.real(innovation.conj().T @ np.linalg.pinv(s_mat) @ innovation) / max(d, 1))
                nis_list.append(nis_val)
                kh_diag = np.diag(k_gain @ h_mat)
                kh_diag_mean = float(np.real(np.mean(kh_diag)))
                kh_diag_list.append(kh_diag_mean)

            z_hat = z_pred + k_gain @ innovation
            p_hat = (eye_pd - k_gain @ h_mat) @ p_pred

        z_next = f_aug @ z_hat
        y_next = z_next[:d]

        if not return_debug_stats:
            return y_next

        stats = {
            "innovation_power_mean": float(np.mean(innovation_power_list)) if innovation_power_list else 0.0,
            "innovation_power_std": float(np.std(innovation_power_list)) if innovation_power_list else 0.0,
            "nis_mean": float(np.mean(nis_list)) if nis_list else 0.0,
            "nis_std": float(np.std(nis_list)) if nis_list else 0.0,
            "kh_diag_mean": float(np.mean(kh_diag_list)) if kh_diag_list else 0.0,
            "kh_diag_std": float(np.std(kh_diag_list)) if kh_diag_list else 0.0,
        }
        return y_next, stats

    def predict(self, h_freq_csi_history, err_var_history, h_freq_csi_perfect_debug=None):
        """Predict one-step-ahead channel using vector AR(p) Kalman filtering.

        h_freq_csi_history: [T, B, 1, all_rx_ants, 1, all_tx_ants, SYM, SC]
        err_var_history: same shape, real-valued channel estimation error variance.
        """

        h_hist = np.asarray(h_freq_csi_history)
        e_hist = np.asarray(err_var_history)
        if h_freq_csi_perfect_debug is not None:
            h_freq_csi_perfect_debug = np.asarray(h_freq_csi_perfect_debug)

        if h_hist.shape != e_hist.shape:
            sc_diff = h_hist.shape[-1] - e_hist.shape[-1]
            left_pad  = np.repeat(e_hist[..., :1], sc_diff // 2, axis=-1)
            right_pad = np.repeat(e_hist[..., -1:], sc_diff - sc_diff // 2, axis=-1)
            e_hist = np.concatenate([left_pad, e_hist, right_pad], axis=-1)

            if h_hist.shape != e_hist.shape:
                raise ValueError("h_freq_csi_history and err_var_history must have the same shape")
        if h_hist.ndim != 8:
            raise ValueError("Expected history tensor rank 8")

        t_len, num_batches, _, num_rx_ants_all, _, num_tx_ants_all, num_syms, num_sc = h_hist.shape
        num_rx_nodes = ((num_rx_ants_all - self.num_bs_ant) // self.num_ue_ant) + 1
        num_tx_nodes = ((num_tx_ants_all - self.num_bs_ant) // self.num_ue_ant) + 1

        pred = np.zeros_like(h_hist[0], dtype=np.complex64)

        for batch_idx in range(num_batches):
            for rx_node in range(num_rx_nodes):
                for tx_node in range(num_tx_nodes):

                    if tx_node == 0:
                        tx_ant_idx = np.arange(0, self.num_bs_ant)
                    else:
                        tx_ant_idx = np.arange(
                            self.num_bs_ant + (tx_node - 1) * self.num_ue_ant,
                            self.num_bs_ant + (tx_node) * self.num_ue_ant,
                        )
                    TxAnt = len(tx_ant_idx)
                    if rx_node == 0:
                        rx_ant_idx = np.arange(0, self.num_bs_ant)
                    else:
                        rx_ant_idx = np.arange(
                            self.num_bs_ant + (rx_node - 1) * self.num_ue_ant,
                            self.num_bs_ant + (rx_node) * self.num_ue_ant,
                        )
                    RxAnt = len(rx_ant_idx)

                    curr_h_freq_csi_history = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:, :, :, :, :, tx_ant_idx, ...]

                    curr_e_hist = e_hist[:, :, :, rx_ant_idx, :, ...]
                    curr_e_hist = curr_e_hist[:, :, :, :, :, tx_ant_idx, ...]

                    curr_hist = curr_h_freq_csi_history[:, batch_idx, 0, :, 0, :, :, :]
                    y_hist_tiles = curr_hist.transpose(0, 3, 4, 1, 2).reshape(t_len, num_syms * num_sc, -1)

                    curr_evar = curr_e_hist[:, batch_idx, 0, :, 0, :, :, :]
                    e_tiles = np.real(curr_evar).transpose(0, 3, 4, 1, 2).reshape(t_len, num_syms * num_sc, -1)

                    p = min(self.ar_order, t_len - 1)
                    a_blocks, q_proc = self._estimate_ar_p_q_joint(y_hist_tiles, p)

                    use_debug = self.debug and h_freq_csi_perfect_debug is not None

                    if use_debug and h_freq_csi_perfect_debug is not None:
                        joint_wiener_pred_tiles = np.zeros((num_syms * num_sc, RxAnt * TxAnt), dtype=np.complex128)
                        for lag_idx in range(1, p + 1):
                            joint_wiener_pred_tiles += y_hist_tiles[-lag_idx] @ a_blocks[lag_idx - 1].conj().T

                        joint_wiener_pred = joint_wiener_pred_tiles.reshape(num_syms, num_sc, RxAnt, TxAnt).transpose(2, 3, 0, 1)
                        curr_perfect_block = h_freq_csi_perfect_debug[ :, :, :, rx_ant_idx,  ...]
                        curr_perfect_block = np.squeeze(curr_perfect_block[:, :, :, :, :, tx_ant_idx, ...])
                        numer = np.linalg.norm(joint_wiener_pred - curr_perfect_block[-1,...]) ** 2
                        denom = np.linalg.norm(curr_perfect_block[-1,...]) ** 2 + self.eps
                        weiner_nmse = float(np.real(numer / denom))
                        print("Weiner Filter NMSE: ", weiner_nmse)
                    
                    selected_model_name = "conj"
                    selected_a_blocks = [a_block.conj() for a_block in a_blocks]
                    selected_f_aug, selected_q_aug = self._build_augmented_system(selected_a_blocks, q_proc)

                    if use_debug:
                        has_perfect_history = h_freq_csi_perfect_debug.shape[0] == t_len
                        self._debug_pass_fail(
                            "Perfect history availability",
                            has_perfect_history,
                            f"expected first dim={t_len}, got={h_freq_csi_perfect_debug.shape[0]}"
                        )

                        if has_perfect_history:
                            curr_perfect_hist = h_freq_csi_perfect_debug[:, batch_idx, 0, :, 0, :, :, :]
                            curr_perfect_hist = curr_perfect_hist[:, rx_ant_idx, :, :, :]
                            curr_perfect_hist = curr_perfect_hist[:, :, tx_ant_idx, :, :]
                            x_hist_tiles = curr_perfect_hist.transpose(0, 3, 4, 1, 2).reshape(t_len, num_syms * num_sc, -1).astype(np.complex128)

                            residual_list = []
                            for t_idx in range(p, t_len):
                                x_hat_t = np.zeros((num_syms * num_sc, RxAnt * TxAnt), dtype=np.complex128)
                                for lag_idx in range(1, p + 1):
                                    x_hat_t += x_hist_tiles[t_idx - lag_idx] @ a_blocks[lag_idx - 1].conj().T
                                residual_list.append(x_hist_tiles[t_idx] - x_hat_t)
                            if residual_list:
                                w_res = np.concatenate(residual_list, axis=0)
                                q_oracle = (w_res.conj().T @ w_res) / max(w_res.shape[0], 1)
                                q_oracle = 0.5 * (q_oracle + q_oracle.conj().T) + self.eps * np.eye(q_oracle.shape[0], dtype=np.complex128)
                            else:
                                q_oracle = self.eps * np.eye(q_proc.shape[0], dtype=np.complex128)

                            v_res = y_hist_tiles - x_hist_tiles
                            r_oracle_diag = np.real(np.mean(np.abs(v_res) ** 2, axis=(0, 1))) + self.eps
                            r_curr_diag = np.real(np.mean(e_tiles, axis=(0, 1))) + self.eps

                            q_diag_ratio = np.real(np.diag(q_oracle)) / (np.real(np.diag(q_proc)) + self.eps)
                            r_diag_ratio = r_oracle_diag / (r_curr_diag + self.eps)
                            q_diag_ratio_median = float(np.median(q_diag_ratio))
                            r_diag_ratio_median = float(np.median(r_diag_ratio))

                            q_pass = 0.5 <= q_diag_ratio_median <= 2.0
                            r_pass = 0.5 <= r_diag_ratio_median <= 2.0
                            self._debug_pass_fail(
                                "Oracle Q consistency",
                                q_pass,
                                f"median(diag(Q_oracle)/diag(Q_est))={q_diag_ratio_median:.4f}"
                            )
                            self._debug_pass_fail(
                                "Oracle R consistency",
                                r_pass,
                                f"median(diag(R_oracle)/diag(R_est))={r_diag_ratio_median:.4f}"
                            )

                            model_candidates = {
                                "none": a_blocks,
                                "conj": [a_block.conj() for a_block in a_blocks],
                                "conjT": [a_block.conj().T for a_block in a_blocks],
                            }
                            model_nmse = {}
                            x_target_tiles = x_hist_tiles[-1]
                            for model_name, model_blocks in model_candidates.items():
                                x_pred_tiles = np.zeros_like(x_target_tiles)
                                for lag_idx in range(1, p + 1):
                                    x_pred_tiles += x_hist_tiles[-lag_idx - 1] @ model_blocks[lag_idx - 1].conj().T
                                num = np.linalg.norm(x_pred_tiles - x_target_tiles) ** 2
                                den = np.linalg.norm(x_target_tiles) ** 2 + self.eps
                                model_nmse[model_name] = float(np.real(num / den))

                            best_model_name = min(model_nmse, key=model_nmse.get)
                            self._debug_pass_fail(
                                "AR convention check",
                                best_model_name == selected_model_name,
                                f"nmse={model_nmse}, selected='{best_model_name}', active='{selected_model_name}'"
                            )
                            # Keep inference behavior identical between debug and non-debug runs.
                            # Diagnostics should never switch the active transition model.
                            # Non-debug mode uses the "conj" convention by default.
                            selected_model_name = "conj"
                        else:
                            self._debug_print(
                                "[DEBUG] Skipping oracle Q/R and AR convention diagnostics because perfect history was not provided."
                            )
                    else:
                        self._debug_print("[DEBUG] Diagnostics skipped (requires debug=True and perfect debug channel).")

                    y_next_tiles = np.zeros((num_syms * num_sc, RxAnt * TxAnt), dtype=np.complex128)

                    kalman_debug_stats = []
                    
                    for tile_idx in range(num_syms * num_sc):
                        y_hist = y_hist_tiles[:, tile_idx, :]
                        r_diag = e_tiles[:, tile_idx, :].mean(axis=0)
                        if use_debug:
                            y_next_tile, tile_stats = self._kalman_predict_one_step_ar_p(
                                y_hist, r_diag, f_aug=selected_f_aug, q_aug=selected_q_aug, return_debug_stats=True
                            )
                            kalman_debug_stats.append(tile_stats)
                            y_next_tiles[tile_idx] = y_next_tile
                        else:
                            y_next_tiles[tile_idx] = self._kalman_predict_one_step_ar_p(
                                y_hist, r_diag, f_aug=selected_f_aug, q_aug=selected_q_aug
                            )
                    
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    y_next_block = y_next_tiles.reshape(num_syms, num_sc, RxAnt, TxAnt).transpose(2, 3, 0, 1)
                    pred[batch_idx, 0, rx_idx, 0, tx_idx, :, :] = y_next_block

                    if use_debug and kalman_debug_stats:
                        innovation_power_mean = float(np.mean([s["innovation_power_mean"] for s in kalman_debug_stats]))
                        nis_mean = float(np.mean([s["nis_mean"] for s in kalman_debug_stats]))
                        kh_diag_mean = float(np.mean([s["kh_diag_mean"] for s in kalman_debug_stats]))
                        self._debug_pass_fail(
                            "Kalman NIS check",
                            0.5 <= nis_mean <= 2.0,
                            f"mean NIS={nis_mean:.4f} (target approx 1.0)"
                        )
                        self._debug_pass_fail(
                            "Kalman gain check",
                            0.0 <= kh_diag_mean <= 1.2,
                            f"mean diag(KH)={kh_diag_mean:.4f}"
                        )
                        self._debug_print(
                            f"[DEBUG] Innovation power summary: mean={innovation_power_mean:.6f}, "
                            f"transition='{selected_model_name}'"
                        )

                    if use_debug and h_freq_csi_perfect_debug is not None:
                        numer = np.linalg.norm(y_next_block - curr_perfect_block[-1,...]) ** 2
                        denom = np.linalg.norm(curr_perfect_block[-1,...]) ** 2 + self.eps
                        kalman_nmse = float(np.real(numer / denom))
                        print("Kalman Filter NMSE: ", kalman_nmse, "\n")

                    hold = 1

        return pred.astype(h_hist.dtype, copy=False)
