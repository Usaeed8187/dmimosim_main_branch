import numpy as np


class kalman_filter_pred:

    def __init__(self, lam=1e-3, eps=1e-8):
        self.lam = lam
        self.eps = eps
        self.num_bs_ant = 4
        self.num_ue_ant = 2

    def _estimate_f_q(self, y_hist):
        """Estimate AR(1) transition matrix F and process covariance Q.

        y_hist: [T, D] complex, row-wise temporal samples.
        """
        t_len, d = y_hist.shape
        if t_len < 2:
            return np.eye(d, dtype=np.complex128), self.eps * np.eye(d, dtype=np.complex128)

        x_prev = y_hist[:-1, :]
        x_next = y_hist[1:, :]

        r0 = (x_prev.conj().T @ x_prev) / max(t_len - 1, 1)
        r1 = (x_next.conj().T @ x_prev) / max(t_len - 1, 1)

        f_hat = r1 @ np.linalg.pinv(r0 + self.lam * np.eye(d, dtype=np.complex128))

        residual = x_next - x_prev @ f_hat.T
        q_hat = (residual.conj().T @ residual) / max(residual.shape[0], 1)
        q_hat = 0.5 * (q_hat + q_hat.conj().T)
        q_hat += self.eps * np.eye(d, dtype=np.complex128)

        return f_hat, q_hat

    def _estimate_f_q_joint(self, y_hist_tiles):
        """Estimate shared AR(1) transition F and Q across all (sym, sc) tiles.

        y_hist_tiles: [T, Ntiles, D] complex, where each tile is one (sym, sc).
        """
        t_len, _, d = y_hist_tiles.shape
        if t_len < 2:
            ident = np.eye(d, dtype=np.complex128)
            return ident, self.eps * ident

        x_prev = y_hist_tiles[:-1].reshape(-1, d)
        x_next = y_hist_tiles[1:].reshape(-1, d)

        denom = max(x_prev.shape[0], 1)
        r0 = (x_prev.conj().T @ x_prev) / denom
        r1 = (x_next.conj().T @ x_prev) / denom
        f_hat = r1 @ np.linalg.pinv(r0 + self.lam * np.eye(d, dtype=np.complex128))

        residual = x_next - x_prev @ f_hat.T
        q_hat = (residual.conj().T @ residual) / max(residual.shape[0], 1)
        q_hat = 0.5 * (q_hat + q_hat.conj().T)
        q_hat += self.eps * np.eye(d, dtype=np.complex128)
        return f_hat, q_hat

    def _kalman_predict_one_step(self, y_hist, r_diag, f_hat=None, q_hat=None):

        """Run Kalman filter over y_hist and return one-step-ahead prediction."""
        t_len, d = y_hist.shape
        if f_hat is None or q_hat is None:
            f_hat, q_hat = self._estimate_f_q(y_hist)

        r_diag = np.maximum(np.asarray(r_diag, dtype=np.float64), self.eps)
        r_mat = np.diag(r_diag.astype(np.complex128))

        x_hat = y_hist[0].astype(np.complex128)
        p_hat = np.diag(r_diag + self.eps).astype(np.complex128)
        eye = np.eye(d, dtype=np.complex128)

        for t_idx in range(1, t_len):
            x_pred = f_hat @ x_hat
            p_pred = f_hat @ p_hat @ f_hat.conj().T + q_hat

            y_obs = y_hist[t_idx].astype(np.complex128)
            innovation = y_obs - x_pred
            s_mat = p_pred + r_mat
            k_gain = p_pred @ np.linalg.pinv(s_mat)

            x_hat = x_pred + k_gain @ innovation
            p_hat = (eye - k_gain) @ p_pred

        return f_hat @ x_hat

    def predict(self, h_freq_csi_history, err_var_history, h_freq_csi_perfect_debug=None):
        """Predict one-step-ahead channel using vector AR(1) Kalman filtering.

        h_freq_csi_history: [T, B, 1, all_rx_ants, 1, all_tx_ants, SYM, SC]
        err_var_history: same shape, real-valued channel estimation error variance.
        """

        h_hist = np.asarray(h_freq_csi_history)
        e_hist = np.asarray(err_var_history)
        if h_freq_csi_perfect_debug is not None:
            h_freq_csi_perfect_debug = np.asarray(h_freq_csi_perfect_debug)

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

                    f_hat, q_hat = self._estimate_f_q_joint(y_hist_tiles)

                    if h_freq_csi_perfect_debug is not None:
                        last_hist_tiles = y_hist_tiles[-1]
                        joint_wiener_pred_tiles = last_hist_tiles @ f_hat.T
                        joint_wiener_pred = joint_wiener_pred_tiles.reshape(num_syms, num_sc, RxAnt, TxAnt).transpose(2, 3, 0, 1)
                        curr_perfect_block = h_freq_csi_perfect_debug[ :, :, rx_ant_idx,  ...]
                        curr_perfect_block = np.squeeze(curr_perfect_block[:, :, :, :, tx_ant_idx, ...])
                        numer = np.linalg.norm(joint_wiener_pred - curr_perfect_block) ** 2
                        denom = np.linalg.norm(curr_perfect_block) ** 2 + self.eps
                        last_joint_f_nmse = float(np.real(numer / denom))
                        print("Weiner Filter NMSE: ", last_joint_f_nmse)

                    y_next_tiles = np.zeros((num_syms * num_sc, RxAnt * TxAnt), dtype=np.complex128)
                    
                    for tile_idx in range(num_syms * num_sc):
                        y_hist = y_hist_tiles[:, tile_idx, :]
                        r_diag = e_tiles[:, tile_idx, :].mean(axis=0)
                        y_next_tiles[tile_idx] = self._kalman_predict_one_step(
                            y_hist, r_diag, f_hat=f_hat, q_hat=q_hat
                        )
                    
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    y_next_block = y_next_tiles.reshape(num_syms, num_sc, RxAnt, TxAnt).transpose(2, 3, 0, 1)
                    pred[batch_idx, 0, rx_idx, 0, tx_idx, :, :] = y_next_block

        return pred.astype(h_hist.dtype, copy=False)
