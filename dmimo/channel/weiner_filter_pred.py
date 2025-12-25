import numpy as np


class weiner_filter_pred:

    def __init__(self, method="using_one_link", type=np.complex64):
        
        self.dtype = type

        self.method = method # "using_one_link", "using_one_link_windowed", "using_one_link_MIMO", "using_all_links_MIMO"

    def predict(self, h_freq_csi_history, K=None, C=None):
        

        # Helper to get antenna index ranges
        def ant_range(node_idx, bs_count, ue_count):
            if node_idx == 0:
                return np.arange(0, bs_count)
            start = bs_count + (node_idx - 1) * ue_count
            return np.arange(start, start + ue_count)
            
        if "using_one_link" in self.method:
            
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            num_bs_ant = 4
            num_ue_ant = 2
            num_rx_nodes = int((RxAnt - num_bs_ant) / num_ue_ant)+1
            num_tx_nodes = int((TxAnt - num_bs_ant) / num_ue_ant)+1

            h_freq_csi_predicted = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            
            for tx_node_idx in range(num_tx_nodes):
                tx_ant_idx = ant_range(tx_node_idx, num_bs_ant, num_ue_ant)
                for rx_node_idx in range(num_rx_nodes):
                    rx_ant_idx = ant_range(rx_node_idx, num_bs_ant, num_ue_ant)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]
                    
                    if self.method == "using_one_link":
                        tmp = self.predict_using_single_link_vectorized(np.squeeze(curr_h_freq_csi_history))
                    elif self.method == "using_one_link_windowed":
                        if K is None:
                            K = curr_h_freq_csi_history.shape[0]-1
                        tmp = self.predict_next_using_single_link_rolling(np.squeeze(curr_h_freq_csi_history), K=K)
                    elif self.method == "using_one_link_MIMO":
                        tmp = self.predict_using_single_link_MIMO_ARp(np.squeeze(curr_h_freq_csi_history), K=K)
                    tmp = tmp[np.newaxis, np.newaxis, :, np.newaxis, ...]
                    
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    h_freq_csi_predicted[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)
        
        elif self.method == "using_all_links_MIMO":
            # expects a dict {(r,t): H_rt} with heterogeneous sizes.

            link_hist_dict = {}

            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            num_bs_ant = 4
            num_ue_ant = 2
            num_rx_nodes = int((RxAnt - num_bs_ant) / num_ue_ant)+1
            num_tx_nodes = int((TxAnt - num_bs_ant) / num_ue_ant)+1

            for r_node in range(num_rx_nodes):
                rx_idx = ant_range(r_node, num_bs_ant, num_ue_ant)
                Nr = len(rx_idx)

                for t_node in range(num_tx_nodes):
                    tx_idx = ant_range(t_node, num_bs_ant, num_ue_ant)
                    Nt = len(tx_idx)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_idx,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:, 0, 0, :, 0, :, :, :]

                    link_hist_dict[(r_node, t_node)] = curr_h_freq_csi_history.astype(np.complex128, copy=False)

            h_freq_csi_predicted_dict, model_dict = self.predict_using_all_links_MIMO_ARp_v2(link_hist_dict, K=K, lam=1e-2,
                                                           standardize=False, add_bias=False, solve="auto", C=C)
            h_freq_csi_predicted_dict, model_dict = self.predict_using_all_links_MIMO_ARp_v2_ALS(link_hist_dict, K=K, lam=1e-2, C=C)
            
            h_freq_csi_predicted = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            
            for (r_node, t_node), H_pred_rt in h_freq_csi_predicted_dict.items():
                rx_ant_idx = ant_range(r_node, num_bs_ant, num_ue_ant)
                tx_ant_idx = ant_range(t_node, num_bs_ant, num_ue_ant)

                tmp = H_pred_rt[np.newaxis, np.newaxis, :, np.newaxis, :, :, :]

                rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                h_freq_csi_predicted[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

        else:
            raise Exception("Incompatible method")
        
        return h_freq_csi_predicted


    def predict_using_single_link_vectorized(self, h_freq_csi_history, lam=1e-4):
        """
        Predict the last subframe of a single link using ONLY its own prior subframes.

        Args:
            h_freq_csi_history: np.ndarray, shape [S, N_r, N_t, Nsym, Nsc], complex.
            lam: float, ridge regularization (>=0).

        Returns:
            h_freq_csi_predicted: np.ndarray, shape [N_r, N_t, Nsym, Nsc], complex,
                the prediction for the last subframe from all previous subframes.
        """
        H = h_freq_csi_history
        S, N_r, N_t, Nsym, Nsc = H.shape
        assert S >= 2, "Need at least 2 subframes to leave the last one as target."

        # Flatten per subframe to [B], with B = N_r*N_t*Nsym*Nsc
        B = N_r * N_t * Nsym * Nsc
        # Solve in complex128 for numerical stability, cast back to self.dtype at the end
        H_flat = H.reshape(S, B).astype(np.complex128, copy=False)

        # Target: last subframe of the link
        y = H_flat[-1, :]  # [B]

        # Features: all previous subframes of the same link
        F = S - 1
        Phi = H_flat[:-1, :].T  # [B, F]

        # Complex ridge Wiener solve: w = (Phi^H Phi + lam I)^-1 Phi^H y
        A = Phi.conj().T @ Phi            # [F, F]
        b = Phi.conj().T @ y              # [F]
        A_reg = A + lam * np.eye(F, dtype=np.complex128)
        w = np.linalg.solve(A_reg, b)     # [F]

        # Predict and reshape back to tensor
        Phi = H_flat[1:, :].T  # [B, F]
        y_hat = Phi @ w                   # [B]
        h_freq_csi_predicted = y_hat.reshape(N_r, N_t, Nsym, Nsc).astype(self.dtype, copy=False)

        return h_freq_csi_predicted
    
    def predict_next_using_single_link_rolling(self, H, K=4, lam=1e-2):
        """
        One-step-ahead linear predictor using K lags, trained in the *dual*.
        H: [S, N_r, N_t, Nsym, Nsc] complex
        Returns: prediction for frame S (next after S-1)
        """
        H = H.astype(np.complex128, copy=False)
        S, N_r, N_t, Nsym, Nsc = H.shape
        assert S >= K + 1, "Need at least K+1 frames"
        B = N_r * N_t * Nsym * Nsc
        X = H.reshape(S, B)  # [S, B]

        # Build rolling windows: for t=K..S-2, input = [X[t],...,X[t-K+1]], target = X[t+1]
        idxs = list(range(K, S-1))
        Ttr  = len(idxs)

        # Build Phi (Ttr x K*B) and Y (Ttr x B)
        # (This is ~ Ttr*K*B elements; typically << (K*B)^2.)
        Phi = np.empty((Ttr, K*B), dtype=np.complex128)
        Y   = np.empty((Ttr, B),    dtype=np.complex128)
        for i, t in enumerate(idxs):
            Phi[i, :] = np.concatenate([X[t - k, :] for k in range(K)], axis=0)
            Y[i, :]   = X[t + 1, :]

        # Dual matrices
        # A = Phi Phi^H  (Ttr x Ttr), Hermitian PSD
        A = Phi @ Phi.conj().T
        A += lam * np.eye(Ttr, dtype=np.complex128)  # ridge

        # Solve A * Alpha = Y  (multiple RHS). Use Cholesky for stability.
        try:
            L = np.linalg.cholesky(A)                      # A = L L^H
            # Solve L Z = Y
            Z = np.linalg.solve(L, Y)
            # Solve L^H Alpha = Z
            Alpha = np.linalg.solve(L.conj().T, Z)         # (Ttr x B)
        except np.linalg.LinAlgError:
            # Fallback to generic solve if Cholesky fails (heavy regularization should prevent this)
            Alpha = np.linalg.solve(A, Y)

        # Form last K-lag feature (phi_last) and its dual "kernel" k = Phi * phi_last^H
        phi_last = np.concatenate([X[S - 1 - k, :] for k in range(K)], axis=0)  # (K*B,)
        # k_j = <Phi_j, phi_last> for j=1..Ttr
        k = Phi @ phi_last.conj()  # (Ttr,)

        # Prediction: y_hat = k^H * Alpha   (B,)
        y_hat = k.conj().T @ Alpha
        return y_hat.reshape(N_r, N_t, Nsym, Nsc).astype(self.dtype, copy=False)


    def predict_using_single_link_MIMO_ARp(self, h_freq_csi_history, K=3, lam=1e-2,
                                       standardize=False, add_bias=False):
        """
        Vectorized MIMO AR(p): vec(H_t) = [W1 ... WK] [vec(H_{t-1}); ... ; vec(H_{t-K})] + e_t
        Fit with ridge using ALL (sym, sc) as independent samples, then predict frame S.

        Args:
            h_freq_csi_history: complex array [S, Nr, Nt, Nsym, Nsc]
            K: number of lags (p)
            lam: ridge regularization
            standardize: z-score features across samples (and center outputs)
            add_bias: append a bias column

        Returns:
            pred: [Nr, Nt, Nsym, Nsc] complex
        """
        H = h_freq_csi_history.astype(np.complex128, copy=False)
        S, Nr, Nt, Nsym, Nsc = H.shape
        assert S >= K + 1, "Need at least K+1 frames."
        D = Nr * Nt
        N = (S - K) * Nsym * Nsc  # samples

        # Build design Phi: (N x (K*D [+1 if bias])), and targets Y: (N x D)
        Pdim = K * D + (1 if add_bias else 0)
        Phi = np.empty((N, Pdim), dtype=np.complex128)
        Y   = np.empty((N, D),    dtype=np.complex128)

        idx = 0
        for t in range(K, S):
            Hstack = []
            for k in range(1, K+1):
                Hstack.append(H[t-k].reshape(D, Nsym*Nsc))   # (D, Nsym*Nsc)
            XprevK = np.concatenate(Hstack, axis=0)          # (K*D, Nsym*Nsc)
            Xnow   = H[t].reshape(D, Nsym*Nsc)               # (D, Nsym*Nsc)
            # Fill rows for all (sym, sc)
            for s in range(Nsym*Nsc):
                row = XprevK[:, s]
                if add_bias:
                    Phi[idx, :-1] = row
                    Phi[idx,  -1] = 1.0
                else:
                    Phi[idx, :]   = row
                Y[idx, :] = Xnow[:, s]
                idx += 1

        # Optional standardization (across samples)
        if standardize:
            mu_phi = Phi.mean(axis=0, keepdims=True)
            sd_phi = Phi.std(axis=0, keepdims=True) + 1e-12
            Phi_n = (Phi - mu_phi) / sd_phi
            mu_y  = Y.mean(axis=0)
            Y_n   = Y - mu_y
        else:
            Phi_n, Y_n = Phi, Y

        # Primal ridge: Wbig has shape ((K*D [+1]), D)
        G = Phi_n.conj().T @ Phi_n
        R = Phi_n.conj().T @ Y_n
        A = G + lam * np.eye(Pdim, dtype=np.complex128)
        try:
            L = np.linalg.cholesky(A)
            Z = np.linalg.solve(L, R)
            Wbig = np.linalg.solve(L.conj().T, Z)
        except np.linalg.LinAlgError:
            Wbig = np.linalg.solve(A, R)

        # Predict frame S from last K frames for each (sym, sc)
        pred = np.empty((Nr, Nt, Nsym, Nsc), dtype=np.complex128)
        # Build last K stack once
        Hstack_last = []
        for k in range(1, K+1):
            Hstack_last.append(H[S-k].reshape(D, Nsym*Nsc))
        XlastK = np.concatenate(Hstack_last, axis=0)  # (K*D, Nsym*Nsc)

        for s in range(Nsym*Nsc):
            x = XlastK[:, s]
            if add_bias:
                x = np.concatenate([x, np.array([1.0+0j])])
            if standardize:
                xz = (x - mu_phi.ravel()) / sd_phi.ravel()
                y_hat = xz @ Wbig + mu_y
            else:
                y_hat = x @ Wbig
            y_mat = y_hat.reshape(Nr, Nt)
            # map s back to (sym, sc)
            sym = s // Nsc
            sc  = s %  Nsc
            pred[:, :, sym, sc] = y_mat

        return pred.astype(self.dtype, copy=False)
    

    
    def predict_using_all_links_MIMO_ARp_v2(self, link_hist_dict, K=3, lam=1e-2,
                                        standardize=False, add_bias=False, solve="auto",
                                        C=None):
        """
        Multi-link AR(p) Wiener predictor with CAUSAL neighbor scoring.

        Causal neighbor scoring:
        For a target link l and each candidate neighbor lp != l, we build a
        design matrix Phi_lp(t) = [ vec(V_lp[t-1]); ...; vec(V_lp[t-K]) ] and fit a
        ridge regression Y_l(t) ~ Phi_lp(t). We measure the predictive R^2 over
        the training window t = K..S-1 across all tiles, using only past samples.
        The top-C links by R^2 are selected as neighbors for the final joint fit.

        Arguments
        ---------
        link_hist_dict : dict
            {(r_node, t_node): H} with H shape [S, Nr, Nt, Nsym, NRB] (complex).
        K : int
            Number of lags (AR order).
        lam : float
            Ridge regularization for the FINAL joint fit (and also used for
            scoring fits unless stated otherwise).
        standardize : bool
            If True, z-score features; center targets and add back the mean at the end.
        add_bias : bool
            If True, append a complex bias 1+0j to features.
        solve : {"auto","primal","dual"}
            Linear-solve regime. "auto" selects based on shape.
        C : int or None
            Number of cross-links to include (besides the self-link). If None,
            include all other links.

        Returns
        -------
        pred_dict : dict
            {(r_node, t_node): pred} where pred has shape [Nr, Nt, Nsym, NRB].
        model_dict : dict
            Diagnostics: chosen neighbors per link, feature order, shapes, params.
        """
        # ---------- Input checks & reshape ----------
        assert isinstance(link_hist_dict, dict) and len(link_hist_dict) >= 1
        links = sorted(link_hist_dict.keys())
        # Validate shapes and get common S, Nsym, NRB
        sample_link = links[0]
        S, Nr0, Nt0, Nsym, NRB = link_hist_dict[sample_link].shape
        # Basic consistency
        for l, Hi in link_hist_dict.items():
            assert Hi.ndim == 5, f"Each H must be [S,Nr,Nt,Nsym,NRB], got {Hi.shape} for {l}"
            assert Hi.shape[0] >= K + 1, f"Need at least K+1 frames, got {Hi.shape[0]} for {l}"
            assert Hi.shape[3] == Nsym and Hi.shape[4] == NRB, \
                f"Inconsistent S/Nsym/NRB for link {l}: got {Hi.shape} vs {(S,'...', '...', Nsym, NRB)}."

        # Precompute per-link flattened dims d_l = Nr*Nt and vectorized histories V_l[t] -> (d_l, Ntiles)
        V_by_link = {}
        d_by_link = {}
        for l in links:
            H = link_hist_dict[l].astype(np.complex128, copy=False)  # [S, Nr, Nt, Nsym, NRB]
            S_, Nr, Nt, Nsym_, NRB_ = H.shape
            d = Nr * Nt
            d_by_link[l] = (d, Nr, Nt)
            V_by_link[l] = np.ascontiguousarray(H.reshape(S_, d, Nsym_ * NRB_))  # [S, d, Ntiles]

        Ntiles = Nsym * NRB
        N_time = S - K                 # training times t = K..S-1
        N = N_time * Ntiles            # rows in design/target

        pred_dict = {}
        model_dict = {
            'feature_indexing_order': {},
            'theta_shapes': {},
            'neighbors': {},
            'params': {'K': K, 'lam': lam, 'standardize': standardize,
                    'add_bias': add_bias, 'solve': solve, 'C': C},
        }

        # ---------- helper: ridge solve (primal/dual) ----------
        def ridge_fit(Phi, Y, lam_, standardize_, add_bias_):
            """
            Returns W, and (mu_phi, sd_phi, mu_y) when standardize_=True; else None.
            Shapes:
            Phi: (N, P) complex, Y: (N, D) complex, W: (P, D)
            """
            if standardize_:
                mu_phi = Phi.mean(axis=0).astype(np.complex128, copy=False)
                sd_phi = (Phi.std(axis=0) + 1e-12).astype(np.complex128, copy=False)
                if add_bias_:
                    mu_phi[-1] = 0.0
                    sd_phi[-1] = 1.0
                Phi_n = (Phi - mu_phi[None, :]) / sd_phi[None, :]

                mu_y = Y.mean(axis=0).astype(np.complex128, copy=False)
                Y_n = Y - mu_y[None, :]

                Nrows, Ncols = Phi_n.shape
                use_dual = (solve == "dual") or (solve == "auto" and Ncols > Nrows)
                if not use_dual:
                    G = Phi_n.conj().T @ Phi_n
                    R = Phi_n.conj().T @ Y_n
                    A = G + lam_ * np.eye(Ncols, dtype=np.complex128)
                    try:
                        L = np.linalg.cholesky(A)
                        Z = np.linalg.solve(L, R)
                        W = np.linalg.solve(L.conj().T, Z)
                    except np.linalg.LinAlgError:
                        W = np.linalg.solve(A, R)
                else:
                    Kmat = Phi_n @ Phi_n.conj().T
                    B = Y_n
                    A = Kmat + lam_ * np.eye(Nrows, dtype=np.complex128)
                    try:
                        L = np.linalg.cholesky(A)
                        Z = np.linalg.solve(L, B)
                        Alpha = np.linalg.solve(L.conj().T, Z)
                    except np.linalg.LinAlgError:
                        Alpha = np.linalg.solve(A, B)
                    W = Phi_n.conj().T @ Alpha
                return W, (mu_phi, sd_phi, mu_y)
            else:
                Nrows, Ncols = Phi.shape
                use_dual = (solve == "dual") or (solve == "auto" and Ncols > Nrows)
                if not use_dual:
                    G = Phi.conj().T @ Phi
                    R = Phi.conj().T @ Y
                    A = G + lam_ * np.eye(Ncols, dtype=np.complex128)
                    try:
                        L = np.linalg.cholesky(A)
                        Z = np.linalg.solve(L, R)
                        W = np.linalg.solve(L.conj().T, Z)
                    except np.linalg.LinAlgError:
                        W = np.linalg.solve(A, R)
                else:
                    Kmat = Phi @ Phi.conj().T
                    B = Y
                    A = Kmat + lam_ * np.eye(Nrows, dtype=np.complex128)
                    try:
                        L = np.linalg.cholesky(A)
                        Z = np.linalg.solve(L, B)
                        Alpha = np.linalg.solve(L.conj().T, Z)
                    except np.linalg.LinAlgError:
                        Alpha = np.linalg.solve(A, B)
                    W = Phi.conj().T @ Alpha
                return W, None

        # ---------- CAUSAL neighbor scoring via predictive R^2 ----------
        def causal_score_neighbors_for_target(l, lam_score=None):
            """
            Returns a list of (lp, score) for lp != l, sorted by score desc.

            For each candidate neighbor lp, we build Phi_lp from ONLY lp’s past K lags
            and fit Y_l ~ Phi_lp with ridge (lam_score). Score = predictive R^2 over
            the training window (t = K..S-1) and all tiles.
            """
            if lam_score is None:
                lam_score = lam  # default: reuse final ridge λ

            d_l, _, _ = d_by_link[l]
            # Build target matrix Y_l once: stack across times and tiles
            Y = np.empty((N, d_l), dtype=np.complex128)
            row0 = 0
            for t in range(K, S):
                rows = slice(row0, row0 + Ntiles)
                Y[rows, :] = V_by_link[l][t].T  # (Ntiles, d_l)
                row0 += Ntiles

            # Precompute centered target (for R^2 denominator) independent of lp
            Y_mean = Y.mean(axis=0)
            Yc = Y - Y_mean[None, :]
            denom = float(np.linalg.norm(Yc, 'fro')**2) + 1e-24  # avoid /0

            scores = []
            for lp in links:
                if lp == l:
                    continue
                d_lp, _, _ = d_by_link[lp]
                base_P = K * d_lp
                Pdim = base_P + (1 if add_bias else 0)

                # Build Phi_lp (ONLY this neighbor’s K lags)
                Phi = np.empty((N, Pdim), dtype=np.complex128)
                row0 = 0
                # use a stack buffer for speed
                Xprev_buf = np.empty((base_P, Ntiles), dtype=np.complex128)
                for t in range(K, S):
                    rstart = 0
                    for k in range(1, K + 1):
                        tt = t - k
                        Xprev_buf[rstart:rstart + d_lp, :] = V_by_link[lp][tt]
                        rstart += d_lp
                    rows = slice(row0, row0 + Ntiles)
                    if add_bias:
                        Phi[rows, :-1] = Xprev_buf.T
                        Phi[rows, -1]  = 1.0 + 0j
                    else:
                        Phi[rows, :]   = Xprev_buf.T
                    row0 += Ntiles

                # Fit ridge and predict Y_hat
                W_lp, norm_stats = ridge_fit(Phi, Y, lam_score, standardize, add_bias)
                if standardize and norm_stats is not None:
                    mu_phi, sd_phi, mu_y = norm_stats
                    Phi_n = (Phi - mu_phi[None, :]) / sd_phi[None, :]
                    Y_hat = Phi_n @ W_lp + mu_y[None, :]
                else:
                    Y_hat = Phi @ W_lp

                # Compute R^2 causally (one-step ahead across t)
                resid = Y - Y_hat
                num = float(np.linalg.norm(resid, 'fro')**2)
                R2 = max(0.0, 1.0 - num / denom)  # clip to [0,1] for stability
                scores.append((lp, R2))

            # Sort by predictive power (descending), tie-break by link id
            scores = sorted(scores, key=lambda kv: (-kv[1], kv[0]))
            # print("\nscores = ", scores)
            return scores

        # ---------- Train per target link with top-C neighbors ----------
        for l in links:
            d_l, Nr_l, Nt_l = d_by_link[l]

            # Choose neighbors by CAUSAL predictive score
            if C is None or C >= (len(links) - 1):
                ranked = causal_score_neighbors_for_target(l)
                cross_links = [lp for (lp, _) in ranked]
            else:
                ranked = causal_score_neighbors_for_target(l)
                cross_links = [lp for (lp, _) in ranked[:min(C, len(ranked))]]

            neighbor_list = [l] + cross_links
            model_dict['neighbors'][l] = neighbor_list.copy()

            # Feature stacking order (for interpretability/debug)
            feature_order = []
            for k in range(1, K + 1):
                for lp in neighbor_list:
                    feature_order.append((k, lp))
            model_dict['feature_indexing_order'][l] = feature_order

            sum_d_neighbors = sum(d_by_link[lp][0] for lp in neighbor_list)
            base_P = K * sum_d_neighbors
            Pdim = base_P + (1 if add_bias else 0)

            # Build design Phi and targets Y jointly (fast, contiguous writes)
            Phi = np.empty((N, Pdim), dtype=np.complex128)
            Y   = np.empty((N, d_l), dtype=np.complex128)
            Xprev_buf = np.empty((base_P, Ntiles), dtype=np.complex128)

            row0 = 0
            for t in range(K, S):
                # fill stacked features for all selected neighbors and lags
                rstart = 0
                for k in range(1, K + 1):
                    tt = t - k
                    for lp in neighbor_list:
                        d_lp = d_by_link[lp][0]
                        Xprev_buf[rstart:rstart + d_lp, :] = V_by_link[lp][tt]
                        rstart += d_lp

                # targets at time t
                V_tgt = V_by_link[l][t]  # (d_l, Ntiles)

                rows = slice(row0, row0 + Ntiles)
                if add_bias:
                    Phi[rows, :-1] = Xprev_buf.T
                    Phi[rows, -1]  = 1.0 + 0j
                else:
                    Phi[rows, :]   = Xprev_buf.T
                Y[rows, :] = V_tgt.T
                row0 += Ntiles

            # Fit final joint ridge
            W, norm_stats = ridge_fit(Phi, Y, lam, standardize, add_bias)

            # Predict the next frame t = S (using last K frames)
            rstart = 0
            Xlast = np.empty((base_P,), dtype=np.complex128)
            # Build per-tile last-K features as (Ntiles, Pdim) without loops when possible
            # Fill a (base_P, Ntiles) buffer first
            Xprev_buf[:, :] = 0
            rstart = 0
            for k in range(1, K + 1):
                tt = S - k
                for lp in neighbor_list:
                    d_lp = d_by_link[lp][0]
                    Xprev_buf[rstart:rstart + d_lp, :] = V_by_link[lp][tt]
                    rstart += d_lp

            if add_bias:
                Xlast_full_T = np.empty((Ntiles, Pdim), dtype=np.complex128)
                Xlast_full_T[:, :-1] = Xprev_buf.T
                Xlast_full_T[:, -1]  = 1.0 + 0j
                if standardize and norm_stats is not None:
                    mu_phi, sd_phi, mu_y = norm_stats
                    Xlast_norm = (Xlast_full_T - mu_phi[None, :]) / sd_phi[None, :]
                    Yhat = Xlast_norm @ W + mu_y[None, :]
                else:
                    Yhat = Xlast_full_T @ W
            else:
                Xlast_T = Xprev_buf.T  # (Ntiles, base_P)
                if standardize and norm_stats is not None:
                    mu_phi, sd_phi, mu_y = norm_stats
                    # build a temp (Ntiles, Pdim) only when needed
                    Xlast_norm = (Xlast_T - mu_phi[None, :]) / sd_phi[None, :]
                    Yhat = Xlast_norm @ W + mu_y[None, :]
                else:
                    Yhat = Xlast_T @ W

            # Reshape back to [Nr, Nt, Nsym, NRB]
            assert Yhat.shape == (Ntiles, d_l)
            yhat_mat = Yhat.T.reshape(Nr_l, Nt_l, Nsym, NRB)
            pred_dict[l] = yhat_mat.astype(self.dtype, copy=False)

            # (Optional) record block shapes for interpretability
            block_shapes = []
            for k in range(1, K + 1):
                for lp in neighbor_list:
                    d_lp = d_by_link[lp][0]
                    block_shapes.append((d_l, d_lp))
            model_dict['theta_shapes'][l] = block_shapes

        return pred_dict, model_dict


    def predict_using_all_links_MIMO_ARp_v2_ALS(self, link_hist_dict, K=3, lam=1e-2,
                                            C=None, max_iters=20, tol=1e-6,
                                            init="identity"):
        """
        Multi-link AR(p) predictor with CAUSAL neighbor scoring (R^2 per candidate),
        and final ALS fit of two-sided matrices per lag and per selected neighbor.
        Heterogeneous link sizes are allowed; for each target link l, ALS uses only
        neighbors with the same (Nr,Nt) as l.

        Args:
        link_hist_dict: {(r_node,t_node): H}, H shape [S, Nr, Nt, Nsym, NRB] (complex), Nr,Nt may differ by link
        K: lags (AR order), requires S >= K+1
        lam: ridge regularization for ALS normal equations
        C: keep top-C neighbors (besides self) by causal R^2; if None, keep all same-dim neighbors
        max_iters, tol, init: ALS controls; init ∈ {"identity","zeros"}

        Returns:
        pred_dict: {(r_node,t_node): H_pred_S  with shape [Nr,Nt,Nsym,NRB]}
        model_dict: diagnostics incl. neighbors, excluded_due_to_dim, learned W_left/W_right
        """
        import numpy as np

        # ---------- Input checks (allow heterogeneous Nr,Nt) ----------
        assert isinstance(link_hist_dict, dict) and len(link_hist_dict) >= 1
        links = sorted(link_hist_dict.keys())

        # Record per-link shapes
        shape_by_link = {l: link_hist_dict[l].shape for l in links}
        S = shape_by_link[links[0]][0]
        Nsym = shape_by_link[links[0]][3]
        NRB  = shape_by_link[links[0]][4]
        for l, shp in shape_by_link.items():
            assert shp[0] == S and shp[3] == Nsym and shp[4] == NRB, \
                f"Inconsistent (S,Nsym,NRB) for link {l}: {shp} vs {(S,'...',Nsym,NRB)}"
        assert S >= K + 1, "Need at least K+1 frames"

        Ntiles = Nsym * NRB
        pred_dict = {}
        model_dict = {
            'neighbors': {},
            'excluded_due_to_dim': {},
            'params': {'K': K, 'lam': lam, 'C': C, 'max_iters': max_iters, 'tol': tol, 'init': init},
            'per_link': {}
        }

        # ---------- Precompute vectorized views for scoring (hetero d_l) ----------
        V_by_link = {}
        d_by_link = {}
        NrNt_by_link = {}
        for l in links:
            H = link_hist_dict[l].astype(np.complex128, copy=False)  # [S,Nr,Nt,Nsym,NRB]
            _, Nr_l, Nt_l, _, _ = H.shape
            d_l = Nr_l * Nt_l
            d_by_link[l] = d_l
            NrNt_by_link[l] = (Nr_l, Nt_l)
            V_by_link[l] = np.ascontiguousarray(H.reshape(S, d_l, Ntiles))  # [S, d_l, Ntiles]

        N_time = S - K
        N_rows = N_time * Ntiles

        # ---------- Causal single-neighbor R^2 scoring (hetero dims aware) ----------
        def causal_scores_for_target(l):
            d_l = d_by_link[l]
            # Build target Y (stack times/tiles)
            Y = np.empty((N_rows, d_l), dtype=np.complex128)
            row0 = 0
            for t in range(K, S):
                rows = slice(row0, row0 + Ntiles)
                Y[rows, :] = V_by_link[l][t].T
                row0 += Ntiles
            Yc = Y - Y.mean(axis=0, keepdims=True)
            denom = float(np.linalg.norm(Yc, 'fro')**2) + 1e-24

            scores = []
            for lp in links:
                if lp == l:
                    continue
                d_lp = d_by_link[lp]
                # Design Phi from ONLY lp's past K lags (hetero width)
                Phi = np.empty((N_rows, K * d_lp), dtype=np.complex128)
                row0 = 0
                Xbuf = np.empty((K * d_lp, Ntiles), dtype=np.complex128)
                for t in range(K, S):
                    rstart = 0
                    for k in range(1, K + 1):
                        Xbuf[rstart:rstart + d_lp, :] = V_by_link[lp][t - k]
                        rstart += d_lp
                    rows = slice(row0, row0 + Ntiles)
                    Phi[rows, :] = Xbuf.T
                    row0 += Ntiles

                # ridge fit (complex)
                G = Phi.conj().T @ Phi                          # (K*d_lp, K*d_lp)
                R = Phi.conj().T @ Y                            # (K*d_lp, d_l)
                A = G + lam * np.eye(G.shape[0], dtype=np.complex128)
                try:
                    L = np.linalg.cholesky(A)
                    Z = np.linalg.solve(L, R)
                    W = np.linalg.solve(L.conj().T, Z)
                except np.linalg.LinAlgError:
                    W = np.linalg.solve(A, R)

                Yhat = Phi @ W                                   # (N_rows, d_l)
                resid = Y - Yhat
                num = float(np.linalg.norm(resid, 'fro')**2)
                R2 = max(0.0, 1.0 - num / denom)
                scores.append((lp, R2))

            scores.sort(key=lambda kv: (-kv[1], kv[0]))
            return scores

        # ---------- ALS per target using selected neighbors (same (Nr,Nt) only) ----------
        for l in links:
            Nr_l, Nt_l = NrNt_by_link[l]
            ranked = causal_scores_for_target(l)

            # Filter by same matrix dims as target
            same_dim_candidates = [lp for (lp, _) in ranked if NrNt_by_link[lp] == (Nr_l, Nt_l)]
            excluded = [lp for (lp, _) in ranked if NrNt_by_link[lp] != (Nr_l, Nt_l)]
            model_dict['excluded_due_to_dim'][l] = excluded

            if C is None or C >= len(same_dim_candidates):
                cross = same_dim_candidates
            else:
                cross = same_dim_candidates[:max(0, C)]

            neighbor_list = [l] + cross
            model_dict['neighbors'][l] = neighbor_list.copy()

            # Cast all needed tensors once
            H_by_lp = {lp: link_hist_dict[lp].astype(np.complex128, copy=False) for lp in neighbor_list}

            # Initialize W_left/W_right per (tau, lp)
            if init == "identity":
                Wl = {(tau, lp): np.eye(Nr_l, dtype=np.complex128) for tau in range(1, K+1) for lp in neighbor_list}
                Wr = {(tau, lp): np.eye(Nt_l, dtype=np.complex128) for tau in range(1, K+1) for lp in neighbor_list}
            else:
                Wl = {(tau, lp): np.zeros((Nr_l, Nr_l), dtype=np.complex128) for tau in range(1, K+1) for lp in neighbor_list}
                Wr = {(tau, lp): np.zeros((Nt_l, Nt_l), dtype=np.complex128) for tau in range(1, K+1) for lp in neighbor_list}

            last_obj = np.inf
            for it in range(max_iters):
                # ---- LEFT updates (fix all Wr) ----
                for tau in range(1, K+1):
                    for lp in neighbor_list:
                        G = np.zeros((Nr_l, Nr_l), dtype=np.complex128)  # sum X X^H
                        Rm = np.zeros((Nr_l, Nr_l), dtype=np.complex128) # sum Resid X^H
                        for t in range(K, S):
                            for n in range(Nsym):
                                for m in range(NRB):
                                    H_next = H_by_lp[l][t, :, :, n, m]  # target frame at t
                                    # residual excluding current (tau,lp)
                                    Resid = H_next.copy()
                                    for tau2 in range(1, K+1):
                                        for lp2 in neighbor_list:
                                            if tau2 == tau and lp2 == lp:
                                                continue
                                            Resid -= Wl[(tau2, lp2)] @ H_by_lp[lp2][t - tau2, :, :, n, m] @ Wr[(tau2, lp2)]
                                    X = H_by_lp[lp][t - tau, :, :, n, m] @ Wr[(tau, lp)]  # Nr x Nt
                                    G += X @ X.conj().T
                                    Rm += Resid @ X.conj().T
                        A = G + lam * np.eye(Nr_l, dtype=np.complex128)
                        try:
                            L = np.linalg.cholesky(A)
                            Z = np.linalg.solve(L, Rm)
                            Wl[(tau, lp)] = np.linalg.solve(L.conj().T, Z)
                        except np.linalg.LinAlgError:
                            Wl[(tau, lp)] = np.linalg.solve(A, Rm)

                # ---- RIGHT updates (fix all Wl) ----
                for tau in range(1, K+1):
                    for lp in neighbor_list:
                        G = np.zeros((Nt_l, Nt_l), dtype=np.complex128)   # sum Y^H Y
                        Rm = np.zeros((Nt_l, Nt_l), dtype=np.complex128)  # sum Y^H Resid
                        for t in range(K, S):
                            for n in range(Nsym):
                                for m in range(NRB):
                                    H_next = H_by_lp[l][t, :, :, n, m]
                                    Resid = H_next.copy()
                                    for tau2 in range(1, K+1):
                                        for lp2 in neighbor_list:
                                            if tau2 == tau and lp2 == lp:
                                                continue
                                            Resid -= Wl[(tau2, lp2)] @ H_by_lp[lp2][t - tau2, :, :, n, m] @ Wr[(tau2, lp2)]
                                    Y = Wl[(tau, lp)] @ H_by_lp[lp][t - tau, :, :, n, m]      # Nr x Nt
                                    G += Y.conj().T @ Y
                                    Rm += Y.conj().T @ Resid
                        A = G + lam * np.eye(Nt_l, dtype=np.complex128)
                        try:
                            L = np.linalg.cholesky(A)
                            Z = np.linalg.solve(L, Rm)
                            Wr[(tau, lp)] = np.linalg.solve(L.conj().T, Z)
                        except np.linalg.LinAlgError:
                            Wr[(tau, lp)] = np.linalg.solve(A, Rm)

                # ---- objective for early stop ----
                obj = 0.0
                for t in range(K, S):
                    for n in range(Nsym):
                        for m in range(NRB):
                            pred = np.zeros((Nr_l, Nt_l), dtype=np.complex128)
                            for tau in range(1, K+1):
                                for lp in neighbor_list:
                                    pred += Wl[(tau, lp)] @ H_by_lp[lp][t - tau, :, :, n, m] @ Wr[(tau, lp)]
                            err = H_by_lp[l][t, :, :, n, m] - pred
                            obj += float(np.vdot(err, err).real)
                reg = lam * sum(float(np.vdot(Wl[k], Wl[k]).real + np.vdot(Wr[k], Wr[k]).real) for k in Wl.keys())
                obj += reg

                if abs(last_obj - obj) / (last_obj + 1e-12) < tol:
                    break
                last_obj = obj

            # ---- predict frame S using last K frames (neighbors w/ same dims) ----
            Yhat = np.empty((Nr_l, Nt_l, Nsym, NRB), dtype=np.complex128)
            for n in range(Nsym):
                for m in range(NRB):
                    acc = np.zeros((Nr_l, Nt_l), dtype=np.complex128)
                    for tau in range(1, K+1):
                        for lp in neighbor_list:
                            acc += Wl[(tau, lp)] @ H_by_lp[lp][S - tau, :, :, n, m] @ Wr[(tau, lp)]
                    Yhat[:, :, n, m] = acc
            pred_dict[l] = Yhat.astype(self.dtype, copy=False)

            model_dict['per_link'][l] = {
                'neighbors': neighbor_list,
                'W_left': {k: Wl[k].copy() for k in Wl},
                'W_right': {k: Wr[k].copy() for k in Wr},
                'iters': it + 1,
                'final_obj': last_obj,
                'Nr': Nr_l, 'Nt': Nt_l, 'K': K
            }

        return pred_dict, model_dict