# Zero-Forcing (ZF) Precoder for dMIMO channels
import numpy as np
import tensorflow as tf

from sionna.utils import matrix_inv

import cvxpy as cp


def weighted_mean_precoder(x, h, rx_snr_db, num_iterations=3, return_precoding_matrix=False):
    """
    MU-MIMO zero-forcing precoding supporting rank adaptation.

    :param x: data stream symbols
    :param h: channel coefficients
    :param rx_snr_db: receiver snr for all users
    :param return_precoding_matrix: return precoding matrix
    :return: precoded data symbols
    """

    # Input dimensions:
    # x: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
    # h: [num_rx_ues, num_rx_ant_per_ue, num_tx_ant_per_ue]
    num_streams_per_tx = x.shape[-1]
    total_tx_ant = h.shape[-1]
    num_user = h.shape[0]
    num_user_streams = h.shape[-2]
    assert num_user_streams == num_streams_per_tx, "total number of streams must match"

    h = np.conj(h)

    rx_snr_linear = 10**(rx_snr_db / 10)
    n_var = 1 / rx_snr_linear

    w_k_i = np.ones((num_user, num_user_streams))
    best_w_k_i = w_k_i.copy()
    best_SINR_k_i = np.zeros((num_user, num_user_streams))
    starting_SINR = np.zeros((num_user, num_user_streams))

    SINR_k_i = np.zeros((num_user, num_user_streams))

    weight_floor_constant = 0.1
    weight_update_constant = 0.5
    gram_constant = 0.01
    
    if num_iterations > 0:
        for iter in range(num_iterations+1):

            w_k_i_reshaped = tf.reshape(w_k_i, (num_user, num_user_streams, 1))
            w_k_i_reshaped = tf.cast(w_k_i_reshaped, h.dtype)

            # print("\niteration {}".format(iter))

            gram = tf.matmul(h * np.sqrt(w_k_i_reshaped), h * np.sqrt(w_k_i_reshaped), adjoint_a=True)
            gram_normalizer = gram_constant * np.eye(gram.shape[1])
            sum_gram = tf.reduce_sum(gram, axis=0) + tf.cast(gram_normalizer, gram.dtype)
            eigvals, v = tf.linalg.eigh(sum_gram)
            # eigvals = tf.sort(tf.abs(eigvals), direction='DESCENDING')
            v = tf.gather(v, tf.argsort(tf.abs(eigvals), direction='DESCENDING'), axis=1)[:, :num_user_streams]
            # norm = tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=0, keepdims=True))
            # v = v / tf.cast(norm, v.dtype)

            prev_SINR_k_i = SINR_k_i.copy()

            for k in range(num_user):

                h_ue = tf.gather(h, indices=k, axis=0)

                H_k_v = tf.matmul(h_ue, v)

                if num_user_streams > 1:
                    SINR_k_i[k,0] = tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 + n_var[k])
                    SINR_k_i[k,1] = tf.abs(tf.linalg.norm(H_k_v[:, 1]))**2 / (tf.abs(tf.linalg.norm(H_k_v[:, 0]))**2 + n_var[k])
                else:
                    SINR_k_i[k,0] = tf.abs(tf.linalg.norm(H_k_v))**2 / (n_var[k])

            if iter == 0:
                starting_SINR = SINR_k_i.copy()

            if np.min(SINR_k_i) < 0.0001:
                hold = 1
                SINR_k_i[np.argmin(SINR_k_i)] += 0.0001

            # print("all SINRs at step {} of precoding:\n".format(iter), SINR_k_i)
            # print("min SINR at step {} of precoding: ".format(iter), np.min(SINR_k_i))
            # print("max SINR at step {} of precoding: ".format(iter), np.max(SINR_k_i))
            # print("mean SINR at step {} of precoding: ".format(iter), np.mean(SINR_k_i))

            if np.min(SINR_k_i) > np.min(best_SINR_k_i):
                best_w_k_i = w_k_i.copy()
                best_SINR_k_i = SINR_k_i.copy()

            prev_w_k_i = w_k_i.copy()
            w_k_i = 1 / np.sqrt(SINR_k_i)
            # w_k_i = weight_update_constant*w_k_i + (1-weight_update_constant)*prev_w_k_i
            # print("w_k_i before flooring:", w_k_i)
            floor_value = weight_floor_constant * np.mean(w_k_i)
            w_k_i = np.where(w_k_i < floor_value, floor_value, w_k_i)
            # print("w_k_i after flooring:", w_k_i)
            # w_k_i /= np.sum(w_k_i)

            hold = 1

    # if num_iterations > 0:
    #     print("\nmin SINR after precoding: ", np.min(best_SINR_k_i))
    #     print("max SINR after precoding: ", np.max(best_SINR_k_i))
    #     print("mean SINR after precoding: ", np.mean(best_SINR_k_i))
    #     print("all SINRs after precoding: ", best_SINR_k_i)

    #     hold = 1

    w_k_i_reshaped = tf.reshape(best_w_k_i, (num_user, num_user_streams, 1))
    w_k_i_reshaped = tf.cast(w_k_i_reshaped, h.dtype)

    gram = tf.matmul(h * np.sqrt(w_k_i_reshaped), h * np.sqrt(w_k_i_reshaped), adjoint_a=True)
    gram_normalizer = gram_constant * np.eye(gram.shape[1])
    sum_gram = tf.reduce_sum(gram, axis=0) + tf.cast(gram_normalizer, gram.dtype)
    eigvals, v = tf.linalg.eigh(sum_gram)
    # eigvals = tf.sort(tf.abs(eigvals), direction='DESCENDING')
    v = tf.gather(v, tf.argsort(tf.abs(eigvals), direction='DESCENDING'), axis=1)[:, :num_user_streams]
    # norm = tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=0, keepdims=True))
    # v = v / tf.cast(norm, v.dtype)

    # Normalize each column to unit power
    norm = tf.sqrt(tf.reduce_sum(tf.abs(v)**2, axis=-2, keepdims=True))
    g = v/tf.cast(norm, v.dtype)
    g = tf.cast(g, x.dtype)

    # Expand last dim of `x` for precoding
    x_precoded = tf.expand_dims(x, -1)

    # Precode
    x_precoded = tf.squeeze(tf.matmul(g, x_precoded), -1)

    if return_precoding_matrix:
        return x_precoded, g, starting_SINR, best_SINR_k_i
    else:
        return x_precoded, None, starting_SINR, best_SINR_k_i
    
def grad_ascent_precoder(x, h, rx_snr_db, num_iterations=3, eps=1e-6, alpha=0.2, ptx=1.0, return_precoding_matrix=False):
    """"
    Inputs
    ------
    x : np.ndarray
        Data symbols, shape [1, 1, N_ofdm_symbols, N_subcarriers, N_streams]
    h : np.ndarray
        Type-I PMI feedback directions. Supports:
          - wideband CSI: [N_users, N_streams, N_tx]
          - subband CSI: [N_users, N_subcarriers, N_streams, N_tx]
        (for multicast, you typically have N_streams=1)
    rx_snr_db : np.ndarray
        Per-user SNR in dB, shape [N_users]
    num_iterations : int
        Gradient-ascent iterations (paper/demo uses small number)
    eps : float
        Small constant for stability in log / division
    alpha : float
        Step size for gradient ascent
    ptx : float
        Total transmit power constraint: ||P||_F^2 = ptx

    Returns
    -------
    x_precoded : np.ndarray
        Precoded samples, shape [1, 1, N_ofdm_symbols, N_subcarriers, N_tx]
    P : np.ndarray
        Precoder matrix:
          - wideband CSI: [N_tx, N_streams]
          - subband CSI: [N_subcarriers, N_tx, N_streams]
    """

    # ---- Shapes ----
    x = np.asarray(x)
    h = np.asarray(h)
    rx_snr_db = np.asarray(rx_snr_db)

    is_wideband = len(h.shape) == 3
    if is_wideband:
        K, Ns, Nt = h.shape
    else:
        K, N_sc, Ns, Nt = h.shape
    assert x.shape[-1] == Ns, f"x last dim (streams)={x.shape[-1]} must match h streams={Ns}"
    assert rx_snr_db.shape[0] == K, "rx_snr_db must have length N_users"

    # ---- 1) Convert SNR weights ----
    snr_lin = 10.0 ** (rx_snr_db / 10.0)  # [K]

    # ---- 2) Normalize PMI directions per user/stream ----
    v = h.astype(np.complex64)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    v = v / v_norm

    # ---- 3) Initialize P (simple robust choice): weighted average of directions ----
    # P0 for each stream s: sum_k sqrt(snr_k) * v_{k,s}^H? (we need Nt x Ns)
    # v_{k,s} is length-Nt direction; take conjugate so beam points along v
    if is_wideband:
        P = np.zeros((Nt, Ns), dtype=np.complex64)
        for s in range(Ns):
            P[:, s] = np.sum((np.sqrt(snr_lin)[:, None] * np.conj(v[..., s, :])), axis=0)
    else:
        P = np.zeros((N_sc, Nt, Ns), dtype=np.complex64)
        for s in range(Ns):
            P[:, :, s] = np.sum((np.sqrt(snr_lin)[:, None, None] * np.conj(v[:, :, s, :])), axis=0)

    # Normalize to power constraint
    if is_wideband:
        norm = np.linalg.norm(P) + 1e-12
        P = np.sqrt(ptx) * P / norm
    else:
        for sc in range(N_sc):
            norm = np.linalg.norm(P[sc], "fro") + 1e-12
            P[sc] = np.sqrt(ptx) * P[sc] / norm

    # ---- Helper: compute gamma_k(P) = ||h_k P||^2 / sigma^2
    # With our surrogate R_k = snr_lin[k] * v_k v_k^H, gamma_k(P) = P^H R_k P (per stream aggregate)
    # We'll compute per-user scalar: sum over streams of |v_{k,s}^H p_s|^2 * snr_k
    def gamma_all(Pmat):
        # Pmat: [Nt, Ns]
        gam = np.zeros((K,), dtype=np.float32)
        for k in range(K):
            acc = 0.0
            for s in range(Ns):
                vk = v[k, s, :]  # [Nt]
                ps = Pmat[:, s]  # [Nt]
                inner = np.vdot(vk, ps)  # v^H p
                acc += (np.abs(inner) ** 2)
            gam[k] = (snr_lin[k] * acc).astype(np.float32)
        return gam  # [K]

    def gamma_all_subband(Pmat_sc, sc_idx):
        # Pmat_sc: [Nt, Ns]
        gam = np.zeros((K,), dtype=np.float32)

        for k in range(K):
            acc = 0.0
            for s in range(Ns):
                vk = v[k, sc_idx, s, :]  # [Nt]
                ps = Pmat_sc[:, s]       # [Nt]
                inner = np.vdot(vk, ps)  # v^H p
                acc += (np.abs(inner) ** 2)
            gam[k] = (snr_lin[k] * acc).astype(np.float32)
        return gam  # [K]

    # ---- 4) Gradient ascent on f(P) = sum_k log(gamma_k(P)+eps) ----
    if is_wideband:
        for _ in range(num_iterations):
            gam = gamma_all(P)  # [K]

            # grad = sum_k (R_k / (gamma_k+eps)) P
            # with R_k = snr_k * sum_s v_{k,s} v_{k,s}^H (block-diagonal by stream in our implementation)
            grad = np.zeros_like(P)
            for k in range(K):
                denom = float(gam[k] + eps)
                w = snr_lin[k] / denom  # scalar weight
                for s in range(Ns):
                    vk = v[k, s, :]  # [Nt]
                    # (v v^H) p_s = v * (v^H p_s)
                    proj = np.vdot(vk, P[:, s])  # v^H p
                    grad[:, s] += (w * vk * proj).astype(np.complex64)

            # ascent step + projection to power constraint
            P_tilde = P + alpha * grad
            fro = np.linalg.norm(P_tilde, "fro") + 1e-12
            P = np.sqrt(ptx) * P_tilde / fro
    else:
        for _ in range(num_iterations):
            for sc in range(N_sc):
                P_sc = P[sc]  # [Nt, Ns]
                gam = gamma_all_subband(P_sc, sc)  # [K]

                grad = np.zeros_like(P_sc)
                for k in range(K):
                    denom = float(gam[k] + eps)
                    w = snr_lin[k] / denom  # scalar weight
                    for s in range(Ns):
                        vk = v[k, sc, s, :]  # [Nt]
                        proj = np.vdot(vk, P_sc[:, s])  # v^H p
                        grad[:, s] += (w * vk * proj).astype(np.complex64)

                # ascent step + projection to per-subcarrier power constraint
                P_tilde = P_sc + alpha * grad
                fro = np.linalg.norm(P_tilde, "fro") + 1e-12
                P[sc] = np.sqrt(ptx) * P_tilde / fro

    # ---- 5) Apply precoder: [.., Ns] x [Ns, Nt] -> [.., Nt]
    if is_wideband:
        # x: [1,1,Nos,Nsc,Ns]
        x_precoded = np.matmul(x, P.T)  # P.T: [Ns, Nt]
    else:
        # x: [1,1,Nos,Nsc,Ns], P: [Nsc,Nt,Ns] -> [1,1,Nos,Nsc,Nt]
        x_precoded = np.matmul(x[..., :, None, :], np.transpose(P, (0, 2, 1))).squeeze(-2)

    return x_precoded, P

def multicast_precoder_np_hard_sdr(
    x,
    h,
    rx_snr_db,
    num_randomizations=200,
    ptx=1.0,
    solver="SCS",
    seed=0,
):
    """
    "Complete NP-hard optimization problem" implementation via:
      - NP-hard max-min formulation (rank-1 beamformer)
      - Semidefinite relaxation (SDR) solved by SDP
      - Gaussian randomization to obtain feasible rank-1 precoder

    Inputs (your shapes)
    --------------------
    x : np.ndarray
        [1, 1, N_ofdm_symbols, N_subcarriers, N_streams]
    h : np.ndarray
        Type-I PMI feedback directions [N_users, N_streams, N_tx]
        For true single-group multicast, N_streams should be 1.
    rx_snr_db : np.ndarray
        [N_users]

    Returns
    -------
    x_precoded : np.ndarray
        [1, 1, N_ofdm_symbols, N_subcarriers, N_tx]
    p_best : np.ndarray
        Beamforming vector [N_tx] (single stream)
    W_sdr : np.ndarray
        SDR solution matrix [N_tx, N_tx]
    t_sdr : float
        SDR objective value (upper bound on true optimum)
    """

    x = np.asarray(x)
    h = np.asarray(h)
    rx_snr_db = np.asarray(rx_snr_db)

    K, Ns, Nt = h.shape
    assert Ns == 1, (
        "This NP-hard multicast formulation is for ONE stream (Ns=1). "
        "If you truly have multiple common streams, the formulation changes."
    )
    assert rx_snr_db.shape == (K,)

    # 1) Build surrogate R_k = snr_k * v_k v_k^H from Type-I PMI + SNR
    snr_lin = 10.0 ** (rx_snr_db / 10.0)
    v = h[:, 0, :].astype(np.complex128)  # [K, Nt]
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    R_list = []
    for k in range(K):
        vk = v[k][:, None]  # [Nt,1]
        Rk = snr_lin[k] * (vk @ vk.conj().T)  # [Nt,Nt]
        R_list.append(Rk)

    # 2) SDR: maximize t s.t. Tr(Rk W) >= t, Tr(W) <= P, W >> 0
    W = cp.Variable((Nt, Nt), complex=True, hermitian=True)
    t = cp.Variable(nonneg=True)
    # t = cp.Variable(real=True)

    constraints = [W >> 0, cp.trace(W) <= ptx]
    for k in range(K):
        constraints += [cp.real(cp.trace(R_list[k] @ W)) >= t]

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=solver, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDR solve failed: status={prob.status}")

    W_sdr = W.value
    t_sdr = float(t.value)

    # 3) Extract a feasible rank-1 p via Gaussian randomization
    #    Sample z ~ CN(0, W_sdr); normalize to power; choose best min_k z^H Rk z
    rng = np.random.default_rng(seed)

    # Numerical cleanup: enforce Hermitian
    W_sdr = 0.5 * (W_sdr + W_sdr.conj().T)

    # Cholesky can fail if near-PSD; use eigendecomp for robustness
    eigvals, eigvecs = np.linalg.eigh(W_sdr)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrtW = eigvecs @ np.diag(np.sqrt(eigvals))

    def min_snr_for_p(p):
        # p: [Nt]
        vals = []
        for k in range(K):
            vals.append(np.real(np.conj(p) @ (R_list[k] @ p)))
        return float(np.min(vals))

    p_best = None
    best_obj = -np.inf

    for _ in range(num_randomizations):
        g = (rng.standard_normal(Nt) + 1j * rng.standard_normal(Nt)) / np.sqrt(2.0)
        z = sqrtW @ g  # ~ CN(0, W_sdr)
        # project to power constraint
        nz = np.linalg.norm(z) + 1e-12
        p = np.sqrt(ptx) * z / nz

        obj = min_snr_for_p(p)
        if obj > best_obj:
            best_obj = obj
            p_best = p

    # Fallback: principal eigenvector if randomization didn't help
    if p_best is None:
        idx = np.argmax(eigvals)
        p_best = np.sqrt(ptx) * eigvecs[:, idx] / (np.linalg.norm(eigvecs[:, idx]) + 1e-12)

    # 4) Apply to x: x[..., 1] times p^T -> [.., Nt]
    #    x is [1,1,Nos,Nsc,1], we multiply last dim by [1,Nt]
    p_row = p_best.reshape(1, Nt)
    x_precoded = np.matmul(x, p_row)  # [1,1,Nos,Nsc,Nt]
    x_precoded = tf.cast(x_precoded, tf.complex64)

    return x_precoded, p_best#, W_sdr, t_sdr


def wmmse_precoder(x,                  # [B, N_t, Ns]  – symbols to precode
                   h,                  # [K, N_r, N_t] – channels
                   rx_snr_db,          # scalar or [K]  – per-UE SNR
                   num_iterations=10,
                   p_max=1.0,          # total power budget
                   return_precoding_matrix=False):  # True → also return V, per-UE rates
    """
    Multicast WMMSE precoding (max-min fairness) for K users, Ns common streams.
    """
    K, N_r, N_t = h.shape
    Ns          = x.shape[-1]
    dtype       = x.dtype
    h = tf.cast(h, x.dtype)

    # noise variance per UE
    n0 = tf.cast(1 / (10.0 ** (rx_snr_db / 10.0)), dtype)      # [K] or scalar

    ## -- initial precoder: eigenmodes of mean Gram -------------------------
    g  = tf.reduce_sum(tf.matmul(h, h, adjoint_a=True), axis=0)
    _, V0 = tf.linalg.eigh(g)
    V     = V0[:, -Ns:]                          # Nt × Ns
    V     = tf.linalg.l2_normalize(V, axis=0)
    V = tf.cast(V, x.dtype)

    for _ in range(num_iterations):
        # ---------- step 1: MMSE receivers U_k ----------------------------
        HV   = tf.matmul(h, V)                   # [K, N_r, N_s]
        I_r   = tf.eye(N_r, dtype=dtype)[None, :, :]
        I_n = n0[:, None, None] * I_r
        inv_term = tf.linalg.inv(tf.matmul(HV, HV, adjoint_b=True) + I_n)
        U = tf.matmul(HV, inv_term, adjoint_a=True) # [K, N_s, N_r]

        # ---------- step 2: error & weights W_k ---------------------------
        UHV = tf.matmul(U, HV) # [K, N_s, N_s]
        I_s   = tf.eye(Ns, dtype=dtype)[None, :, :]
        I_UHV = I_s - UHV
        N_I_UH = tf.matmul(I_n, U, adjoint_b=True)
        E_k = tf.matmul(I_UHV, I_UHV, adjoint_b=True) + tf.matmul(U, N_I_UH)

        # ----------- J(V,U,W) monitor -----------------------------------------
        # trace_k = tr(W_k E_k) for each user
        W_k = tf.linalg.inv(E_k)
        mse_k   = tf.linalg.trace(E_k)        # [K]
        J_value = tf.reduce_sum(mse_k)        # scalar

        tf.print("iter", _, ":  J =", tf.abs(J_value), summarize=-1)

        # ---------- step 3: build A and B ----------------------------------
        UH      = tf.matmul(U, h)
        A_k     = tf.matmul(UH, W_k, adjoint_a=True)
        A_k     = tf.matmul(A_k, UH)
        B_k     = tf.matmul(h, tf.matmul(U, W_k, adjoint_a=True), adjoint_a=True)

        # ---------- step 4: sum over users & solve for V -----------------
        A = tf.reduce_sum(A_k, axis=0)                      # [N_t, N_t]
        B = tf.reduce_sum(B_k, axis=0)                      # [N_t, N_s]

        # regularise A very slightly to avoid numerical singularities
        eps = tf.cast(1e-9, dtype)
        A  += eps * tf.eye(N_t, dtype=dtype)

        V_un  = tf.linalg.solve(A, B)                       # unconstrained V

        # enforce the total-power budget: ‖V‖_F^2 = p_max
        p_now = tf.reduce_sum(tf.abs(V_un)**2)
        V     = V_un * tf.cast(tf.sqrt(p_max / p_now), dtype)               # scaled V (Nt×Ns)

    # ---------- precoding --------------------------------------------------
    x = tf.expand_dims(x, -1)
    x_precoded = tf.squeeze(tf.matmul(V, x), -1)   # [B, Ns] broadcasting

    if return_precoding_matrix:
        return x_precoded, V, HV                          # HV gives per-UE gains
    return x_precoded
