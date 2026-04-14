import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Basic helpers
# ============================================================

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def project_to_stable_matrix(F: np.ndarray, max_radius: float = 0.99) -> np.ndarray:
    eigvals = np.linalg.eigvals(F)
    rho = np.max(np.abs(eigvals))
    if rho >= max_radius:
        F = F * (max_radius / (rho + 1e-15))
    return F


# ============================================================
# Steady-state Kalman predictor ingredients
# ============================================================

def solve_discrete_riccati_steady_state(
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-12,
):
    d = F.shape[0]
    P_minus = np.eye(d)

    for _ in range(max_iter):
        S = P_minus + R
        K = P_minus @ np.linalg.inv(S)
        P_plus = P_minus - K @ P_minus
        P_minus_next = F @ P_plus @ F.T + Q
        err = np.linalg.norm(P_minus_next - P_minus, ord="fro")
        P_minus = P_minus_next
        if err < tol:
            break

    S = P_minus + R
    K = P_minus @ np.linalg.inv(S)
    return P_minus, K


def steady_state_predictor_matrices(F_hat: np.ndarray, Q: np.ndarray, R: np.ndarray):
    _, K_ss = solve_discrete_riccati_steady_state(F_hat, Q, R)
    A_p = F_hat - F_hat @ K_ss
    B_p = F_hat @ K_ss
    return A_p, B_p, K_ss


def run_true_steady_state_predictor(
    y_hist: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
):
    """
    Returns xhat_{history_len | history_len-1}
    after ingesting y_0, ..., y_{history_len-1}.
    """
    A_p, B_p, _ = steady_state_predictor_matrices(F, Q, R)
    s = np.zeros((F.shape[0],), dtype=float)
    for t in range(y_hist.shape[0]):
        s = A_p @ s + B_p @ y_hist[t]
    return s


# ============================================================
# Data generation and EM estimation of F
# ============================================================

def generate_state_space_data(
    T: int,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
):
    d = F.shape[0]
    x = np.zeros((T, d))
    y = np.zeros((T, d))

    x_prev = rng.multivariate_normal(np.zeros(d), np.eye(d))

    for t in range(T):
        w_t = rng.multivariate_normal(np.zeros(d), Q)
        x_t = F @ x_prev + w_t

        n_t = rng.multivariate_normal(np.zeros(d), R)
        y_t = x_t + n_t

        x[t] = x_t
        y[t] = y_t
        x_prev = x_t

    return x, y


def kalman_filter_identity_H(y: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
    T, d = y.shape

    x_pred = np.zeros((T, d))
    P_pred = np.zeros((T, d, d))
    x_filt = np.zeros((T, d))
    P_filt = np.zeros((T, d, d))
    K_hist = np.zeros((T, d, d))

    x_plus_prev = np.zeros(d)
    P_plus_prev = np.eye(d)

    for t in range(T):
        x_minus = F @ x_plus_prev
        P_minus = F @ P_plus_prev @ F.T + Q
        P_minus = symmetrize(P_minus)

        S = P_minus + R
        K_t = P_minus @ np.linalg.inv(S)
        innovation = y[t] - x_minus
        x_plus = x_minus + K_t @ innovation
        P_plus = (np.eye(d) - K_t) @ P_minus
        P_plus = symmetrize(P_plus)

        x_pred[t] = x_minus
        P_pred[t] = P_minus
        x_filt[t] = x_plus
        P_filt[t] = P_plus
        K_hist[t] = K_t

        x_plus_prev = x_plus
        P_plus_prev = P_plus

    return x_pred, P_pred, x_filt, P_filt, K_hist


def rts_smoother_with_lag_cov(y: np.ndarray, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
    x_pred, P_pred, x_filt, P_filt, K_hist = kalman_filter_identity_H(y, F, Q, R)

    T, d = y.shape
    x_smooth = np.zeros((T, d))
    P_smooth = np.zeros((T, d, d))
    J = np.zeros((max(T - 1, 1), d, d))

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        J_t = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        J[t] = J_t

        x_smooth[t] = x_filt[t] + J_t @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ J_t.T
        P_smooth[t] = symmetrize(P_smooth[t])

    P_lag = np.zeros((T, d, d))
    if T >= 2:
        P_lag[T - 1] = (np.eye(d) - K_hist[T - 1]) @ F @ P_filt[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = (
                P_filt[t] @ J[t - 1].T
                + J[t] @ (P_lag[t + 1] - F @ P_filt[t]) @ J[t - 1].T
            )
            P_lag[t] = symmetrize(P_lag[t])

    return x_smooth, P_smooth, P_lag


def estimate_F_init_from_covariances(y: np.ndarray, R: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    Y1 = y_centered[1:]
    Y0 = y_centered[:-1]

    Gamma_y_1 = (Y1.T @ Y0) / max(len(Y0), 1)
    Gamma_y_0 = (Y0.T @ Y0) / max(len(Y0), 1)

    Gamma_x_0 = symmetrize(Gamma_y_0 - R)
    Gamma_x_0 = Gamma_x_0 + reg * np.eye(Gamma_x_0.shape[0])

    F0 = Gamma_y_1 @ np.linalg.pinv(Gamma_x_0)
    F0 = project_to_stable_matrix(F0, max_radius=0.99)
    return F0


def estimate_F_from_y_em(
    y: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    F_init: np.ndarray | None = None,
    max_em_iters: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    T, d = y.shape
    if T < 2:
        raise ValueError("Need at least T >= 2 samples to estimate F.")

    if F_init is None:
        F = estimate_F_init_from_covariances(y, R)
    else:
        F = F_init.copy()

    F = project_to_stable_matrix(F, max_radius=0.99)

    for _ in range(max_em_iters):
        F_old = F.copy()

        x_smooth, P_smooth, P_lag = rts_smoother_with_lag_cov(y, F, Q, R)

        S10 = np.zeros((d, d))
        S00 = np.zeros((d, d))

        for t in range(1, T):
            Ex_t_xtm1 = P_lag[t] + np.outer(x_smooth[t], x_smooth[t - 1])
            Ex_tm1_xtm1 = P_smooth[t - 1] + np.outer(x_smooth[t - 1], x_smooth[t - 1])
            S10 += Ex_t_xtm1
            S00 += Ex_tm1_xtm1

        F = S10 @ np.linalg.pinv(S00)
        F = project_to_stable_matrix(F, max_radius=0.99)

        delta = np.linalg.norm(F - F_old, ord="fro")
        if delta < tol:
            break

    return F


# ============================================================
# Transfer sampling / PCA
# ============================================================

def steady_state_predictor_transfer_samples(
    A_p: np.ndarray,
    B_p: np.ndarray,
    num_freqs: int,
):
    """
    Sample H(e^{jw}) = (I - A_p e^{-jw})^{-1} B_p
    over the FULL unit circle [0, 2pi), which matches the papers.
    """
    d = A_p.shape[0]
    omegas = np.linspace(0.0, 2.0 * np.pi, num_freqs, endpoint=False)
    H_samps = np.zeros((num_freqs, d, d), dtype=np.complex128)
    I = np.eye(d, dtype=np.complex128)

    for i, w in enumerate(omegas):
        zinv = np.exp(-1j * w)
        H_samps[i] = np.linalg.inv(I - A_p.astype(np.complex128) * zinv) @ B_p.astype(np.complex128)

    return H_samps, omegas


def vectorize_transfer_samples(H_samps: np.ndarray) -> np.ndarray:
    num_freqs, d_out, d_in = H_samps.shape
    v_blocks = [H_samps[i].reshape(d_out * d_in, order="F") for i in range(num_freqs)]
    return np.concatenate(v_blocks, axis=0)


def estimate_empirical_complex_covariance(V: np.ndarray):
    mean_v = np.mean(V, axis=1, keepdims=True)
    Vc = V - mean_v
    K_vv = (Vc @ Vc.conj().T) / V.shape[1]
    K_vv = 0.5 * (K_vv + K_vv.conj().T)
    return mean_v, Vc, K_vv


def pca_from_covariance(K_vv: np.ndarray):
    evals, evecs = np.linalg.eigh(K_vv)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def reconstruct_with_first_m_modes(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q: np.ndarray,
    m: int,
) -> np.ndarray:
    if m == 0:
        return np.repeat(mean_v, V.shape[1], axis=1)

    Qm = Q[:, :m]
    Vc = V - mean_v
    Vhat = mean_v + Qm @ (Qm.conj().T @ Vc)
    return Vhat


def nmse_between_matrices(Vhat: np.ndarray, V: np.ndarray) -> float:
    num = np.linalg.norm(Vhat - V, ord="fro") ** 2
    den = np.linalg.norm(V, ord="fro") ** 2
    return float(np.real(num / max(den, 1e-15)))


def compute_reconstruction_nmse_curve(
    V: np.ndarray,
    mean_v: np.ndarray,
    Q: np.ndarray,
    max_m: int,
):
    m_vals = np.arange(0, max_m + 1)
    nmse_vals = np.zeros_like(m_vals, dtype=float)

    for i, m in enumerate(m_vals):
        Vhat = reconstruct_with_first_m_modes(V, mean_v, Q, m)
        nmse_vals[i] = nmse_between_matrices(Vhat, V)

    return m_vals, nmse_vals


# ============================================================
# Chunked transfer-vector generation
# ============================================================

def generate_chunkwise_transfer_vectors(
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    history_len: int,
    num_chunks: int,
    num_freqs: int,
    seed: int = 0,
    max_em_iters: int = 50,
    em_tol: float = 1e-6,
):
    rng = np.random.default_rng(seed)
    total_T = num_chunks * (history_len + 1)
    x_all, y_all = generate_state_space_data(total_T, F_true, Q, R, rng)
    chunk_len = history_len + 1

    V_list = []
    F_hats = []
    A_ps = []
    B_ps = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len

        y_chunk = y_all[start:end]
        y_hist = y_chunk[:history_len]

        F_init = estimate_F_init_from_covariances(y_hist, R)
        F_hat = estimate_F_from_y_em(
            y=y_hist,
            Q=Q,
            R=R,
            F_init=F_init,
            max_em_iters=max_em_iters,
            tol=em_tol,
        )

        A_p, B_p, _ = steady_state_predictor_matrices(F_hat, Q, R)
        H_samps, omegas = steady_state_predictor_transfer_samples(A_p, B_p, num_freqs=num_freqs)
        v_k = vectorize_transfer_samples(H_samps)

        V_list.append(v_k)
        F_hats.append(F_hat)
        A_ps.append(A_p)
        B_ps.append(B_p)

    V = np.stack(V_list, axis=1)

    return {
        "x_all": x_all,
        "y_all": y_all,
        "V": V,
        "F_hats": np.stack(F_hats, axis=0),
        "A_ps": np.stack(A_ps, axis=0),
        "B_ps": np.stack(B_ps, axis=0),
        "omegas": omegas,
    }


# ============================================================
# Exact-Q bank (SISO): mean + top-m PCA modes
# ============================================================

def q_column_to_frequency_vector(q_col: np.ndarray, num_freqs: int) -> np.ndarray:
    if q_col.ndim != 1 or q_col.size != num_freqs:
        raise ValueError("For SISO, q_col must be length num_freqs.")
    return q_col.astype(np.complex128)


def sampled_freq_response_to_causal_fir(H_w: np.ndarray, fir_len: int) -> np.ndarray:
    """
    With full-circle uniform sampling, use IDFT to obtain a length-N circular
    impulse response and keep the first fir_len taps as a long FIR realization.
    """
    h_full = np.fft.ifft(H_w)
    taps = h_full[:fir_len].copy()
    return taps


class SISOFIRBank:
    def __init__(self, taps: np.ndarray):
        """
        taps shape: (num_filters, fir_len)
        """
        self.taps = taps.astype(np.complex128)
        self.num_filters, self.fir_len = taps.shape
        self.state = np.zeros((self.num_filters, self.fir_len), dtype=np.complex128)

    def reset(self):
        self.state.fill(0.0)

    def step(self, u_t: complex) -> np.ndarray:
        self.state[:, 1:] = self.state[:, :-1]
        self.state[:, 0] = u_t
        y = np.sum(self.taps * self.state, axis=1)
        return y


def build_exact_q_fir_bank_with_mean(
    mean_v: np.ndarray,
    Q_eig: np.ndarray,
    num_freqs: int,
    m: int,
    fir_len: int,
):
    """
    Bank contains:
      1) mean transfer function
      2) first m PCA eigenvectors
    Each is realized as a long FIR using full-circle IDFT.
    """
    num_filters = 1 + m
    taps = np.zeros((num_filters, fir_len), dtype=np.complex128)

    mean_H = q_column_to_frequency_vector(mean_v[:, 0], num_freqs)
    taps[0] = sampled_freq_response_to_causal_fir(mean_H, fir_len)

    for j in range(m):
        Hj = q_column_to_frequency_vector(Q_eig[:, j], num_freqs)
        taps[1 + j] = sampled_freq_response_to_causal_fir(Hj, fir_len)

    return SISOFIRBank(taps=taps)


def run_bank_and_collect_features(bank: SISOFIRBank, y_seq: np.ndarray) -> np.ndarray:
    bank.reset()
    feats = []
    for t in range(y_seq.shape[0]):
        out = bank.step(complex(y_seq[t, 0]))
        feats.append(out.copy())
    return np.stack(feats, axis=0)


# ============================================================
# Global readout training/testing
# ============================================================

def collect_global_train_pairs(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    bank_builder,
    target_kind: str = "x",
):
    """
    Collect all one-step pairs across all training chunks:
        feature_t -> target_{t+1}
    for t=0,...,history_len-2 in each chunk.
    """
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len

    Z_blocks = []
    Y_blocks = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]

        bank = bank_builder()
        feat_seq = run_bank_and_collect_features(bank, y_chunk[:history_len])

        target_seq = x_chunk if target_kind == "x" else y_chunk
        Z_blocks.append(feat_seq[:-1])
        Y_blocks.append(target_seq[1:history_len])

    Z = np.concatenate(Z_blocks, axis=0)
    Y = np.concatenate(Y_blocks, axis=0)
    return Z, Y


def solve_global_complex_readout(
    Z: np.ndarray,
    Y: np.ndarray,
    reg: float,
) -> np.ndarray:
    G = Z.conj().T @ Z
    rhs = Z.conj().T @ Y
    I = np.eye(G.shape[0], dtype=np.complex128)
    return np.linalg.solve(G + reg * I, rhs)


def evaluate_prediction_nmse_with_fixed_global_readout(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    bank_builder,
    W_out: np.ndarray,
    target_kind: str = "x",
):
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len

    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]

        bank = bank_builder()
        feat_seq = run_bank_and_collect_features(bank, y_chunk[:history_len])

        z_last = feat_seq[-1]
        pred = z_last @ W_out

        target_seq = x_chunk if target_kind == "x" else y_chunk
        true = target_seq[-1]

        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


def evaluate_true_kalman_baseline_nmse(
    x_all: np.ndarray,
    y_all: np.ndarray,
    F_true: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    history_len: int,
    target_kind: str = "x",
):
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len

    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]

        pred = run_true_steady_state_predictor(y_chunk[:history_len], F_true, Q, R)
        true = x_chunk[-1] if target_kind == "x" else y_chunk[-1]

        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


def evaluate_previous_observation_baseline_nmse(
    x_all: np.ndarray,
    y_all: np.ndarray,
    history_len: int,
    target_kind: str = "x",
):
    chunk_len = history_len + 1
    num_chunks = x_all.shape[0] // chunk_len

    preds = []
    trues = []

    for k in range(num_chunks):
        start = k * chunk_len
        end = (k + 1) * chunk_len
        y_chunk = y_all[start:end]
        x_chunk = x_all[start:end]

        pred = y_chunk[history_len - 1]
        true = x_chunk[-1] if target_kind == "x" else y_chunk[-1]

        preds.append(pred)
        trues.append(true)

    P = np.stack(preds, axis=0)
    T = np.stack(trues, axis=0)
    num = np.linalg.norm(P - T) ** 2
    den = np.linalg.norm(T) ** 2
    return float(np.real(num / max(den, 1e-15)))


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # SISO case only
    # --------------------------------------------------------
    F_true = np.array([[0.7]], dtype=float)
    F_true = project_to_stable_matrix(F_true, max_radius=0.99)

    Q_process = np.array([[1e-2]], dtype=float)
    R_obs = np.array([[5e-7]], dtype=float)

    history_len = 100
    num_train_chunks = 400
    num_test_chunks = 400
    num_freqs = 256

    max_em_iters = 50
    em_tol = 1e-6
    seed_train = 12345
    seed_test = 67890

    max_m = 10
    target_kind = "x"
    readout_reg = 1e-2
    exact_q_fir_len = 96

    # --------------------------------------------------------
    # Offline configuration stage: gather transfer statistics
    # --------------------------------------------------------
    train_stats = generate_chunkwise_transfer_vectors(
        F_true=F_true,
        Q=Q_process,
        R=R_obs,
        history_len=history_len,
        num_chunks=num_train_chunks,
        num_freqs=num_freqs,
        seed=seed_train,
        max_em_iters=max_em_iters,
        em_tol=em_tol,
    )

    V = train_stats["V"]
    mean_v, _, K_vv = estimate_empirical_complex_covariance(V)
    evals, Q_eig = pca_from_covariance(K_vv)

    max_m = min(max_m, Q_eig.shape[1])

    print("Collected transfer-function sample matrix V with shape:", V.shape)
    print("Empirical covariance K_vv shape:", K_vv.shape)
    print("Top 10 eigenvalues:")
    print(np.real(evals[:10]))

    # --------------------------------------------------------
    # Transfer-space NMSE vs m
    # --------------------------------------------------------
    m_vals, nmse_exact_q = compute_reconstruction_nmse_curve(
        V=V,
        mean_v=mean_v,
        Q=Q_eig,
        max_m=max_m,
    )

    # --------------------------------------------------------
    # Separate train/test time-domain data for readout learning
    # --------------------------------------------------------
    rng_train = np.random.default_rng(seed_train + 1000)
    total_train_T = num_train_chunks * (history_len + 1)
    x_train_all, y_train_all = generate_state_space_data(
        total_train_T, F_true, Q_process, R_obs, rng_train
    )

    rng_test = np.random.default_rng(seed_test)
    total_test_T = num_test_chunks * (history_len + 1)
    x_test_all, y_test_all = generate_state_space_data(
        total_test_T, F_true, Q_process, R_obs, rng_test
    )

    pred_nmse_exact = np.full_like(m_vals, np.nan, dtype=float)

    for i, m in enumerate(m_vals):
        if m == 0:
            continue

        def bank_builder(m_local=m):
            return build_exact_q_fir_bank_with_mean(
                mean_v=mean_v,
                Q_eig=Q_eig,
                num_freqs=num_freqs,
                m=m_local,
                fir_len=exact_q_fir_len,
            )

        Z_train, Y_train = collect_global_train_pairs(
            x_all=x_train_all,
            y_all=y_train_all,
            history_len=history_len,
            bank_builder=bank_builder,
            target_kind=target_kind,
        )

        W_out = solve_global_complex_readout(Z_train, Y_train, reg=readout_reg)

        pred_nmse_exact[i] = evaluate_prediction_nmse_with_fixed_global_readout(
            x_all=x_test_all,
            y_all=y_test_all,
            history_len=history_len,
            bank_builder=bank_builder,
            W_out=W_out,
            target_kind=target_kind,
        )

        print(
            f"m={m:2d} | transfer NMSE={nmse_exact_q[i]:.4e} | "
            f"prediction NMSE={pred_nmse_exact[i]:.4e}"
        )

    kalman_baseline_nmse = evaluate_true_kalman_baseline_nmse(
        x_all=x_test_all,
        y_all=y_test_all,
        F_true=F_true,
        Q=Q_process,
        R=R_obs,
        history_len=history_len,
        target_kind=target_kind,
    )

    prev_obs_nmse = evaluate_previous_observation_baseline_nmse(
        x_all=x_test_all,
        y_all=y_test_all,
        history_len=history_len,
        target_kind=target_kind,
    )

    print(f"True steady-state Kalman baseline NMSE: {kalman_baseline_nmse:.4e}")
    print(f"Previous-observation baseline NMSE:    {prev_obs_nmse:.4e}")

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------
    np.savez(
        "siso_kalman_exact_q_prediction_stats.npz",
        F_true=F_true,
        Q_process=Q_process,
        R_obs=R_obs,
        history_len=history_len,
        num_train_chunks=num_train_chunks,
        num_test_chunks=num_test_chunks,
        num_freqs=num_freqs,
        V=V,
        mean_v=mean_v,
        K_vv=K_vv,
        evals=evals,
        Q_eig=Q_eig,
        m_vals=m_vals,
        nmse_exact_q=nmse_exact_q,
        pred_nmse_exact=pred_nmse_exact,
        target_kind=target_kind,
        readout_reg=readout_reg,
        exact_q_fir_len=exact_q_fir_len,
        kalman_baseline_nmse=kalman_baseline_nmse,
        prev_obs_nmse=prev_obs_nmse,
    )

    # --------------------------------------------------------
    # Plot 1: transfer-space NMSE vs m
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(m_vals, nmse_exact_q, marker="o")
    plt.xlabel("Number of retained eigenvectors (m)")
    plt.ylabel("Reconstruction NMSE of transfer samples")
    plt.title("SISO exact PCA basis: transfer-space NMSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("siso_nmse_vs_m_exactQ_transfer.png", dpi=200)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: prediction NMSE vs m
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(
        m_vals[1:],
        pred_nmse_exact[1:],
        marker="o",
        label="Exact-Q bank + global LS readout",
    )
    plt.axhline(
        kalman_baseline_nmse,
        linestyle="--",
        label="True steady-state Kalman baseline",
    )
    plt.axhline(
        prev_obs_nmse,
        linestyle=":",
        label="Previous-observation baseline",
    )
    plt.xlabel("Number of retained eigenvectors (m)")
    plt.ylabel(f"Prediction NMSE vs true {target_kind}_N")
    plt.title("SISO exact PCA basis: prediction NMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("siso_prediction_nmse_vs_m_exactQ.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
