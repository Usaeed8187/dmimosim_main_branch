import numpy as np
import matplotlib.pyplot as plt


def solve_discrete_riccati_steady_state(F, Q, R, max_iter=100, tol=1e-12):
    """
    Solve for the steady-state predicted error covariance P_minus
    for the model

        x_t = F x_{t-1} + w_t
        y_t = x_t + n_t

    where C = I.

    The iteration is:
        P_plus  = P_minus - P_minus (P_minus + R)^(-1) P_minus
        P_minus_next = F P_plus F^T + Q

    Returns
    -------
    P_minus : ndarray
        Steady-state predicted covariance, i.e. cov(x_t - xhat_{t|t-1})
    K : ndarray
        Steady-state Kalman gain for correction at time t:
            K = P_minus (P_minus + R)^(-1)
    """
    d = F.shape[0]
    P_minus = np.eye(d)

    for _ in range(max_iter):
        S = P_minus + R
        K = P_minus @ np.linalg.inv(S)
        P_plus = P_minus - K @ P_minus
        P_minus_next = F @ P_plus @ F.T + Q

        err = np.linalg.norm(P_minus_next - P_minus, ord='fro')
        P_minus = P_minus_next
        # print("err = ", err)
        if err < tol:
            break

    S = P_minus + R
    K = P_minus @ np.linalg.inv(S)
    return P_minus, K


def generate_state_space_data(T, F, Q, R, seed=0):
    """
    Generate data from:
        x_t = F x_{t-1} + w_t
        y_t = x_t + n_t
    """
    rng = np.random.default_rng(seed)
    d = F.shape[0]

    x = np.zeros((T, d))
    y = np.zeros((T, d))
    w = np.zeros((T, d))
    n = np.zeros((T, d))

    x_prev = rng.multivariate_normal(np.zeros(d), np.eye(d))

    for t in range(T):
        w_t = rng.multivariate_normal(np.zeros(d), Q)
        x_t = F @ x_prev + w_t

        n_t = rng.multivariate_normal(np.zeros(d), R)
        y_t = x_t + n_t

        x[t] = x_t
        y[t] = y_t
        w[t] = w_t
        n[t] = n_t

        x_prev = x_t

    return x, y, w, n


def run_steady_state_predictor(y, F, K):
    """
    Run the steady-state one-step predictor:

        xhat_{t|t-1} = (F - F K) xhat_{t-1|t-2} + F K y_{t-1}

    since C = I.

    Input
    -----
    y : array, shape (T, d)

    Returns
    -------
    xhat_pred : array, shape (T, d)
        xhat_pred[t] = xhat_{t|t-1}
    """
    T, d = y.shape
    A_p = F - F @ K
    B_p = F @ K

    xhat_pred = np.zeros((T, d))
    s_prev = np.zeros(d)  # this is xhat_{t-1|t-2}

    for t in range(T):
        if t == 0:
            # no y_{-1}; keep zero initialization
            s_t = A_p @ s_prev
        else:
            s_t = A_p @ s_prev + B_p @ y[t - 1]

        xhat_pred[t] = s_t
        s_prev = s_t

    return xhat_pred


def analytical_transfer_response(F, K, omega_grid):
    """
    Compute the analytical transfer matrix

        H_pred(e^{jω}) = (I - A_p e^{-jω})^{-1} B_p e^{-jω}

    with
        A_p = F - F K
        B_p = F K
    """
    d = F.shape[0]
    A_p = F - F @ K
    B_p = F @ K

    H = np.zeros((len(omega_grid), d, d), dtype=np.complex128)
    I = np.eye(d)

    for i, w in enumerate(omega_grid):
        z_inv = np.exp(-1j * w)
        H[i] = np.linalg.inv(I - A_p * z_inv) @ (B_p * z_inv)

    return H


def empirical_transfer_response(y, xhat_pred, nfft=4096, discard=500):
    """
    Estimate empirical transfer function from y -> xhat_pred using FFT:

        H_emp[k, i, j] = S_xy[k, i, j] / S_yy[k, j, j]

    where
        input  = y[:, j]
        output = xhat_pred[:, i]

    We use a simple frequency-domain ratio on one long realization.
    For verification/sanity checks this is fine.

    Returns
    -------
    omega : ndarray, shape (nfft,)
    H_emp : ndarray, shape (nfft, d_out, d_in)
    """
    y_use = y[discard:]
    x_use = xhat_pred[discard:]

    T, d_in = y_use.shape
    _, d_out = x_use.shape

    Y = np.fft.fft(y_use, n=nfft, axis=0)
    X = np.fft.fft(x_use, n=nfft, axis=0)

    eps = 1e-12
    H_emp = np.zeros((nfft, d_out, d_in), dtype=np.complex128)

    for i in range(d_out):
        for j in range(d_in):
            S_xy = X[:, i] * np.conj(Y[:, j])
            S_yy = Y[:, j] * np.conj(Y[:, j])
            H_emp[:, i, j] = S_xy / (S_yy + eps)

    omega = 2 * np.pi * np.arange(nfft) / nfft
    return omega, H_emp


def main():
    # ------------------------------------------------------------------
    # 1) Define a nontrivial stable F
    # ------------------------------------------------------------------
    F = np.array([
        [0.8, 0.6],
        [-0.4, 0.7],
    ], dtype=float)

    eigvals = np.linalg.eigvals(F)
    print("Eigenvalues of F:", eigvals)
    print("Max |eig|:", np.max(np.abs(eigvals)))

    # Make sure F is stable
    if np.max(np.abs(eigvals)) >= 1.0:
        raise ValueError("F is not stable. Pick a matrix with spectral radius < 1.")

    # Process and observation noise covariances
    Q = np.array([
        [0.03, 0.01],
        [0.01, 0.02],
    ], dtype=float)

    R = np.array([
        [0.08, 0.02],
        [0.02, 0.07],
    ], dtype=float)

    # ------------------------------------------------------------------
    # 2) Solve for steady-state Kalman gain
    # ------------------------------------------------------------------
    P_minus, K = solve_discrete_riccati_steady_state(F, Q, R)

    # print("\nSteady-state predicted covariance P_minus:\n", P_minus)
    # print("\nSteady-state Kalman gain K:\n", K)

    # Predictor matrices
    A_p = F - F @ K
    B_p = F @ K

    # print("\nA_p = F - F K:\n", A_p)
    # print("\nB_p = F K:\n", B_p)

    # ------------------------------------------------------------------
    # 3) Generate data
    # ------------------------------------------------------------------
    T = 300
    x, y, w, n = generate_state_space_data(T=T+1, F=F, Q=Q, R=R, seed=123)

    # ------------------------------------------------------------------
    # 4) Run the steady-state predictor recursion
    # ------------------------------------------------------------------
    xhat_pred = run_steady_state_predictor(y[:-1,:], F, K)

    xhat_pred_T_plus_1 = xhat_pred[-1, :]
    x_T_plus_1 = x[-1, :]

    pred_nmse = np.mean(np.abs(xhat_pred_T_plus_1 - x_T_plus_1)**2) / np.mean(np.abs(x_T_plus_1)**2)
    print("Prediction NMSE Using Steady State Predictor: ", pred_nmse)

    # ------------------------------------------------------------------
    # 5) Analytical transfer response
    # ------------------------------------------------------------------
    nfft = 64
    omega_grid = 2 * np.pi * np.arange(nfft) / nfft
    H_ana = analytical_transfer_response(F, K, omega_grid)

    # ------------------------------------------------------------------
    # 6) Empirical transfer response from simulated data
    # ------------------------------------------------------------------
    omega_emp, H_emp = empirical_transfer_response(y, xhat_pred, nfft=nfft, discard=1000)

    # ------------------------------------------------------------------
    # 7) Compare analytical vs empirical transfer function
    # ------------------------------------------------------------------

    mse_ana_pred = np.mean((H_ana - H_emp) ** 2)

    print("MSE between analytical vs empirical transfer function:", mse_ana_pred)

    entry_list = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    half = nfft // 2  # plot [0, pi)

    for (i, j) in entry_list:
        plt.figure(figsize=(10, 4))
        plt.plot(omega_grid[:half], np.abs(H_ana[:half, i, j]), label="|H analytical|")
        plt.plot(omega_emp[:half], np.abs(H_emp[:half, i, j]), '--', label="|H empirical|")
        plt.xlabel("Temporal frequency ω [rad/sample]")
        plt.ylabel(f"Magnitude entry ({i},{j})")
        plt.title(f"Analytical vs empirical magnitude: H[{i},{j}]")
        plt.legend()
        plt.tight_layout()

    for (i, j) in entry_list:
        plt.figure(figsize=(10, 4))
        plt.plot(omega_grid[:half], np.unwrap(np.angle(H_ana[:half, i, j])), label="∠H analytical")
        plt.plot(omega_emp[:half], np.unwrap(np.angle(H_emp[:half, i, j])), '--', label="∠H empirical")
        plt.xlabel("Temporal frequency ω [rad/sample]")
        plt.ylabel(f"Phase entry ({i},{j}) [rad]")
        plt.title(f"Analytical vs empirical phase: H[{i},{j}]")
        plt.legend()
        plt.tight_layout()

    # ------------------------------------------------------------------
    # 8) Optional sanity check in time domain
    # ------------------------------------------------------------------
    mse_pred = np.mean((x[1:] - xhat_pred[1:]) ** 2)
    mse_obs = np.mean((x - y) ** 2)

    print("\nTime-domain sanity checks:")
    print("MSE of predictor vs true state:", mse_pred)
    print("MSE of raw observation vs true state:", mse_obs)

    plt.figure(figsize=(10, 4))
    t = np.arange(T)
    plt.plot(t, x[:T, 0], label="true x[0]")
    plt.plot(t, y[:T, 0], label="obs y[0]", alpha=0.7)
    plt.plot(t, xhat_pred[:T, 0], label="pred xhat[0| - ]", alpha=0.8)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Time-domain sanity check, state dimension 0")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()