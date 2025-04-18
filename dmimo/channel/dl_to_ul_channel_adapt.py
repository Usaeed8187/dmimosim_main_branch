import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class dl_to_ul_channel_adapt:
    def __init__(self, carrier_freq=3.5e9, antenna_spacing_ue=0.5, antenna_spacing_gnb=0.5, coherence_time=0.01, time_gap=1e-3, max_paths=3, num_ues=10, subcarrier_spacing=15e3):
        self.num_ue_antennas = 2
        self.num_gnb_antennas = 4
        self.num_ues = num_ues
        self.ofdm_syms = 14
        self.subcarriers = 512
        self.max_paths = max_paths
        self.subcarrier_spacing = subcarrier_spacing

        self.carrier_freq = carrier_freq
        self.speed_of_light = 3e8
        self.wavelength = self.speed_of_light / self.carrier_freq
        self.antenna_spacing_ue = antenna_spacing_ue
        self.antenna_spacing_gnb = antenna_spacing_gnb
        self.coherence_time = coherence_time
        self.time_gap = time_gap

        self.subcarrier_indices = tf.range(0, self.subcarriers, 10)
        self.num_subcarriers = len(self.subcarrier_indices)
        self.psi_grid = np.linspace(-1, 1, 100)  # Discretized psi values

    def fourier_matrix(self, num_antennas, antenna_spacing, curr_wavelength):
        
        K = num_antennas
        indices = tf.range(K, dtype=tf.float32)[:, None]  # [K, 1]
        j_prime = tf.range(K, dtype=tf.float32)[None, :]  # [1, K]
        psi_prime = 2.0 / K  # psi' = 2/K as per paper
        phase = -2 * np.pi * indices * antenna_spacing * self.wavelength * j_prime * psi_prime / curr_wavelength  # [K, K]
        F = tf.complex(tf.cos(phase), tf.sin(phase))
        
        return F

    def sinc_matrix(self, lambda_i, psi_j, L, psi_prime):
        """
        Compute S_i matrix for subcarrier i with wavelength lambda_i.
        S_i[j] = (L / lambda_i) * sinc( (L (i * psi' - psi_j)) / lambda_i )
        Shape: [K, N]
        """
        K = self.num_gnb_antennas
        N = self.max_paths
        # Antenna indices (though paper uses 'i', assuming it’s a typo for 'k')
        k = tf.range(K, dtype=tf.float32)[:, None]  # [K, 1]
        psi_j = psi_j[None, :]  # [1, N]
        # Assuming 'i' in sinc argument is subcarrier index, but we need K antennas
        # Interpreting as k * psi' for each antenna; psi' is a fixed parameter
        arg = (L * (k * psi_prime - psi_j)) / lambda_i
        S_i = (L / lambda_i) * tf.experimental.numpy.sinc(arg)  # [K, N]
        return S_i

    def initialize_paths(self, channel_data, wavelengths):
        """Initialize psi_n^gNB and d_n per UE antenna as per Section 5.3.
        
        Args:
            channel_data: Channel matrix [num_gnb_antennas, num_ue_antennas, num_subcarriers].
            wavelengths: Wavelengths for each subcarrier [num_subcarriers].
        
        Returns:
            init_psi_gnb: Initial gNB angles [max_paths].
            init_psi_ue: Initial UE angles [max_paths], set to 0.0.
            init_d: Initial path distances [max_paths].
        """
        num_d = 100
        num_psi = 100
        d_grid = np.linspace(1, 100, num_d)
        psi_grid = np.linspace(-1, 1, num_psi)
        P = tf.zeros((num_d, num_psi, self.num_ue_antennas), dtype=tf.float32)

        for m in range(self.num_ue_antennas):
            sum_result = tf.zeros((num_d, num_psi), dtype=tf.complex64)
            for i, lambda_i in enumerate(wavelengths):
                h_i = channel_data[:, :, i]  # [4, 2]
                for k in range(self.num_gnb_antennas):
                    phase = 2 * np.pi * (
                        d_grid[:, None] + k * self.antenna_spacing_gnb * psi_grid[None, :]
                    ) / lambda_i
                    sum_result += h_i[k, m] * tf.exp(tf.complex(0.0, tf.cast(phase, tf.float32)))

            indices = tf.stack([
                tf.repeat(tf.range(num_d), num_d),  # Row indices
                tf.tile(tf.range(num_psi), [num_psi]),  # Column indices
                tf.fill([num_d * num_psi], m)         # mth index in the last dimension
            ], axis=-1)
            updates = tf.reshape(tf.abs(sum_result)**2, [-1])

            P = tf.tensor_scatter_nd_update(P, indices, updates)

        # Select peaks per UE antenna
        P = P.numpy()
        init_d = np.zeros((self.max_paths, self.num_ue_antennas), dtype=np.float32)
        init_psi_gnb = np.zeros((self.max_paths, self.num_ue_antennas), dtype=np.float32)
        init_psi_ue = np.zeros((self.max_paths, self.num_ue_antennas), dtype=np.float32)

        for m in range(self.num_ue_antennas):
            curr_P = P[:, :, m]
            indices = np.unravel_index(np.argsort(curr_P.ravel())[-self.max_paths:], curr_P.shape)
            # Sort peaks by index to maintain order
            peak_indices = list(zip(indices[0], indices[1]))
            peak_indices.sort(key=lambda x: curr_P[x[0], x[1]], reverse=True)
            for i, (d_idx, psi_idx) in enumerate(peak_indices[:self.max_paths]):
                init_d[i, m] = d_grid[d_idx]
                init_psi_gnb[i, m] = psi_grid[psi_idx]

        tf.print("P max values per UE antenna:", [np.max(P[:, :, m]) for m in range(self.num_ue_antennas)])
        tf.print("Selected init_d:", init_d)
        tf.print("Selected init_psi_gnb:", init_psi_gnb)

        return init_psi_gnb, init_d

    def estimate_paths_for_ue(self, channel_ue):
        """Estimate path parameters for a single UE using spatial-domain optimization."""
        selected_channels = tf.gather(channel_ue, self.subcarrier_indices, axis=-1)  # [4, 2, 14, num_subcarriers]
        channel_mean = tf.reduce_mean(selected_channels, axis=2)  # [4, 2, num_subcarriers]
        
        self.wavelengths = self.speed_of_light / (self.carrier_freq + self.subcarrier_spacing * tf.cast(self.subcarrier_indices - 256, tf.float32))
        
        # Compute spatial profiles P_i^m for each subcarrier and UE antenna
        P_list = []
        for m in range(self.num_ue_antennas):
            P_m = []
            for i in range(self.num_subcarriers):
                F_inv = tf.linalg.inv(self.fourier_matrix(self.num_gnb_antennas, self.antenna_spacing_gnb, self.wavelengths[i]))  # [4, 100]
                h_i_m = channel_mean[:, m, i]  # [4]
                P_i_m = tf.matmul(F_inv, tf.cast(h_i_m[:, None], tf.complex64))  # [100]
                P_i_m = tf.squeeze(P_i_m)
                P_m.append(P_i_m)
            P_m = tf.concat(P_m, axis=0)  # [self.num_gnb_antennas * self.num_subcarriers]
            P_list.append(P_m)
        self.P = tf.cast(tf.stack(P_list, axis=-1), tf.complex128)  # [self.num_gnb_antennas * self.num_subcarriers, 2]

        # Initialize parameters
        init_psi_gnb, init_d = self.initialize_paths(channel_mean, self.wavelengths)
        
        # ---------------------------------------------------------------------------
        # configuration and dtype helpers
        # ---------------------------------------------------------------------------
        R = tf.float64                # real scalars
        C = tf.complex128             # complex scalars

        l_gnb = self.antenna_spacing_gnb          # 0.5 (in λ units)
        l_ue  = self.antenna_spacing_ue           # 0.5 (in λ units)
        K     = self.num_gnb_antennas
        M     = 1                                 # we fit one UE antenna at a time
        I     = self.num_subcarriers
        N     = self.max_paths

        # ---------------------------------------------------------------------------
        # build S_i(d,ψ_gNB) · D_i(d) · A_UE(ψ_ue,m)   and stack over i
        # ---------------------------------------------------------------------------
        def build_big_S(psi_gnb, psi_ue, d, m_ant):
            blocks = []
            psi_prime = 2.0 / K              # Eq.(3) footnote
            L_const = K * l_gnb * self.wavelength

            for lam in self.wavelengths:     # scalar, float64
                lam = tf.cast(lam, R)

                # ----- gNB spatial sinc -----------------------------------------
                k   = tf.range(K, dtype=R)[:, None]
                psi = tf.constant(psi_gnb, dtype=R)[None, :]
                x   = (L_const * (k * psi_prime - psi)) / lam      # same as  k*l – ψ term
                S_i = (L_const / lam) * tf.experimental.numpy.sinc(x)

                # ----- propagation delay -------------------------------------------
                phase_d = -2.0 * np.pi * tf.constant(d, R) / lam   # (N,)
                D_i     = tf.linalg.diag(tf.exp(1j * tf.cast(phase_d, C)))  # (N,N)

                # ----- UE steering (single antenna m_ant) --------------------------
                phase_ue = -2.0 * np.pi * m_ant * l_ue * tf.constant(psi_ue, R) / lam
                A_ue     = tf.exp(1j * tf.cast(phase_ue, C))[None, :]        # (1,N)

                # stack:  (K,N)·(N,N)→(K,N)  then multiply UE scalar row‑wise
                blocks.append(tf.cast(S_i, C) @ D_i * A_ue)                  # (K,N)

            return tf.concat(blocks, axis=0)                                 # (I*K,N)
        # ---------------------------------------------------------------------------

        all_psi_gnb = np.zeros((N, self.num_ue_antennas))
        all_psi_ue  = np.zeros_like(all_psi_gnb)
        all_d       = np.zeros_like(all_psi_gnb)
        all_a       = np.zeros_like(all_psi_gnb, dtype=np.complex128)

        for m in range(self.num_ue_antennas):

            # ---------- initial guess ------------------------------------------------
            psi0_g = init_psi_gnb[:, m]                # from peak picker
            psi0_u = np.zeros(N)                       # broadside
            d0     = init_d[:, m]

            x0 = np.concatenate([psi0_g, psi0_u, d0])  # length 3N

            P_m = tf.reshape(self.P[:, m], [-1, 1])    # (I*K ,1) complex128

            # ---------- objective function ------------------------------------------
            def obj(x):
                psi_g = x[0:N]
                psi_u = x[N:2*N]
                d     = x[2*N:3*N]

                # hard bounds
                if (np.any(np.abs(psi_g) > 1.0) or
                    np.any(np.abs(psi_u) > 1.0) or
                    np.any(d < 0.0) or np.any(d > 100.0)):
                    return 1e20

                S_big = build_big_S(psi_g, psi_u, d, m_ant=m)

                # closed‑form least squares for a
                a_ls = tf.linalg.lstsq(S_big, P_m, fast=False)     # (N,1)
                res = tf.norm(P_m - S_big @ a_ls)**2
                return res.numpy().astype(np.float64)

            bounds = [(-1.0, 1.0)] * N + \
                    [(-1.0, 1.0)] * N + \
                    [(0.0, 100.0)] * N

            res = minimize(obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'disp': True})

            # ---------- save fitted parameters --------------------------------------
            psi_g_opt = res.x[0:N]
            psi_u_opt = res.x[N:2*N]
            d_opt     = res.x[2*N:3*N]

            S_big_opt = build_big_S(psi_g_opt, psi_u_opt, d_opt, m_ant=m)
            a_opt     = tf.squeeze(tf.linalg.lstsq(S_big_opt, P_m, fast=False))  # (N,)

            all_psi_gnb[:, m] = psi_g_opt
            all_psi_ue[:,  m] = psi_u_opt
            all_d      [:, m] = d_opt
            all_a      [:, m] = a_opt.numpy()
        
        debug = True
        if debug:
            m_dbg = 0
            psi_g = all_psi_gnb[:, m_dbg]
            psi_u = all_psi_ue[:, m_dbg]
            d_g   = all_d[:, m_dbg]
            a_g   = all_a[:, m_dbg][:, None]             # (N,1) complex128

            # --- reconstruct full channel -----------------------------------------
            h_hat = []
            for i, lam in enumerate(self.wavelengths):
                F_i  = self.fourier_matrix(K, l_gnb, lam)   # uses fractional λ spacing
                blk  = build_big_S(psi_g, psi_u, d_g, m_ant=m_dbg)
                SiDi = blk[i*K:(i+1)*K, :]                  # pick sub‑carrier i
                h_i  = tf.cast(F_i, C) @ (SiDi @ a_g)       # (K,1)
                h_hat.append(tf.squeeze(h_i))
            h_hat = tf.stack(h_hat, axis=-1)                # (K , I)

            plt.figure()
            plt.plot(np.real(h_hat[0, :]), label='reconstructed')
            plt.plot(np.real(channel_mean[0, m_dbg, :]), label='original')
            plt.legend(); plt.title(f'UE antenna {m_dbg}')
            plt.savefig('debug_h.png')

            rel = tf.abs(tf.norm(channel_mean[:, m_dbg, :] - tf.cast(h_hat, tf.complex64)) / \
                tf.norm(channel_mean[:, m_dbg, :]))
            tf.print("relative L2 error (dB):", 20*tf.math.log(rel)/tf.math.log(10.0))

        return all_psi_gnb, all_d, all_a

    def compute_steering_vectors(self, num_antennas, antenna_spacing, wavelengths, psi):
        """Compute steering vectors for all paths and subcarriers.
        
        Args:
            num_antennas: Number of antennas (int).
            antenna_spacing: Antenna spacing in wavelengths (float).
            wavelengths: Wavelengths for each subcarrier [subcarriers].
            psi: Angle parameters (sin(theta)) for each path [max_paths].
        
        Returns:
            Steering vectors [num_antennas, max_paths, subcarriers].
        """
        indices = tf.range(num_antennas, dtype=tf.float32)[:, None, None]  # [num_antennas, 1, 1]
        psi = psi[None, :, None]  # [1, max_paths, 1]
        wavelengths = wavelengths[None, None, :]  # [1, 1, subcarriers]
        phase = -2 * np.pi * indices * antenna_spacing * psi / wavelengths
        a = tf.complex(tf.cos(phase), tf.sin(phase))  # [num_antennas, max_paths, subcarriers]
        return a
    
    def reconstruct_uplink_channel(self, psi_gnb, psi_ue, d, a, phi, carrier_freq):
        """Reconstruct uplink channel for the given carrier frequency, predicting 1 ms into the future.
        
        Args:
            psi_gnb: gNB angle parameters [max_paths].
            psi_ue: UE angle parameters [max_paths].
            d: Path distances in meters [max_paths].
            a: Path amplitudes [max_paths].
            phi: Path phases [max_paths].
            carrier_freq: Carrier frequency in Hz (float).
        
        Returns:
            Uplink channel tensor [num_gnb_antennas, num_ue_antennas, subcarriers] at t + 1 ms.
        """
        # Compute wavelengths and frequencies for all subcarriers
        wavelengths = self.speed_of_light / (carrier_freq + self.subcarrier_spacing * tf.range(self.subcarriers, dtype=tf.float32))
        freqs = carrier_freq + self.subcarrier_spacing * tf.range(self.subcarriers, dtype=tf.float32)
        
        # Compute phase at t + time_gap (1 ms)
        time_future = self.time_gap  # 0.001 s
        tau = d / self.speed_of_light  # Path delays [max_paths]
        phase_freq = -2 * np.pi * tau[None, :] * freqs[:, None]  # Base phase due to delay [subcarriers, max_paths]
        doppler_phase = -2 * np.pi * tau[None, :] * freqs[:, None] * time_future / self.coherence_time  # Approximate Doppler effect
        total_phase = phase_freq + doppler_phase + phi[None, :]  # Total phase including future shift
        amplitude_freq = a[None, :] * tf.exp(tf.complex(0.0, tf.cast(total_phase, tf.float32)))  # [subcarriers, max_paths]
        
        # Compute steering vectors
        a_gnb_all = self.compute_steering_vectors(self.num_gnb_antennas, self.antenna_spacing_gnb, wavelengths, psi_gnb)  # [num_gnb_antennas, max_paths, subcarriers]
        a_ue_all = self.compute_steering_vectors(self.num_ue_antennas, self.antenna_spacing_ue, wavelengths, psi_ue)    # [num_ue_antennas, max_paths, subcarriers]
        
        # Reconstruct channel using einsum
        h = tf.einsum('gnk,unk,kn->guk', a_gnb_all, tf.math.conj(a_ue_all), amplitude_freq)  # [num_gnb_antennas, num_ue_antennas, subcarriers]
        return h

    def __call__(self, precoding_channel):
        """Adapt uplink channel to downlink channel for TDD."""
        batch_size, num_rx, num_rx_ant, _, num_tx_ant, num_syms, nfft = precoding_channel.shape
        self.num_ues = num_tx_ant // self.num_ue_antennas  # Infer num_ues from total UE antennas

        # Reshape to [batch_size, num_rx, num_rx_ant, num_ues, num_ue_antennas, num_syms, nfft]
        precoding_channel = tf.reshape(
            precoding_channel,
            [batch_size, num_rx, num_rx_ant, self.num_ues, self.num_ue_antennas, num_syms, nfft]
        )

        downlink_channels = []
        
        for b in range(batch_size):
            batch_channels = []
            for r in range(num_rx):
                ue_channels = []
                for ue_idx in range(self.num_ues):
                    # Extract downlink channel for this UE: [num_gnb_antennas, num_ue_antennas, num_syms, nfft]
                    channel_ue = precoding_channel[b, r, :, ue_idx, :, :, :]  # [4, 2, num_syms, nfft]

                    psi_gnb, psi_ue, d, a, phi = self.estimate_paths_for_ue(channel_ue)
                    uplink_channel = self.reconstruct_uplink_channel(psi_gnb, psi_ue, d, a, phi, self.carrier_freq)
                    downlink_channel = tf.transpose(uplink_channel, [1, 0, 2])  # [2, 4, subcarriers]
                    downlink_channels.append(downlink_channel)
        
        precoding_channel = tf.stack(downlink_channels, axis=0)
        precoding_channel = tf.transpose(precoding_channel[None, None, ...], perm=[0, 1, 4, 2, 3, 5])
        precoding_channel = tf.expand_dims(precoding_channel, axis=5)
        precoding_channel = tf.tile(precoding_channel, multiples=[1, 1, 1, 1, 1, 14, 1])
        precoding_channel = tf.reshape(precoding_channel, [batch_size, num_rx, num_rx_ant, -1, num_tx_ant, num_syms, nfft])
        
        return precoding_channel