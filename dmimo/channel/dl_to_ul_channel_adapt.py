import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class dl_to_ul_channel_adapt:
    def __init__(self, carrier_freq=3.5e9, antenna_spacing_ue=0.5, antenna_spacing_gnb=0.5, coherence_time=0.01, time_gap=1e-3, max_paths=2, num_ues=10, subcarrier_spacing=15e3):
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

    def fourier_matrix(self, num_antennas, antenna_spacing, wavelength):
        
        K = num_antennas
        indices = tf.range(K, dtype=tf.float32)
        psi_prime = tf.range(K, dtype=tf.float32) * (wavelength / (K * antenna_spacing * wavelength))  # ψ' = j' * λ/(K l)
        phase = -2 * np.pi * indices[:, None] * antenna_spacing * wavelength * psi_prime[None, :] / wavelength
        F = tf.complex(tf.cos(phase), tf.sin(phase)) / tf.complex(tf.sqrt(tf.cast(K, tf.float32)), 0.0)
        
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
        """Initialize psi_n^gNB, psi_n^UE, and d_n using power profile P(d, psi^gNB, psi^UE)."""
        P = tf.zeros((100, len(self.psi_grid), len(self.psi_grid)), dtype=tf.float32)
        d_grid = np.linspace(0, 100, 100)  # Distance grid in meters
        psi_grid_ue = self.psi_grid
        for i, lambda_i in enumerate(wavelengths):
            h_i = channel_data[:, :, i]  # [4, 2]
            for k in range(self.num_gnb_antennas):
                for m in range(self.num_ue_antennas):
                    phase = 2 * np.pi * (
                        d_grid[:, None, None] +
                        k * self.antenna_spacing_gnb * self.wavelength * self.psi_grid[None, :, None] +
                        m * self.antenna_spacing_ue * self.wavelength * psi_grid_ue[None, None, :]
                    ) / lambda_i
                    P += tf.abs(tf.reduce_sum(h_i[k, m] * tf.exp(tf.complex(0.0, tf.cast(phase, tf.float32))), axis=-1)) ** 2
        P = P.numpy()
        indices = np.unravel_index(np.argsort(P.ravel())[-self.max_paths:], P.shape)
        init_d = d_grid[indices[0]]
        init_psi_gnb = self.psi_grid[indices[1]]
        init_psi_ue = self.psi_grid[indices[2]]
        return init_psi_gnb, init_psi_ue, init_d

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
        self.P = tf.stack(P_list, axis=-1)  # [self.num_gnb_antennas * self.num_subcarriers, 2]

        # Initialize parameters
        init_psi_gnb, init_psi_ue, init_d = self.initialize_paths(channel_mean, self.wavelengths)
        init_params = np.concatenate([
            init_psi_gnb, init_psi_ue, init_d,
            np.ones(self.max_paths) * 0.1, np.zeros(self.max_paths)
        ])

        def objective(params):
            """
            Compute the L2 norm error between P and S * a.
            """
            psi_gnb = params[:self.max_paths]  # Path angles
            psi_ue = params[self.max_paths:2 * self.max_paths]
            d = params[2 * self.max_paths:3 * self.max_paths]  # Delays
            a = params[3 * self.max_paths:4 * self.max_paths]  # Amplitudes
            phi = params[4 * self.max_paths:]  # Phases

            psi_gnb = np.clip(psi_gnb, -1, 1)
            psi_ue = np.clip(psi_ue, -1, 1)
            d = np.clip(d, 0, 100)
            a = np.clip(a, 0, 1)
            phi = np.clip(phi, -np.pi, np.pi)

            error = 0.0
            # Assuming P is provided externally with shape [I * K, 2]
            P = self.P

            for m in range(self.num_ue_antennas):
                S_concat = []
                for i, lambda_i in enumerate(self.wavelengths):
                    # Compute S_i;
                    S_i = self.sinc_matrix(lambda_i, psi_gnb, L=self.num_gnb_antennas * self.antenna_spacing_gnb * self.wavelength, psi_prime=2/self.num_gnb_antennas)  # [K, N]
                    # Compute D_i
                    D_i = tf.linalg.diag(tf.exp(tf.complex(0.0, -2 * np.pi * d / lambda_i)))  # [N, N]
                    # Compute S_i * D_i
                    S_i_D_i = tf.matmul(S_i, tf.cast(D_i, S_i.dtype))  # [K, N]
                    S_concat.append(S_i_D_i)
                # Concatenate across subcarriers
                S_concat = tf.concat(S_concat, axis=0)  # [I * K, N]

                # Compute a vector
                a_complex = a * tf.exp(tf.complex(0.0, tf.cast(phi, tf.float32)))  # [N]
                a_vec = tf.cast(a_complex, tf.complex64)[:, None]  # [N, 1]

                # Predicted profile: S * a
                P_pred = tf.matmul(tf.complex(S_concat, 0.0), a_vec)  # [I * K, 1]

                # L2 norm error
                error += tf.reduce_sum(tf.abs(P[:, m:m+1] - P_pred) ** 2).numpy()
            return error

        result = minimize(objective, init_params, method='SLSQP', bounds=[
            (-1, 1)] * self.max_paths + [(-1, 1)] * self.max_paths + [(0, 100)] * self.max_paths +
            [(0, 1)] * self.max_paths + [(-np.pi, np.pi)] * self.max_paths)
        
        psi_gnb = result.x[:self.max_paths]
        psi_ue = result.x[self.max_paths:2*self.max_paths]
        d = result.x[2*self.max_paths:3*self.max_paths]
        a = result.x[3*self.max_paths:4*self.max_paths]
        phi = result.x[4*self.max_paths:]

        return psi_gnb, psi_ue, d, a, phi

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
        """Reconstruct uplink channel for the given carrier frequency.
        
        Args:
            psi_gnb: gNB angle parameters [max_paths].
            psi_ue: UE angle parameters [max_paths].
            d: Path distances in meters [max_paths].
            a: Path amplitudes [max_paths].
            phi: Path phases [max_paths].
            carrier_freq: Carrier frequency in Hz (float).
        
        Returns:
            Uplink channel tensor [num_gnb_antennas, num_ue_antennas, subcarriers].
        """
        wavelengths = self.speed_of_light / (carrier_freq + self.subcarrier_spacing * tf.range(self.subcarriers, dtype=tf.float32))
        freqs = carrier_freq + self.subcarrier_spacing * tf.range(self.subcarriers, dtype=tf.float32)
        
        phase_freq = -2 * np.pi * d[None, :] * freqs[:, None] / self.speed_of_light
        amplitude_freq = a[None, :] * tf.exp(tf.complex(0.0, tf.cast(phase_freq + phi[None, :], tf.float32)))  # [subcarriers, max_paths]
        
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