import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import LMMSEEqualizer, RemoveNulledSubcarriers
from sionna.utils import matrix_inv, matrix_pinv
from sionna.mapping import Mapper, Demapper


class SICLMMSEEqualizer(Layer):
    """SIC LMMSE Equalizer for Phase 3"""

    def __init__(self,
                 resource_grid,
                 stream_management,
                 modulation_order,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self.rg = resource_grid
        self.sm = stream_management
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)

        self.demapper = Demapper("maxlog", "qam", modulation_order)
        self.mapper = Mapper("qam", modulation_order)


    def call(self, inputs):

        y, h, err_var, no, num_streams_per_tx = inputs
        # y has shape
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        #   
        # We assume that h is the effective channel (product of channel and precoder)

        batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers = h.shape
        
        # Remove nulled subcarriers from y:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers]
        y_effective_scs = self.remove_nulled_scs(y)

        # Transformations to bring h and y in the desired shapes

        # Transpose y:
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]
        y_equalized = tf.transpose(y_effective_scs, [0, 1, 3, 4, 2])
        y_equalized = tf.cast(y_equalized, self._dtype)

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers, batch_size]
        h_eq = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers, batch_size]
        h_eq_desired = tf.gather(h_eq, self.sm.precoding_ind,
                                 axis=1, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers, batch_size]
        h_eq_desired = flatten_dims(h_eq_desired, 2, axis=1)

        # Transpose: #TODO: add support for single stream transmission here
        # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_tx * num_streams_per_tx]
        h_eq_desired = tf.transpose(h_eq_desired, [5, 3, 4, 1, 0, 2])
        h_eq_desired = tf.reshape(h_eq_desired, [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, -1])
        h_eq_desired = tf.cast(h_eq_desired, self._dtype)

        curr_y = y_equalized[..., tf.newaxis]

        num_iterations = num_tx * num_streams_per_tx
        inf_value = tf.complex(tf.constant(float('inf')), tf.constant(float('inf')))
        x_hard_all = tf.zeros((batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_tx, 1), dtype=self._dtype)
        no_eff_all = tf.zeros((batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_tx), dtype=tf.float32)
        streams_all = tf.range(num_tx * num_streams_per_tx)
        for iter_idx in range(num_iterations): #TODO: g not updated. should be updated every loop
            
            # Find LMMSE inverse matrix for current iteration
            _, g, no_eff = self.lmmse_equalization(y_equalized, h_eq_desired, no)

            # Select current UE and stream to detect and cancel interference from
            squared_abs = tf.math.square(tf.math.abs(g))
            sum_squares = tf.reduce_sum(squared_abs, axis=[0,1,2,4])
            g_norms = tf.sqrt(sum_squares)
            min_index = tf.argmin(tf.reshape(g_norms, [-1]), output_type=tf.int32)

            # Slice tensors based on current selection
            curr_no_eff = tf.gather(no_eff, min_index, axis=-1)
            curr_no_eff = curr_no_eff[..., tf.newaxis]
            curr_g = tf.gather(g, min_index, axis=-2)
            curr_g = curr_g[..., tf.newaxis, :]

            # Equalize current stream
            curr_z = tf.matmul(curr_g, curr_y)
            curr_z = tf.squeeze(curr_z, axis=-1)

            # Estimate current stream's symbols
            llr = self.demapper([curr_z, curr_no_eff])
            d_hard = tf.cast(llr > 0, tf.float32)
            x_hard = self.mapper(d_hard)
            x_hard = x_hard[..., tf.newaxis]

            # Save current detected symbols and curr_no_eff
            ue_index = streams_all[min_index] // num_streams_per_tx
            stream_index = streams_all[min_index] % num_streams_per_tx
            # print("currently removing interference of ue {} stream {}".format(ue_index, stream_index))
            streams_all = tf.concat([streams_all[:min_index], streams_all[min_index+1:]], axis=0)
            ue_mask = tf.scatter_nd(indices=[[ue_index]], updates=[True], shape=[num_tx])
            stream_mask = tf.scatter_nd(indices=[[stream_index]], updates=[True], shape=[num_streams_per_tx])
            ue_mask = tf.reshape(ue_mask, [1, -1, 1, 1, 1, 1])  # Shape: [1, num_tx, 1, 1, 1, 1]
            stream_mask = tf.reshape(stream_mask, [1, 1, 1, 1, -1, 1])  # Shape: [1, 1, 1, 1, num_streams_per_tx, 1]
            final_mask = tf.logical_and(ue_mask, stream_mask)  # Shape: [1, num_tx, 1, 1, num_streams_per_tx, 1]
            final_mask = tf.broadcast_to(final_mask, tf.shape(x_hard_all))  # Broadcast to x_hard_all.shape
            x_hard_all = tf.where(final_mask, x_hard, x_hard_all)
            curr_no_eff = curr_no_eff[:, tf.newaxis, ...]
            no_eff_all = tf.where(final_mask[...,0], curr_no_eff, no_eff_all)

            # Remove interference of current stream from y
            curr_h = tf.gather(h_eq_desired, min_index, axis = -1)
            curr_h = curr_h[..., tf.newaxis]
            curr_y = curr_y - tf.matmul(curr_h, x_hard)

            # Remove current column of h_eq_desired
            h_eq_desired = tf.concat(
                [h_eq_desired[..., :min_index],
                h_eq_desired[..., min_index+1:]],
                axis=-1
            )
                  
        # Transpose output to desired shape:
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        x_hard_all = tf.squeeze(x_hard_all, axis=-1)
        x_hard_all = tf.transpose(x_hard_all, [0, 1, 4, 2, 3])

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        no_eff_all = tf.transpose(no_eff_all, [0, 1, 4, 2, 3])

        return x_hard_all, no_eff_all
    
    def lmmse_equalization(self, y_equalized, h_eq_desired, no):

        # LMMSE equalization
        g = tf.matmul(h_eq_desired, h_eq_desired, adjoint_b=True)
        g = g + tf.cast(no, g.dtype) * tf.eye(tf.shape(g)[-1], batch_shape=tf.shape(g)[:-2], dtype=g.dtype)
        g = tf.matmul(h_eq_desired, matrix_inv(g), adjoint_a=True)
        y_equalized = tf.expand_dims(y_equalized, -1)
        gy = tf.squeeze(tf.matmul(g, y_equalized), axis=-1)

        # Compute GH
        gh = tf.matmul(g, h_eq_desired)

        # Compute diag(GH)
        d = tf.linalg.diag_part(gh)

        # Compute x_hat
        x_hat = gy/d

        # Compute no_eff
        one = tf.cast(1, dtype=d.dtype)
        no_eff = tf.math.real(one/d - one)

        return x_hat, g, no_eff

