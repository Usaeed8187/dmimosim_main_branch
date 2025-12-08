"""
LMMSE and linear interpolation for dMIMO channels
"""

import tensorflow as tf
import numpy as np
import itertools

from sionna.utils import flatten_last_dims, expand_to_rank
from sionna.ofdm import BaseChannelInterpolator, LinearInterpolator, PilotPattern


class RBwiseLinearInterp(LinearInterpolator):
    r"""RBwiseLinearInterpolator(pilot_pattern, rb_size)

    Linear interpolation of channel estimates on a resource grid
    resource block by resource block.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    rb_size : int
        The size of a resource block in subcarriers

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern: PilotPattern, rb_size: int):
        super().__init__(pilot_pattern)

        assert rb_size > 0, "`rb_size` must be a positive integer"

        self._rb_size = rb_size
        # Sionna's documentation states that:
        # # # # # Linear interpolation works as follows:
        # # # # # We compute for each resource element (RE)
        # # # # # x_0 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        # # # # #       the first channel measurement was taken
        # # # # # x_1 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        # # # # #       the second channel measurement was taken
        # # # # # y_0 : The first channel estimate
        # # # # # y_1 : The second channel estimate
        # # # # # x   : The x-value (i.e., sub-carrier index or OFDM symbol)
        # # # # #
        # # # # # The linearly interpolated value y is then given as:
        # # # # # y = (x-x_0) * (y_1-y_0) / (x_1-x_0) + y_0
        # To dive deeper, here is some info about the various quantities computed by Sionna:
        # self._x_freq is of shape [B=1, M=1, Mt=1, N=1, Nt=1, nS=1, num_eff_subcarriers]
        # self._x_freq is simply tf.range(0, num_eff_subcarriers)
        # self._x_freq is used "x" in the above formula
        # self._x_0_freq and self._x_1_freq are of shape [B=1, M=1, Mt=1, N, Nt, nS, num_eff_subcarriers]
        # where N is num_tx, Nt is num_streams_per_tx and nS is num_ofdm_symbols (usually 14)
        # self._x_0_freq and self._x_1_freq are used as "x_0" and "x_1" in the above formula
        # self._x_0_freq and self._x_1_freq contain, for each resource element, the subcarrier 
        # index of the closest pilots to them from left and right in the frequency domain
        # respectively. If the resource element does not have any pilot to its left, for example,
        # the index of the closest pilot to its right would be used for self._x_0_freq and
        # the index of the second closest pilot to its right would be used for self._x_1_freq for that RE.
        # If there were no pilot to the right, the closest pilot index to the left would be used
        # as self.x_1_freq and the second closest pilot index to the left would be used as self.x_0_freq.
        # This is done to ensure that all resource elements have two pilots to interpolate from.
        # # Note: on the REs that do not contain pilots (or nulls),
        # # self._x_0_freq and self._x_1_freq are set to -1 
        # # #
        # We now look into self._y_0_freq_ind and self._y_1_freq_ind which are of shape [N, Nt, nS, num_eff_subcarriers]
        # Essentially, self._y_0_freq_ind and self._y_1_freq_ind are the self._x_0_freq + 1 and self._x_1_freq + 1 and
        # of dtype int. They are used to gather the channel estimates at the pilot locations (y_0 and y_1 in the formula above)
        # # # # # # # 
        # What we need to do to make this RB-wise is to modify self._x_0_freq and self._x_1_freq as well 
        # as self._y_0_freq_ind and self._y_1_freq_ind such that the pilots from other RBs are not considered
        x_0_freq = np.asarray(self._x_0_freq).real.astype(np.int32) # shape: [1,1,1,N,Nt,nS,num_eff_subcarriers]
        x_1_freq = np.asarray(self._x_1_freq).real.astype(np.int32) # shape: [1,1,1,N,Nt,nS,num_eff_subcarriers]
        y_0_freq_ind = (self._y_0_freq_ind).copy() # shape: [N,Nt,nS,num_eff_subcarriers]
        y_1_freq_ind = (self._y_1_freq_ind).copy() # shape: [N,Nt,nS,num_eff_subcarriers]
        num_eff_subcarriers = x_0_freq.shape[-1]
        # num_ofdm_symbols = x_0_freq.shape[-2]
        # num_tx_streams = x_0_freq.shape[-3]
        # num_tx = x_0_freq.shape[-4]
        x_0_freq_flat  = np.reshape(x_0_freq, (-1, num_eff_subcarriers))
        x_1_freq_flat  = np.reshape(x_1_freq, (-1, num_eff_subcarriers))
        y_0_freq_ind_flat  = np.reshape(y_0_freq_ind, (-1, num_eff_subcarriers))
        y_1_freq_ind_flat  = np.reshape(y_1_freq_ind, (-1, num_eff_subcarriers))

        B = x_0_freq_flat.shape[0]
        num_rbs = num_eff_subcarriers // self._rb_size
        for i_b in range(B):
            for rb_idx in range(num_rbs):
                rb_starting_sc = rb_idx * rb_size
                rb_ending_sc = rb_starting_sc + rb_size
                if rb_idx == num_rbs - 1:
                    rb_ending_sc = num_eff_subcarriers
                if np.all(x_0_freq_flat[i_b, rb_starting_sc:rb_ending_sc] == -1):
                    pass  # No pilots in this RB, skip
                else:
                    if np.any(x_0_freq_flat[i_b, rb_starting_sc:rb_ending_sc] == -1):
                        raise NotImplementedError("RB-wise linear interpolation not implemented for RBs with partial pilots")
                    unique_pilot_indices_in_this_rb = \
                        np.unique(np.concatenate([x_0_freq_flat[i_b, rb_starting_sc:rb_ending_sc] , 
                                                x_1_freq_flat[i_b, rb_starting_sc:rb_ending_sc]]))
                    valid_pilot_indices_in_this_rb = [idx for idx in unique_pilot_indices_in_this_rb if (rb_starting_sc <= idx < rb_ending_sc)]
                    if len(valid_pilot_indices_in_this_rb) == 0:
                        raise Exception(f"RBwiseLinearInterp: No pilots available in RB {rb_idx} for interpolation. Maybe increase the RB size?")
                    if not np.all( (rb_starting_sc <= unique_pilot_indices_in_this_rb) & (unique_pilot_indices_in_this_rb < rb_ending_sc) ):
                        if len(valid_pilot_indices_in_this_rb) == 1:
                            first_pilot_idx = valid_pilot_indices_in_this_rb[0]
                            second_pilot_idx = valid_pilot_indices_in_this_rb[0]
                            last_pilot_idx = valid_pilot_indices_in_this_rb[0]
                            second_last_pilot_idx = valid_pilot_indices_in_this_rb[0]
                        else:
                            first_pilot_idx = valid_pilot_indices_in_this_rb[0]
                            second_pilot_idx = valid_pilot_indices_in_this_rb[1]
                            last_pilot_idx = valid_pilot_indices_in_this_rb[-1]
                            second_last_pilot_idx = valid_pilot_indices_in_this_rb[-2]
                        
                        # Some pilots from other RBs are being used for interpolation in this RB
                        # We need to fix that
                        for i_sc in range(rb_starting_sc, rb_ending_sc):
                            if x_0_freq_flat[i_b, i_sc] < rb_starting_sc:
                                # x_0_freq is from left of the RB
                                x_0_freq_flat[i_b, i_sc] = first_pilot_idx
                                x_1_freq_flat[i_b, i_sc] = second_pilot_idx
                                y_0_freq_ind_flat[i_b, i_sc] = x_0_freq_flat[i_b, i_sc] + 1
                                y_1_freq_ind_flat[i_b, i_sc] = x_1_freq_flat[i_b, i_sc] + 1
                            if x_1_freq_flat[i_b, i_sc] >= rb_ending_sc:
                                # x_1_freq is from right of the RB
                                x_1_freq_flat[i_b, i_sc] = last_pilot_idx
                                x_0_freq_flat[i_b, i_sc] = second_last_pilot_idx
                                y_1_freq_ind_flat[i_b, i_sc] = x_1_freq_flat[i_b, i_sc] + 1
                                y_0_freq_ind_flat[i_b, i_sc] = x_0_freq_flat[i_b, i_sc] + 1
        
        x_0_freq = np.reshape(x_0_freq_flat, x_0_freq.shape).astype(self._x_0_freq.dtype.as_numpy_dtype)
        x_1_freq = np.reshape(x_1_freq_flat, x_1_freq.shape).astype(self._x_1_freq.dtype.as_numpy_dtype)
        y_0_freq_ind = np.reshape(y_0_freq_ind_flat, y_0_freq_ind.shape)
        y_1_freq_ind = np.reshape(y_1_freq_ind_flat, y_1_freq_ind.shape)
        self._x_0_freq = tf.constant(x_0_freq)
        self._x_1_freq = tf.constant(x_1_freq)
        self._y_0_freq_ind = y_0_freq_ind
        self._y_1_freq_ind = y_1_freq_ind
                            
                        

# Adapted from sionna.ofdm.channel_estimation
class LinearInterp1D(BaseChannelInterpolator):
    r"""LinearInterpolator(pilot_pattern)

    Linear channel estimate interpolation on a resource grid.

    The interpolation is done across OFDM symbols.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams
    """
    def __init__(self, pilot_pattern):
        super().__init__()

        assert(pilot_pattern.num_pilot_symbols > 0),\
            """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)
        mask_shape = mask.shape  # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))
        assert max_num_zero_pilots < pilots.shape[-1], \
            """Each pilot sequence must have at least one nonzero entry"""

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        # Linear interpolation works as follows:
        # We compute for each resource element (RE)
        # x_0 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the first channel measurement was taken
        # x_1 : The x-value (i.e., sub-carrier index or OFDM symbol) at which
        #       the second channel measurement was taken
        # y_0 : The first channel estimate
        # y_1 : The second channel estimate
        # x   : The x-value (i.e., sub-carrier index or OFDM symbol)
        #
        # The linearly interpolated value y is then given as:
        # y = (x-x_0) * (y_1-y_0) / (x_1-x_0) + y_0
        #
        # The following code pre-computes various quantities and indices
        # that are needed to compute x_0, x_1, y_0, y_1, x for frequency- and
        # time-domain interpolation.

        ##
        ## Time-domain interpolation
        ##
        self._x_time = tf.expand_dims(tf.range(0, mask.shape[-2]), -1)
        self._x_time = tf.cast(expand_to_rank(self._x_time, 7, axis=0), dtype=pilots.dtype)

        # Indices used to gather estimates
        self._perm_fwd_time = tf.roll(tf.range(7), -3, 0)  # [3, 4, 5, 6, 0, 1, 2]

        # Undo permutation of batch_dims for gather
        self._perm_bwd = tf.roll(tf.range(7), 3, 0)  # [4, 5, 6, 0, 1, 2, 3]

        y_0_time_ind = np.zeros(z.shape[:2], np.int32)  # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int32)  # Gather indices

        # For each stream
        for a in range(z.shape[0]):

            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind) == 1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind) >= 2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a, i] = ofdm_ind[x0]
                    y_1_time_ind[a, i] = ofdm_ind[x1]
                    if i == ofdm_ind[x1] and x1 < len(ofdm_ind)-1:
                        x0 = x1
                        x1 += 1

        self._y_0_time_ind = np.reshape(y_0_time_ind, mask_shape[:-1])
        self._y_1_time_ind = np.reshape(y_1_time_ind, mask_shape[:-1])

        self._x_0_time = expand_to_rank(tf.expand_dims(self._y_0_time_ind, -1), 7, axis=0)
        self._x_0_time = tf.cast(self._x_0_time, dtype=pilots.dtype)
        self._x_1_time = expand_to_rank(tf.expand_dims(self._y_1_time_ind, -1), 7, axis=0)
        self._x_1_time = tf.cast(self._x_1_time, dtype=pilots.dtype)

    def _interpolate_1d(self, inputs, x, x0, x1, y0_ind, y1_ind):
        # Gather the right values for y0 and y1
        y0 = tf.gather(inputs, y0_ind, axis=2, batch_dims=2)
        y1 = tf.gather(inputs, y1_ind, axis=2, batch_dims=2)

        # Undo the permutation of the inputs
        y0 = tf.transpose(y0, self._perm_bwd)
        y1 = tf.transpose(y1, self._perm_bwd)

        # Compute linear interpolation
        slope = tf.math.divide_no_nan(y1-y0, tf.cast(x1-x0, dtype=y0.dtype))
        return tf.cast(x-x0, dtype=y0.dtype)*slope + y0

    def _interpolate(self, inputs):
        #
        # Prepare inputs
        #
        # inputs has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]

        # Transpose h_hat_freq to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers, k, l, m]
        h_hat_time = tf.transpose(inputs, self._perm_fwd_time)

        # h_hat_time has shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        h_hat_time = self._interpolate_1d(h_hat_time,
                                          self._x_time,
                                          self._x_0_time,
                                          self._x_1_time,
                                          self._y_0_time_ind,
                                          self._y_1_time_ind)

        return h_hat_time

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)

        # the interpolator requires complex-valued inputs
        err_var = tf.cast(err_var, tf.complex64)
        err_var = self._interpolate(err_var)
        err_var = tf.math.real(err_var)

        return h_hat, err_var


# Copy from sionna.ofdm.channel_estimation because it is not exported
class LMMSEInterpolator1D:
    r"""LMMSEInterpolator1D(pilot_mask, cov_mat)

    This class performs the linear interpolation across the inner dimension of the input ``h_hat``.

    Parameters
    ----------
    pilot_mask : [:math:`N`, :math:`M`] : int
        Mask indicating the allocation of resource elements.
        0 : Data,
        1 : Pilot,
        2 : Not used,

    cov_mat : [:math:`M`, :math:`M`], tf.complex
        Covariance matrix of the channel across the inner dimension.

    last_step : bool
        Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimates.

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimation error variances.

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, :math:`N`, :math:`M`], tf.complex
        Channel estimates interpolated across the inner dimension.

    err_var : Same shape as ``h_hat``, tf.float
        The channel estimation error variances of the interpolated channel estimates.
    """

    def __init__(self, pilot_mask, cov_mat, last_step):

        self._cdtype = cov_mat.dtype
        assert self._cdtype in (tf.complex64, tf.complex128),\
            "`cov_mat` dtype must be one of tf.complex64 or tf.complex128"
        self._rdtype = self._cdtype.real_dtype
        self._rzero = tf.constant(0.0, self._rdtype)

        # Interpolation is performed along the inner dimension of
        # the resource grid, which may be either the subcarriers
        # or the OFDM symbols dimension.
        # This dimension is referred to as the inner dimension.
        # The other dimension of the resource grid is referred to
        # as the outer dimension.

        # Size of the inner dimension.
        inner_dim_size = tf.shape(pilot_mask)[-1]
        self._inner_dim_size = inner_dim_size

        # Size of the outer dimension.
        outer_dim_size = tf.shape(pilot_mask)[-2]
        self._outer_dim_size = outer_dim_size

        self._cov_mat = cov_mat
        self._last_step = last_step

        # Computation of the interpolation matrix is done solving the least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        # Extracting the locations of pilots from the pilot mask
        num_tx = tf.shape(pilot_mask)[0]
        num_streams_per_tx = tf.shape(pilot_mask)[1]

        # List of indices of pilots in the inner dimension for every
        # transmit antenna, stream, and outer dimension element.
        pilot_indices = []
        # Maximum number of pilots carried by an inner dimension.
        max_num_pil = 0
        # Indices used to add the error variance to the diagonal
        # elements of the covariance matrix restricted
        # to the elements carrying pilots.
        # These matrices are computed below.
        add_err_var_indices = np.zeros([num_tx, num_streams_per_tx,
                                        outer_dim_size, inner_dim_size, 5], int)
        for tx in range(num_tx):
            pilot_indices.append([])
            for st in range(num_streams_per_tx):
                pilot_indices[-1].append([])
                for oi in range(outer_dim_size):
                    pilot_indices[-1][-1].append([])
                    num_pil = 0  # Number of pilots on this outer dim
                    for ii in range(inner_dim_size):
                        # Check if this RE is carrying a pilot
                        # for this stream
                        if pilot_mask[tx,st,oi,ii] == 0:
                            continue
                        if pilot_mask[tx,st,oi,ii] == 1:
                            pilot_indices[tx][st][oi].append(ii)
                            indices = [tx, st, oi, num_pil, num_pil]
                            add_err_var_indices[tx, st, oi, ii] = indices
                            num_pil += 1
                    max_num_pil = max(max_num_pil, num_pil)
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, 5]
        self._add_err_var_indices = tf.cast(add_err_var_indices, tf.int32)

        # Different subcarriers/symbols may carry a different number of pilots.
        # To handle such cases, we create a tensor of square matrices of
        # size the maximum number of pilots carried by an inner dimension
        # and zero-padding is used to handle axes with less pilots than the
        # maximum value. The obtained structure is:
        #
        # |B 0|
        # |0 0|
        #
        pil_cov_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                                max_num_pil, max_num_pil], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            tmp = np.take(cov_mat, pil_ind, axis=0)
            pil_cov_mat_ = np.take(tmp, pil_ind, axis=1)
            pil_cov_mat[tx,st,oi,:num_pil,:num_pil] = pil_cov_mat_
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        self._pil_cov_mat = tf.constant(pil_cov_mat, self._cdtype)

        # Pre-compute the covariance matrix with only the columns corresponding
        # to pilots.
        b_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            b_mat_ = np.take(cov_mat, pil_ind, axis=0)
            b_mat[tx,st,oi,:num_pil,:] = b_mat_
        self._b_mat = tf.constant(b_mat, self._cdtype)

        # Indices used to fill with zeros the columns of the interpolation
        # matrix not corresponding to zeros.
        # The result is a matrix of size inner_dim_size x inner_dim_size
        # where rows and columns not corresponding to pilots are set to zero.
        pil_loc = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                            inner_dim_size, max_num_pil, 5], dtype=int)
        for tx,st,oi,p,ii in itertools.product(range(num_tx),
                                                range(num_streams_per_tx),
                                                range(outer_dim_size),
                                                range(max_num_pil),
                                                range(inner_dim_size)):
            if p >= len(pilot_indices[tx][st][oi]):
                # An extra dummy subcarrier is added to push there padding
                # identity matrix
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              inner_dim_size,
                                              inner_dim_size]
            else:
                pil_loc[tx, st, oi, ii, p] = [tx, st, oi,
                                              ii,
                                              pilot_indices[tx][st][oi][p]]
        self._pil_loc = tf.cast(pil_loc, tf.int32)

        # Covariance matrix for each stream with only the row corresponding
        # to a pilot carrying RE not set to 0.
        # This is required to compute the estimation error variances.
        err_var_mat = np.zeros([num_tx, num_streams_per_tx, outer_dim_size,
                inner_dim_size, inner_dim_size], complex)
        for tx,st,oi in itertools.product(range(num_tx),
                                          range(num_streams_per_tx),
                                          range(outer_dim_size)):
            pil_ind = pilot_indices[tx][st][oi]
            mask = np.zeros([inner_dim_size], complex)
            mask[pil_ind] = 1.0
            mask = np.expand_dims(mask, axis=1)
            err_var_mat[tx,st,oi] = cov_mat*mask
        self._err_var_mat = tf.constant(err_var_mat, self._cdtype)

    def __call__(self, h_hat, err_var):

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx,
        #          num_streams_per_tx, outer_dim_size, inner_dim_size]

        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        outer_dim_size = self._outer_dim_size
        inner_dim_size = self._inner_dim_size

        #####################################
        # Compute the interpolation matrix
        #####################################

        # Computation of the interpolation matrix is done solving the
        # least-square problem:
        #
        # X = min_Z |AZ - B|_F^2
        #
        # where A = (\Pi_T R \Pi + S) and
        # B = R \Pi
        # where R is the channel covariance matrix, S the error
        # diagonal covariance matrix, and \Pi the matrix that spreads the pilots
        # according to the pilot pattern along the inner axis.

        #
        # Computing A
        #

        # Covariance matrices restricted to pilot locations
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = self._pil_cov_mat

        # Adding batch, receive, and receive antennas dimensions to the
        # covariance matrices restricted to pilot locations and to the
        # regularization values
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = expand_to_rank(pil_cov_mat, 8, 0)
        pil_cov_mat = tf.tile(pil_cov_mat, [batch_size, num_rx, num_rx_ant, 1, 1, 1, 1, 1])

        # Adding the noise variance to the covariance matrices restricted to
        # pilots
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat_ = tf.transpose(pil_cov_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        err_var_ = tf.complex(err_var, self._rzero)
        err_var_ = tf.transpose(err_var_, [3, 4, 5, 6, 0, 1, 2])
        a_mat = tf.tensor_scatter_nd_add(pil_cov_mat_, self._add_err_var_indices, err_var_)
        a_mat = tf.transpose(a_mat, [5, 6, 7, 0, 1, 2, 3, 4])

        #
        # Computing B
        #

        # B is pre-computed as it only depend on the channel covariance and
        # pilot pattern.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = self._b_mat
        b_mat = expand_to_rank(b_mat, 8, 0)
        b_mat = tf.tile(b_mat, [batch_size, num_rx, num_rx_ant,
                                1, 1, 1, 1, 1])

        #
        # Computing the interpolation matrix
        #

        # Using lstsq to compute the columns of the interpolation matrix
        # corresponding to pilots.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, max_num_pil]
        ext_mat = tf.linalg.lstsq(a_mat, b_mat, fast=False)
        ext_mat = tf.transpose(ext_mat, [0,1,2,3,4,5,7,6], conjugate=True)

        # Filling with zeros the columns not corresponding to pilots.
        # An extra dummy outer dim is added to scatter there the coefficients
        # of the identity matrix used for padding.
        # This dummy dim is then removed.
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, inner_dim_size]
        ext_mat = tf.transpose(ext_mat, [3, 4, 5, 6, 7, 0, 1, 2])
        ext_mat = tf.scatter_nd(self._pil_loc, ext_mat,
                                            [num_tx, num_tx_stream,
                                             outer_dim_size,
                                             inner_dim_size+1,
                                             inner_dim_size+1,
                                             batch_size, num_rx, num_rx_ant])
        ext_mat = tf.transpose(ext_mat, [5, 6, 7, 0, 1, 2, 3, 4])
        ext_mat = ext_mat[...,:inner_dim_size,:inner_dim_size]

        ################################################
        # Apply interpolation over the inner dimension
        ################################################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        h_hat = tf.expand_dims(h_hat, axis=-1)
        h_hat = tf.matmul(ext_mat, h_hat)
        h_hat = tf.squeeze(h_hat, axis=-1)

        ##############################
        # Compute the error variances
        ##############################

        # Keep track of the previous estimation error variances for later use
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        err_var_old = err_var

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        cov_mat = expand_to_rank(self._cov_mat, 8, 0)
        err_var = tf.linalg.diag_part(cov_mat)
        err_var_mat = expand_to_rank(self._err_var_mat, 8, 0)
        err_var_mat = tf.transpose(err_var_mat, [0, 1, 2, 3, 4, 5, 7, 6])
        err_var = err_var - tf.reduce_sum(ext_mat*err_var_mat, axis=-1)
        err_var = tf.math.real(err_var)
        err_var = tf.maximum(err_var, self._rzero)

        #####################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        #####################################
        if not self._last_step:
            #
            # Variance of h_hat
            #
            # Conjugate transpose of LMMSE matrix
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            ext_mat_h = tf.transpose(ext_mat, [0, 1, 2, 3, 4, 5, 7, 6],
                                     conjugate=True)
            # First part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            h_hat_var_1 = tf.matmul(cov_mat, ext_mat_h)
            h_hat_var_1 = tf.transpose(h_hat_var_1, [0, 1, 2, 3, 4, 5, 7, 6])
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var_1 = tf.reduce_sum(ext_mat*h_hat_var_1, axis=-1)
            # Second part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_old_c = tf.complex(err_var_old, self._rzero)
            err_var_old_c = tf.expand_dims(err_var_old_c, axis=-1)
            h_hat_var_2 = err_var_old_c*ext_mat_h
            h_hat_var_2 = tf.transpose(h_hat_var_2, [0, 1, 2, 3, 4, 5, 7, 6])
            h_hat_var_2 = tf.reduce_sum(ext_mat*h_hat_var_2, axis=-1)
            # Variance of h_hat
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var = h_hat_var_1 + h_hat_var_2
            # Scaling factor
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_c = tf.complex(err_var, self._rzero)
            h_var = tf.linalg.diag_part(cov_mat)
            s = tf.math.divide_no_nan(2.*h_var, h_hat_var + h_var - err_var_c)
            # Apply scaling to estimate
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat = s*h_hat
            # Updated variance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var = s*(s-1.)*h_hat_var + (1.-s)*h_var + s*err_var_c
            err_var = tf.math.real(err_var)
            err_var = tf.maximum(err_var, self._rzero)

        return h_hat, err_var


# Adapted from sionna.ofdm.LMMSEInterpolator
class LMMSELinearInterp(BaseChannelInterpolator):
    # pylint: disable=line-too-long
    r"""LMMSELinearInterpolator(pilot_pattern, cov_mat_time, cov_mat_freq, cov_mat_space=None, order='t-f')

    LMMSE interpolation in frequency direction and linear interpolation in time direction.

    This class computes for each element of an OFDM resource grid a channel estimate and error variance
    through linear minimum mean square error (LMMSE) interpolation/smoothing across subcarriers and
    linear interpolation across OFDM symbols.

    It is assumed that the measurements were taken at the nonzero positions of a :class:`~sionna.ofdm.PilotPattern`.

    Note
    ----
    This layer does not support graph mode with XLA.

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    cov_mat_freq : [fft_size, fft_size], tf.complex
        Frequency covariance matrix of the channel

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, pilot_pattern, cov_mat_freq):

        self._num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
        self._num_effective_subcarriers =pilot_pattern.num_effective_subcarriers

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask(pilot_pattern)

        # Build indices for mapping channel estimates and error variances
        # that are given as input to a resource grid
        num_pilots = pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices = self._build_inputs2rg_indices(pilot_mask, num_pilots)
        self._inputs_to_rg_indices = tf.cast(inputs_to_rg_indices, tf.int32)

        # Frequency-domain LMMSE interpolator
        self._freq_interp = LMMSEInterpolator1D(pilot_mask, cov_mat_freq, last_step=False)
        pilot_mask = self._update_pilot_mask_interp(pilot_mask)
        self._freq_err_var_mask = tf.cast(pilot_mask == 1, cov_mat_freq.dtype.real_dtype)

        # Time-domain linear interpolator
        pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
        self._time_interp = LinearInterp1D(pilot_pattern)
        pilot_mask = self._update_pilot_mask_interp(pilot_mask)
        pilot_mask = tf.transpose(pilot_mask, [0, 1, 3, 2])
        self._time_err_var_mask = tf.cast(pilot_mask == 1, cov_mat_freq.dtype.real_dtype)

    def _build_pilot_mask(self, pilot_pattern):
        """
        Build for every transmitter and stream a pilot mask indicating
        which REs are allocated to pilots, data, or not used.
        # 0 -> Data
        # 1 -> Pilot
        # 2 -> Not used
        """
        mask = np.array(pilot_pattern.mask)
        pilots = np.array(pilot_pattern.pilots)
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros([num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], int)
        for tx,st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0
            for sb,sc in itertools.product(range(num_ofdm_symbols), range(num_effective_subcarriers)):
                if mask[tx,st,sb,sc] == 1:
                    if np.abs(pilots[tx,st,pil_index]) > 0.0:
                        pilot_mask[tx,st,sb,sc] = 1
                    else:
                        pilot_mask[tx,st,sb,sc] = 2
                    pil_index += 1

        return pilot_mask

    def _build_inputs2rg_indices(self, pilot_mask, num_pilots):
        """
        Builds indices for mapping channel estimates and error variances
        that are given as input to a resource grid
        """

        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros([num_tx, num_streams_per_tx, num_pilots, 4], int)
        for tx,st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0 # Pilot index for this stream
            for sb,sc in itertools.product(range(num_ofdm_symbols), range(num_effective_subcarriers)):
                if pilot_mask[tx,st,sb,sc] == 0:
                    continue
                if pilot_mask[tx,st,sb,sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                pil_index += 1

        return inputs_to_rg_indices

    def _update_pilot_mask_interp(self, pilot_mask):
        """
        Update the pilot mask to label the resource elements for which the
        channel was interpolated.
        """

        interpolated = np.any(pilot_mask == 1, axis=-1, keepdims=True)
        pilot_mask = np.where(interpolated, 1, pilot_mask)

        return pilot_mask

    def __call__(self, h_hat, err_var):

        # h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilots]
        # err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilots]

        batch_size = tf.shape(h_hat)[0]
        num_rx = tf.shape(h_hat)[1]
        num_rx_ant = tf.shape(h_hat)[2]
        num_tx = tf.shape(h_hat)[3]
        num_tx_stream = tf.shape(h_hat)[4]
        num_ofdm_symbols = self._num_ofdm_symbols
        num_effective_subcarriers = self._num_effective_subcarriers

        # For some estimator, err_var might not have the same shape as h_hat
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))

        # Mapping the channel estimates and error variances to a resource grid
        # all : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #           num_ofdm_symbols, num_effective_subcarriers]
        h_hat = tf.transpose(h_hat, [3, 4, 5, 0, 1, 2])
        err_var = tf.transpose(err_var, [3, 4, 5, 0, 1, 2])
        h_hat = tf.scatter_nd(self._inputs_to_rg_indices, h_hat,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        err_var = tf.scatter_nd(self._inputs_to_rg_indices, err_var,
                                            [num_tx, num_tx_stream,
                                             num_ofdm_symbols,
                                             num_effective_subcarriers,
                                             batch_size, num_rx, num_rx_ant])
        h_hat = tf.transpose(h_hat, [4, 5, 6, 0, 1, 2, 3])
        err_var = tf.transpose(err_var, [4, 5, 6, 0, 1, 2, 3])

        # Frequency-domain LMMSE interpolation
        # h_hat has shape [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #                  num_ofdm_symbols, num_effective_subcarriers]
        h_hat, err_var = self._freq_interp(h_hat, err_var)
        err_var_mask = expand_to_rank(self._freq_err_var_mask, tf.rank(err_var), 0)
        err_var = err_var*err_var_mask

        # Time-domain linear interpolation
        # h_hat has shape [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #                  num_ofdm_symbols, num_effective_subcarriers]
        h_hat, err_var = self._time_interp(h_hat, err_var)
        err_var_mask = expand_to_rank(self._time_err_var_mask, tf.rank(err_var), 0)
        err_var = err_var*err_var_mask

        return h_hat, err_var

