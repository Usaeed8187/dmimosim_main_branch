import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .phase_3_sic_lmmse_decoding import phase_3_sic_lmmse_decoding


class phase_3_sic_lmmse_decoder(Layer):

    def __init__(self,
                 rg, 
                 sm,
                 architecture,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self.rg = rg
        self.sm = sm

        self.architecture = architecture
        self.num_BS_Ant = 4
        self.num_UE_Ant = 2

    def call(self, inputs):
        
        if len(inputs) == 4:
            y_rg, h_hat, MU_MIMO_RG_populated, noise_var = inputs
        else:
            ValueError("calling phase 3 uplink decoder with incorrect params")

        # Decoding
        x_hat, no_eff = phase_3_sic_lmmse_decoding(y_rg, h_hat, MU_MIMO_RG_populated, self.rg, noise_var)

        return x_hat, no_eff
