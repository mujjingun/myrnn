# -*- coding: utf-8 -*-
import tensorflow as tf
from ind_rnn_cell import IndRNNCell

class SampleCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, recurrent_max_abs):
        super(SampleCell, self).__init__()
        self._indrnn = IndRNNCell(
            256,
            recurrent_max_abs=recurrent_max_abs)

    @property
    def state_size(self):
        return self._indrnn.state_size

    @property
    def output_size(self):
        return 2 * 256

    def build(self, inputs_shape):
        self._indrnn.build(inputs_shape)

    def __call__(self, inputs, state, scope=None):
        # inputs: (B, 256 + 2)
        logits, x = tf.split(inputs, (256, 2), 1)
        coarse = tf.multinomial(logits, 1, output_dtype=tf.int32) # (B, 1)
        coarse_norm = tf.scalar_mul(1 / 256, tf.cast(coarse, tf.float32))
        cat = tf.concat([coarse_norm, x], 1) # (B, 1 + 2)
        out, new_state = self._indrnn(cat, state, scope)

        #FOR TESTS
        self.cat = cat
        self.coarse_norm = coarse_norm
        #########

        return tf.concat([logits, out], 1), new_state