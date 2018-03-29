# -*- coding: utf-8 -*-
import tensorflow as tf
from ind_rnn_cell import IndRNNCell
from tensorflow.python.ops import rnn_cell_impl

# an IndRNN cell with its input concatted on the output
class IndCatCell(rnn_cell_impl._LayerRNNCell):
    def __init__(self, num_units, recurrent_max_abs):
        super(IndCatCell, self).__init__()
        self._indrnn = IndRNNCell(
            num_units,
            recurrent_max_abs=recurrent_max_abs)
            
    @property
    def state_size(self):
        return self._indrnn.state_size
        
    @property
    def output_size(self):
        return self._indrnn.output_size
    
    def build(self, inputs_shape):
        self._indrnn.build(inputs_shape)
        
    def __call__(self, inputs, state, scope=None):
        out, state = self._indrnn(inputs, state, scope)
        out = tf.concat([out, inputs[:, -2:]], 1)
        return out, state