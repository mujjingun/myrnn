# -*- coding: utf-8 -*-
import tensorflow as tf
from ind_rnn_cell import IndRNNCell
from tensorflow.python.ops import rnn_cell_impl

def batchwise_conv(seq, filt):
    # seq: (B, seq_len, chan)
    # filt: (B, filt_len, in_chan, out_chan)
    
    BATCH_SIZE = tf.shape(filt)[0]
    FILT_LEN = tf.shape(filt)[1]
    FILT_IN = tf.shape(filt)[2]
    FILT_OUT = tf.shape(filt)[3]
    
    filt = tf.transpose(filt, (1, 0, 2, 3))
    # filt is now (width, batch, in_chan, out_chan)
    filt = tf.reshape(filt, 
        (FILT_LEN, 1, BATCH_SIZE*FILT_IN, FILT_OUT))
    # filt is now (width, height=1, batch*in_chan, out_chan)
    
    BATCH_SIZE = tf.shape(seq)[0]
    FEAT_LEN = tf.shape(seq)[1]
    FEAT_CHAN = tf.shape(seq)[2]
    
    features = tf.transpose(seq, (1, 0, 2))
    # features is now (width, batch, channels)
    features = tf.reshape(features, 
        (1, FEAT_LEN, 1, BATCH_SIZE*FEAT_CHAN))
    # features is now (1, width, height=1, batch*channels)
    
    # convolve different filter for each instance of batch
    conv = tf.nn.depthwise_conv2d(
        features,
        filt,
        strides=[1, 1, 1, 1],
        padding="SAME")
    # conv is (1, width, height=1, batch*channels*out_channels)
    
    conv = tf.reshape(conv, 
        (FEAT_LEN, BATCH_SIZE, FEAT_CHAN, FILT_OUT))    
    # conv is now (width, batch, channels, out_channels)
    conv = tf.transpose(conv, [1, 0, 2, 3])
    # conv is now (batch, width, channels, out_channels)
    conv = tf.reduce_sum(conv, axis=2)
    # conv is now (batch, width, out_channels)
    
    return conv
        

class AttentionCell(rnn_cell_impl._LayerRNNCell):
    def __init__(self, num_units, features, recurrent_max_abs):
        super(AttentionCell, self).__init__()
        
        self._num_units = num_units
        self._in_channels = features.get_shape()[2].value # DICT_SIZE
        self._features = features
        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=tf.zeros_initializer())
        self._filt_shape = (-1, 10, self._in_channels, num_units)
        self._indrnn = IndRNNCell(
            self._filt_shape[1] * self._in_channels * num_units,
            recurrent_max_abs=recurrent_max_abs)
        
    @property
    def state_size(self):
        return self._indrnn.state_size
    
    @property
    def output_size(self):
        return self._num_units + 2
        
    def build(self, inputs_shape):
        self._indrnn.build(inputs_shape)
    
    def __call__(self, inputs, state, scope=None):
        filt, new_state = self._indrnn(inputs, state, scope)

        filt = tf.reshape(filt, self._filt_shape)
        # filt has shape (B, width, in_channels, out_channels)
        
        conv = batchwise_conv(self._features, filt)
        # conv has shape (B, width, out_channels)
        
        conv = tf.nn.relu(conv + self._bias)
        
        # TODO: try squashing or normalizing with softmax, etc
        output = tf.reduce_mean(conv, 1) # (B, out_size)

        output = tf.concat([output, inputs], 1) # (B, out_size + 2)
        
        return output, new_state
