# -*- coding: utf-8 -*-
import tensorflow as tf
from ind_rnn_cell import IndRNNCell

class DecoderCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, att_cell, c_cell, f_cell):
         self._att_cell = att_cell
         self._c_cell = c_cell
         self._f_cell = f_cell
    
    @property
    def state_size(self):
         return (self._att_cell.state_size,
                 self._c_cell.state_size, 
                 self._f_cell.state_size)

    @property
    def output_size(self):
         return 2 * 256

    def call(self, inputs, state):
         att_out, att_state = self._att_cell(inputs, state[0])
         c_out, c_state = self._c_cell(att_out, state[1])
         att_out += tf.multinomial(c_out, 1)
         f_out, f_state = self._f_cell(att_out, state[2])
         return tf.concat([c_out, f_out], 2), (att_state, c_state, f_state)
         
         
class WaveRNN:
def __init__(input_data, # (B, T, 2) uint8
             valid_samp_cnt,
             features, # (B, N_f) uint8
             BATCH_SIZE,
             DICT_SIZE,
             TIME_STEPS,
             NUM_UNITS = 256):

    # (B, T, coarse + fine)
    input_norm = tf.scalar_mul(1 / 256, tf.cast(input_data, tf.float32))
    # input data is now in [0, 1]

    # c_t-1 . f_t-1
    cf_tm1 = input_norm[:, :-1]

    # one-hot encode the features
    features_1h = tf.one_hot(features, DICT_SIZE) # (B, N_f, DICT_SIZE)

    # Regulate each neuron's recurrent weight as recommended in the indRNN paper
    # TODO: try the gradient punishing method used in improved gan
    RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

    # Sentence Encoder RNN
    encoder_cell = tf.contrib.rnn.MultiRNNCell(
           [IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX),
            tf.contrib.rnn.ResidualWrapper(IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX))])

    enc_init_state = encoder_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          encoder_cell,
          features_1h,
          initial_state=enc_init_state,
          dtype=tf.float32)    

    # Attention RNN
    att_cell = tf.contrib.seq2seq.AttentionWrapper(
          tf.contrib.rnn.MultiRNNCell(
              [IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX),
               tf.contrib.rnn.ResidualWrapper(IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX))]
          ),
          tf.contrib.seq2seq.BahdanauMonotonicAttention(NUM_UNITS, encoder_outputs),
          attention_layer_size=NUM_UNITS)

    att_init_state = att_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

    attention, att_state = tf.nn.dynamic_rnn(
          att_cell, 
          cf_tm1, 
          initial_state=att_init_state, 
          dtype=tf.float32)

    # rnn cell for predicting coarse
    c_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX, name="cell00")),
             tf.contrib.rnn.ResidualWrapper(IndRNNCell(256, recurrent_max_abs=RECURRENT_MAX, name="cell01"))])
    
    c_init_state = c_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

    coarse_logits, c_state = tf.nn.dynamic_rnn(
          c_cell, 
          attention, 
          initial_state=c_init_state,
          dtype=tf.float32)

    # rnn cell for predicting fine
    f_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.ResidualWrapper(IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX, name="cell10")),
             tf.contrib.rnn.ResidualWrapper(IndRNNCell(256, recurrent_max_abs=RECURRENT_MAX, name="cell11"))])

    # c_t-1 + f_t-1 + c_t
    f_input = attention + input_norm[:, 1:, :1]

    f_init_state = f_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

    fine_logits, f_state = tf.nn.dynamic_rnn(
          f_cell, 
          f_input, 
          initial_state=f_init_state,
          dtype=tf.float32)

    # Decoder
    init_states = (enc_init_state, att_init_state, c_init_state, f_init_state)
    #decoder = tf.contrib.seq2seq.BasicDecoder(
    #      decode_cell,
    #      helper,
    #      init_states)
    #decode = tf.nn.dynamic_decode(decoder)

    # Calculate loss and accuracy
    c_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_data[:, 1:, 0], logits=coarse_logits)) / valid_samp_cnt

    c_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(
        input_data[:, 1:, 0],
        tf.argmax(coarse_logits, axis=2, output_type=tf.int32)
        ), tf.float32))

    f_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_data[:, 1:, 1], logits=fine_logits)) / valid_samp_cnt

    f_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(
        input_data[:, 1:, 1],
        tf.argmax(fine_logits, axis=2, output_type=tf.int32)
        ), tf.float32))

    # probability distribution for introspection
    prs = tf.concat([tf.nn.softmax(fine_logits), tf.nn.softmax(coarse_logits)], 2)

    return((c_loss, f_loss), 
           (c_accuracy, f_accuracy), 
           (enc_init_state, att_init_state, c_init_state, f_init_state), 
           (encoder_state, att_state, c_state, f_state), 
           prs)

