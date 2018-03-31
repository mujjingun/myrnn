# -*- coding: utf-8 -*-
import tensorflow as tf
from attention_cell import AttentionCell
from ind_cat_cell import IndCatCell
from ind_rnn_cell import IndRNNCell
from sample_cell import SampleCell

def build_model(input_data, # (B, T, 2) uint8
                features, # (B, N_f) uint8
                DICT_SIZE = 30,
                TIME_STEPS = 1000):
    # (B, T, coarse + fine)
    input_norm = tf.scalar_mul(1 / 256, tf.cast(input_data, tf.float32))
    # input data is now in [0, 1]

    # c_t-1 . f_t-1
    cf_tm1 = input_norm[:, :-1]

    # one-hot encode the features
    #features_1h = tf.one_hot(features, DICT_SIZE) # (B, N_f, DICT_SIZE)

    # Regulate each neuron's recurrent weight as recommended in the indRNN paper
    # TODO: try the gradient punishing method used in improved gan
    RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

    first_input_init = tf.random_uniform_initializer(-RECURRENT_MAX, RECURRENT_MAX)

    # rnn cell for predicting coarse
    cell = tf.contrib.rnn.MultiRNNCell(
            [#AttentionCell(features_1h, recurrent_max),
             IndRNNCell(128, recurrent_max_abs=RECURRENT_MAX, name="cell00", recurrent_kernel_initializer=first_input_init),
             IndRNNCell(256, recurrent_max_abs=RECURRENT_MAX, name="cell01")])
    coarse_logits, state1 = tf.nn.dynamic_rnn(cell, cf_tm1, dtype=tf.float32)

    # rnn cell for predicting fine
    cell2 = tf.contrib.rnn.MultiRNNCell(
            [IndRNNCell(128, recurrent_max_abs=RECURRENT_MAX, name="cell10", recurrent_kernel_initializer=first_input_init),
             IndRNNCell(256, recurrent_max_abs=RECURRENT_MAX, name="cell11")])

    # c_t-1 + f_t-1 + c_t
    cell2_input = tf.concat([cf_tm1, input_norm[:, 1:, :1]], 2)
    fine_logits, state2 = tf.nn.dynamic_rnn(cell2, cell2_input, dtype=tf.float32)

    # TODO: Original paper uses MSE
    c_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_data[:, 1:, 0], logits=coarse_logits))

    f_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=input_data[:, 1:, 1], logits=fine_logits))

    return c_loss, f_loss