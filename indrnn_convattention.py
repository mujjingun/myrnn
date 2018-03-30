# -*- coding: utf-8 -*-
import tensorflow as tf
from attention_cell import AttentionCell
from ind_cat_cell import IndCatCell
from sample_cell import SampleCell

#TODO: adjust TIME_STEPS
def build_model(input_data, # (B, T, 2) uint8
                features, # (B, N_f) uint8
                DICT_SIZE = 80,
                TIME_STEPS = 100000):
    # (B, T, coarse + fine)
    input_norm = tf.scalar_mul(1 / 256, tf.cast(input_data, tf.float32))
    # input data is now in [0, 1]

    # one-hot encode the features
    features_1h = tf.one_hot(features, DICT_SIZE) # (B, N_f, DICT_SIZE)

    # Regulate each neuron's recurrent weight as recommended in the indRNN paper
    recurrent_max = pow(2, 1 / TIME_STEPS)

    cell = tf.contrib.rnn.MultiRNNCell(
            [AttentionCell(features_1h, recurrent_max),
             IndCatCell(300, recurrent_max),
             SampleCell(recurrent_max)])
    output, state = tf.nn.dynamic_rnn(cell, input_norm, dtype=tf.float32)
    # output: output over time (B, T, 2 * 256)
    # state: final state
    coarse_logits, fine_logits = tf.split(output, [256, 256], 2)

    input_data = tf.cast(input_data, tf.int32)
    coarse = input_data[:, 1:, 0]
    fine   = input_data[:, 1:, 1]

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=coarse, logits=coarse_logits[:, :-1]))

    loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=fine, logits=fine_logits[:, :-1]))

    return loss