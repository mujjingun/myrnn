import tensorflow as tf
from attention_cell import AttentionCell
from ind_cat_cell import IndCatCell
from sample_cell import SampleCell

# (B, T, coarse + fine)
input_data = tf.placeholder(tf.uint8, shape=(None, None, 2))
input_norm = tf.scalar_mul(1 / 256, tf.cast(input_data, tf.float32))
# input data is now in [0, 1]

# Regulate each neuron's recurrent weight as recommended in the indRNN paper
TIME_STEPS = 100000 # TODO: adjust this
recurrent_max = pow(2, 1 / TIME_STEPS)

DICT_SIZE = 80
features = tf.placeholder(tf.uint8, shape = (None, None)) # (B, N_f)
features_1h = tf.one_hot(features, DICT_SIZE) # (B, N_f, DICT_SIZE)

cell = tf.contrib.rnn.MultiRNNCell(
        [AttentionCell(32, features_1h, recurrent_max),
         IndCatCell(256, recurrent_max),
         SampleCell(recurrent_max)])
output, state = tf.nn.dynamic_rnn(cell, input_norm, dtype=tf.float32)

# output: output over time
# state: final state