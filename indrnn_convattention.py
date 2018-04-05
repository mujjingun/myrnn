# -*- coding: utf-8 -*-
import tensorflow as tf
from ind_rnn_cell import IndRNNCell

class DecoderCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, att_cell, c_cell, f_cell):
        super(DecoderCell, self).__init__(_scope=None)
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
        return 2

    def call(self, inputs, state):
        att_out, att_state = self._att_cell(inputs, state[0])

        c_out, c_state = self._c_cell(att_out, state[1])
        c_sample = tf.multinomial(c_out, 1, output_dtype=tf.int32)
        c_sample = tf.scalar_mul(1 / 256, tf.cast(c_sample, tf.float32))

        att_out = tf.concat([att_out, c_sample], 1)

        f_out, f_state = self._f_cell(att_out, state[2])
        f_sample = tf.multinomial(f_out, 1, output_dtype=tf.int32)
        f_sample = tf.scalar_mul(1 / 256, tf.cast(f_sample, tf.float32))

        out = tf.concat([c_sample, f_sample], 1)
        out.set_shape((1, 2))
        return out, (att_state, c_state, f_state)


class WaveRNN:
    def __init__(self,
                 input_data, # (B, T, 2) uint8
                 valid_samp_cnt,
                 features, # (B, N_f) uint8
                 DICT_SIZE,
                 TIME_STEPS,
                 NUM_UNITS = 256):

        BATCH_SIZE = tf.shape(input_data)[0]

        # (B, T, coarse + fine)
        input_norm = tf.scalar_mul(1 / 256, tf.cast(input_data, tf.float32))
        # input data is now in [0, 1]

        # c_t-1 . f_t-1
        cf_tm1 = input_norm[:, :-1]

        # one-hot encode the features
        features_1h = tf.one_hot(features, DICT_SIZE) # (B, N_f, DICT_SIZE)

        # Regulate each neuron's recurrent weight as recommended in the indRNN paper
        # TODO: try the gradient punishing method used in improved gan
        # RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

        def make_cell(num_units, name):
            return tf.contrib.rnn.GRUCell(num_units, name=name)
            #return IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX)

        # Sentence Encoder RNN
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
               [make_cell(NUM_UNITS, "enc0"),
                tf.contrib.rnn.ResidualWrapper(make_cell(NUM_UNITS, "enc1"))])

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
              encoder_cell,
              features_1h,
              initial_state=encoder_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32))

        # Attention RNN
        att_cell = tf.contrib.seq2seq.AttentionWrapper(
              tf.contrib.rnn.MultiRNNCell(
                  [make_cell(NUM_UNITS, "att0"),
                   tf.contrib.rnn.ResidualWrapper(make_cell(NUM_UNITS, "att1"))]
              ),
              tf.contrib.seq2seq.BahdanauMonotonicAttention(NUM_UNITS, encoder_outputs),
              attention_layer_size=NUM_UNITS,
              name="att_cell")

        att_init_state = att_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

        attention, att_state = tf.nn.dynamic_rnn(
              att_cell,
              cf_tm1,
              initial_state=att_init_state,
              dtype=tf.float32)

        # rnn cell for predicting coarse
        c_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.ResidualWrapper(make_cell(NUM_UNITS, "c_cell0")),
                 tf.contrib.rnn.ResidualWrapper(make_cell(256, "c_cell1"))])

        c_init_state = c_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

        coarse_logits, c_state = tf.nn.dynamic_rnn(
              c_cell,
              attention,
              initial_state=c_init_state,
              dtype=tf.float32)

        # rnn cell for predicting fine
        f_cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell(NUM_UNITS, "f_cell0"),
                 tf.contrib.rnn.ResidualWrapper(make_cell(256, "f_cell1"))])

        # c_t-1 + f_t-1 + c_t
        f_input = tf.concat([attention, input_norm[:, 1:, :1]], 2)

        f_init_state = f_cell.zero_state(dtype=tf.float32, batch_size=BATCH_SIZE)

        fine_logits, f_state = tf.nn.dynamic_rnn(
              f_cell,
              f_input,
              initial_state=f_init_state,
              dtype=tf.float32)


        self.init_states = (att_init_state, c_init_state, f_init_state)
        self.new_states = (att_state, c_state, f_state)

        # Decoder
        decode_cell = DecoderCell(att_cell, c_cell, f_cell)

        helper = tf.contrib.seq2seq.CustomHelper(
              initialize_fn=lambda: (False, tf.constant([[0.5, 0.0]])),
              sample_fn=lambda time, outputs, state: tf.constant([0]),
              next_inputs_fn=lambda time, outputs, state, sample_ids: (False, outputs, state))

        decoder = tf.contrib.seq2seq.BasicDecoder(
              decode_cell,
              helper,
              self.init_states)

        self.decode = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=10000)

        # Calculate loss and accuracy
        self.c_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=input_data[:, 1:, 0], logits=coarse_logits)) / valid_samp_cnt

        self.c_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
            input_data[:, 1:, 0],
            tf.argmax(coarse_logits, axis=2, output_type=tf.int32)
            ), tf.float32))

        self.f_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=input_data[:, 1:, 1], logits=fine_logits)) / valid_samp_cnt

        self.f_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
            input_data[:, 1:, 1],
            tf.argmax(fine_logits, axis=2, output_type=tf.int32)
            ), tf.float32))

        # probability distribution for introspection
        self.prs = tf.concat([tf.nn.softmax(fine_logits), tf.nn.softmax(coarse_logits)], 2)
