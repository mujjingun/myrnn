# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test

import attention_cell
from sample_cell import SampleCell
from ind_cat_cell import IndCatCell
import time

TIME_STEPS = 100000 # TODO: adjust this
recurrent_max = pow(2, 1 / TIME_STEPS)

class ModelTest(test.TestCase):
    def sample_cell_test(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=(1, 2))
            logits = tf.placeholder(tf.float32, shape=(1, 256))
            init_state = tf.placeholder(tf.float32, shape=(1, 256))

            cell = SampleCell(recurrent_max)
            out, _ = cell(tf.concat([logits, x], 1), init_state)

            sess.run([tf.global_variables_initializer()])
            cat, coarse = sess.run([cell.cat, cell.coarse_norm],
                     {x.name: np.array([[0.5, 0.1]]),
                      logits.name: np.random.normal(size=(1, 256)),
                      init_state.name: np.random.normal(size=(1, 256))})

    def batchwise_conv_test(self):
        with self.test_session() as sess:
            filt_shape = (32, 10, 80, 32)
            seq_shape = (32, 100, 80)
            filt = tf.placeholder(tf.float32, shape=filt_shape)
            seq = tf.placeholder(tf.float32, shape=seq_shape)

            conv = attention_cell.batchwise_conv(seq, filt)
            conv2 = attention_cell.batchwise_conv_2(seq, filt)

            filt_d = np.random.normal(size=filt_shape)
            seq_d = np.random.normal(size=seq_shape)

            start_time = time.time()
            result1 = sess.run([conv],
                     {filt.name: filt_d, seq.name: seq_d})
            print(time.time() - start_time)
            start_time = time.time()
            result2 = sess.run([conv2],
                     {filt.name: filt_d, seq.name: seq_d})
            print(time.time() - start_time)
            self.assertAllClose(result1, result2, atol=1e-04)

    def indcat_cell_test(self):
        with self.test_session() as sess:
            prev = tf.placeholder(tf.float32, shape=(1, 32))
            x = tf.placeholder(tf.float32, shape=(1, 2))
            init_state = tf.placeholder(tf.float32, shape=(1, 256))

            cell = IndCatCell(256, recurrent_max)
            out, _ = cell(tf.concat([prev, x], 1), init_state)

            sess.run([tf.global_variables_initializer()])
            result = sess.run([out],
                     {x.name: np.array([[0.5, 0.1]]),
                      prev.name: np.random.normal(size=(1, 32)),
                      init_state.name: np.random.normal(size=(1, 256))})
            print(result)

    def attention_cell_test(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, shape=(1, 2))
            features = tf.placeholder(tf.float32, shape=(1, 7, 5)) # batch, length, dict size
            init_state = tf.placeholder(tf.float32, shape=(1, 800))

            cell = attention_cell.AttentionCell(32, features, recurrent_max)
            out, _ = cell(x, init_state)

            sess.run([tf.global_variables_initializer()])
            result = sess.run([out],
                     {x.name: np.array([[0.5, 0.1]]),
                      features.name: np.random.normal(size=(1, 7, 5)),
                      init_state.name: np.random.normal(size=(1, 800))})
            print(result)

tests = ModelTest()