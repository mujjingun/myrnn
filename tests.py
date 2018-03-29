# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import test

import attention_cell
from sample_cell import SampleCell
from ind_cat_cell import IndCatCell

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
            filt = tf.placeholder(tf.float32, shape=(2, 3, 1, 2))
            seq = tf.placeholder(tf.float32, shape=(2, 10, 1))
            
            conv = attention_cell.batchwise_conv(seq, filt)
            result = sess.run([conv],
                     {filt.name: np.array([[[[0, 1]], [[2, 3]], [[-1, 0]]],
                                            [[[1, 0]], [[3, 2]], [[0, -1]]]]),
                      seq.name: np.array([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
                                          [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]])})
            print(result)
        
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
            init_state = tf.placeholder(tf.float32, shape=(1, 1600))
            
            cell = attention_cell.AttentionCell(32, features, recurrent_max)
            out, _ = cell(x, init_state)
            
            sess.run([tf.global_variables_initializer()])
            result = sess.run([out], 
                     {x.name: np.array([[0.5, 0.1]]),
                      features.name: np.random.normal(size=(1, 7, 5)),
                      init_state.name: np.random.normal(size=(1, 1600))})
            print(result)

tests = ModelTest()