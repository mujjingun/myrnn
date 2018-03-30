# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from indrnn_convattention import build_model

input_data = tf.placeholder(tf.uint8, shape=(None, None, 2)) # (B, T, c + s)
features = tf.placeholder(tf.uint8, shape = (None, None)) # (B, N_f)

loss = build_model(input_data, features)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

BATCH_SIZE = 2

with tf.Session(config=tf.ConfigProto()) as sess:
    sess.run([tf.global_variables_initializer()])

    for iterations in range(1000):
        mock_sentence = np.random.randint(0, 30, size=(BATCH_SIZE, 100))
        mock_samples_c = np.repeat(mock_sentence, 10, axis=1)
        mock_samples_f = mock_samples_c + np.tile(np.repeat(np.expand_dims(np.arange(10), 0), 100, axis=0).flatten(), (BATCH_SIZE, 1))
        _, L = sess.run([train_op, loss],
                 {
                     input_data.name: np.stack([mock_samples_c, mock_samples_f], 2),
                     features.name: mock_sentence
                 })
        print(L)
