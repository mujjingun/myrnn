# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from indrnn_convattention import build_model

input_data = tf.placeholder(tf.int32, shape=(None, None, 2)) # (B, T, c + s)
features =   tf.placeholder(tf.int32, shape=(None, None)) # (B, N_f)

c_loss, f_loss, c_acc, f_acc = build_model(input_data, features, DICT_SIZE=30, TIME_STEPS=100)

LEARNING_RATE_INIT = 0.0001
LEARNING_RATE_DECAY_STEPS = 50000

global_step = tf.get_variable("global_step", shape=[], trainable=False,
                              initializer=tf.zeros_initializer)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_INIT, global_step,
    LEARNING_RATE_DECAY_STEPS, 0.1,
    staircase=True)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(c_loss + f_loss, global_step=global_step)

merged = tf.summary.merge([
    tf.summary.scalar('coarse_loss', c_loss),
    tf.summary.scalar('fine_loss', f_loss),
    tf.summary.scalar('total_loss', c_loss + f_loss)])

accuracies = tf.summary.merge([
    tf.summary.scalar('coarse_accuracy', c_acc),
    tf.summary.scalar('fine_accuracy', f_acc)])

def gen_batch(BATCH_SIZE=50, SEN_LEN=10):
    mock_sentence = np.random.randint(0, 30, size=(BATCH_SIZE, SEN_LEN))
    mock_samples_c = np.repeat(mock_sentence, 10, axis=1)
    mock_samples_f = mock_samples_c + np.tile(np.repeat(np.expand_dims(np.arange(10), 0), SEN_LEN, axis=0).flatten(), (BATCH_SIZE, 1))
    return mock_sentence, np.stack([mock_samples_c, mock_samples_f], 2)

with tf.Session(config=tf.ConfigProto()) as sess:

    train_writer = tf.summary.FileWriter('../train_logs', sess.graph)

    sess.run([tf.global_variables_initializer()])

    for iteration in range(1000000):
        mock_sentence, mock_data = gen_batch()
        feed_dict = {input_data.name: mock_data,
                     features.name: mock_sentence}
        _, Lc, Lf, summary = sess.run([train_op, c_loss, f_loss, merged], feed_dict)

        train_writer.add_summary(summary, iteration)

        if iteration % 10 == 0:
            print(iteration, Lc, Lf, Lc + Lf)

        if iteration % 100 == 0:
            print("evaluating...")
            mock_sentence, mock_data = gen_batch()
            feed_dict = {input_data.name: mock_data,
                         features.name: mock_sentence}
            acc_summ, c, f = sess.run([accuracies, c_acc, f_acc], feed_dict)
            train_writer.add_summary(acc_summ, iteration)
            print("coarse acc", c, "fine acc", f)
