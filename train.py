# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from indrnn_convattention import build_model
import datetime

input_data = tf.placeholder(tf.int32, shape=(None, None, 2)) # (B, T, c + s)
features =   tf.placeholder(tf.int32, shape=(None, None)) # (B, N_f)

c_loss, f_loss, c_acc, f_acc, prs = build_model(input_data, features, DICT_SIZE=30, TIME_STEPS=100)

LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 100000

global_step = tf.get_variable("global_step", shape=[], trainable=False,
                              initializer=tf.zeros_initializer, dtype=tf.int32)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_INIT, global_step,
    LEARNING_RATE_DECAY_STEPS, 0.1,
    staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

gradients, variables = zip(*optimizer.compute_gradients(c_loss + f_loss))
clipped_grads, _ = tf.clip_by_global_norm(gradients, 5.)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=global_step)

merged = tf.summary.merge([
    tf.summary.scalar('coarse_loss', c_loss),
    tf.summary.scalar('fine_loss', f_loss),
    tf.summary.scalar('total_loss', c_loss + f_loss)])

accuracies = tf.summary.merge([
    tf.summary.scalar('coarse_accuracy', c_acc),
    tf.summary.scalar('fine_accuracy', f_acc),
    tf.summary.histogram('distribution', prs)])

def gen_batch(BATCH_SIZE=32, SEN_LEN=10):
    mock_sentence = np.random.randint(0, 30, size=(BATCH_SIZE, SEN_LEN))
    mock_samples_c = np.repeat(mock_sentence, 10, axis=1)
    mock_samples_f = mock_samples_c + np.tile(np.repeat(np.expand_dims(np.arange(10), 0), SEN_LEN, axis=0).flatten(), (BATCH_SIZE, 1))
    return mock_sentence, np.stack([mock_samples_c, mock_samples_f], 2)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto()) as sess:

    train_writer = tf.summary.FileWriter('../train_logs/toy_clipped_grad_and_runits', sess.graph)

    sess.run([tf.global_variables_initializer()])
    if input('Restore Model? ') == 'y':
        saver.restore(sess, input('Model Path: '))
        print('Restored. Resuming from iteration', global_step.eval())

    start_time = datetime.datetime.now()
    for _ in range(1000000):
        mock_sentence, mock_data = gen_batch()
        feed_dict = {input_data.name: mock_data,
                     features.name: mock_sentence}

        _, iteration, Lc, Lf, summary = sess.run(
            [train_op, global_step, c_loss, f_loss, merged],
            feed_dict)

        train_writer.add_summary(summary, iteration)

        if iteration % 10 == 0:
            print(iteration, Lc, Lf, Lc + Lf)

        if iteration % 100 == 0:
            elapsed = datetime.datetime.now() - start_time

            print("evaluating...")
            mock_sentence, mock_data = gen_batch()
            feed_dict = {input_data.name: mock_data,
                         features.name: mock_sentence}
            acc_summ, c, f = sess.run([accuracies, c_acc, f_acc], feed_dict)
            train_writer.add_summary(acc_summ, iteration)
            print("coarse acc", c, "fine acc", f)

            print(elapsed / 100 / mock_data.shape[0], "per data point")

            save_path = saver.save(sess, '../models/myrnn', iteration)
            print('Saved to', save_path)

            start_time = datetime.datetime.now()


