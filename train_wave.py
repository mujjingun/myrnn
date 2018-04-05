# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from indrnn_convattention import WaveRNN
from fetch_audio import gen_batch
import datetime

# Build Model.

input_data = tf.placeholder(tf.int32, shape=(None, None, 2)) # (B, T, c + s)
features   = tf.placeholder(tf.int32, shape=(None, None)) # (B, N_f)
valid_samp_cnt = tf.placeholder(tf.float32, shape=[])

TIME_STEPS = 1000
DICT_SIZE  = 30

rnn = WaveRNN(input_data, valid_samp_cnt, features,
              DICT_SIZE=DICT_SIZE,
              TIME_STEPS=TIME_STEPS)

# Build Training Ops

BATCH_SIZE = 8
LEARNING_RATE_INIT = 1e-6
LEARNING_RATE_DECAY_STEPS = 100000

global_step = tf.get_variable("global_step", shape=[], trainable=False,
                              initializer=tf.zeros_initializer, dtype=tf.int32)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_INIT, global_step,
    LEARNING_RATE_DECAY_STEPS, 0.1,
    staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

total_loss = rnn.c_loss + rnn.f_loss
gradients, variables = zip(*optimizer.compute_gradients(total_loss))
clipped_grads, _ = tf.clip_by_global_norm(gradients, 2.)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=global_step)

merged = tf.summary.merge([
    tf.summary.scalar('coarse_loss', rnn.c_loss),
    tf.summary.scalar('fine_loss', rnn.f_loss),
    tf.summary.scalar('total_loss', total_loss)])

acc_summary = tf.summary.merge([
    tf.summary.scalar('coarse_accuracy', rnn.c_accuracy),
    tf.summary.scalar('fine_accuracy', rnn.f_accuracy),
    tf.summary.histogram('distribution', rnn.prs)])

def main():
    saver = tf.train.Saver(sharded=True)

    with tf.Session(config=tf.ConfigProto()) as sess:

        train_writer = tf.summary.FileWriter('../train_logs/speak2', sess.graph)

        sess.run([tf.global_variables_initializer()])
        if input('Restore Model? ') == 'y':
            saver.restore(sess, input('Model Path: '))
            print('Restored. Resuming from iteration', global_step.eval())

        if input('Start training? ') != 'y':
            return

        start_time = datetime.datetime.now()
        for sentence, data, s_lens, d_lens in gen_batch(BATCH_SIZE):

            states = None
            for start in range(0, data.shape[1], TIME_STEPS):
                valid_cnt = np.sum(np.clip(d_lens - start, 0, TIME_STEPS))

                feed_dict = {input_data.name: data[:, start:start + TIME_STEPS],
                             features.name: sentence,
                             valid_samp_cnt.name: valid_cnt}
                if states is not None:
                    feed_dict[rnn.init_states] = states

                _, iteration, Lc, Lf, summary, states = sess.run(
                    [train_op, global_step, rnn.c_loss, rnn.f_loss, merged, rnn.new_states],
                    feed_dict)

                train_writer.add_summary(summary, iteration)

                if iteration % 10 == 0:
                    print(iteration, Lc, Lf, Lc + Lf)

                if iteration % 100 == 0:
                    elapsed = datetime.datetime.now() - start_time

                    print(elapsed / 100 / BATCH_SIZE, "per data point")

                    save_path = saver.save(sess, '../models/speak2', iteration)
                    print('Saved to', save_path)

                    start_time = datetime.datetime.now()


if __name__ == "__main__":
    main()
