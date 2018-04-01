# -*- coding: utf-8 -*-
"""Module using IndRNNCell to solve the addition problem

The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should converge
to a MSE around zero after 1000-20000 steps, depending on the number of time
steps.
"""
import tensorflow as tf
import numpy as np

from ind_rnn_cell import IndRNNCell

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 128
NUM_UNITS = 128
LEARNING_RATE_INIT = 2e-4
LEARNING_RATE_DECAY_STEPS = 200000
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
NUM_LAYERS = 4

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50


def main():
  # Placeholders for training data
  inputs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIME_STEPS))

  # Build the graph
  cell = tf.nn.rnn_cell.MultiRNNCell([
     tf.contrib.rnn.DropoutWrapper(
       tf.contrib.rnn.ResidualWrapper(
          IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX),
          lambda i, o: o + tf.pad(i, [[0,0], [0, tf.shape(o)[1] - tf.shape(i)[1]]])
       ), output_keep_prob=0.75
     ) for _ in range(NUM_LAYERS)
  ])
  #cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs

  output, state = tf.nn.dynamic_rnn(cell, tf.expand_dims(inputs_ph, 2), dtype=tf.float32)

  logits = output[:, :-1]
  targets = tf.cast((inputs_ph + 1) / 2 * 127, tf.int32)[:, 1:]
  loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

  kernels = tf.get_collection("recurrent_kernel")
  penalty = sum(tf.reduce_mean(tf.maximum(0.0, (k * (k - RECURRENT_MAX)))) for k in kernels) / len(kernels)

  summary = tf.summary.merge(
      [tf.summary.scalar('loss', loss_op),
       tf.summary.histogram('distribution', tf.nn.softmax(logits)),
       tf.summary.scalar('penalty', penalty)])

  global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                initializer=tf.zeros_initializer)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                             LEARNING_RATE_DECAY_STEPS, 0.1,
                                             staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize = optimizer.minimize(loss_op + 10 * penalty, global_step=global_step)

  # Train the model
  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('../train_logs/correlated_noise', sess.graph)
    sess.run(tf.global_variables_initializer())
    step = 0
    while True:
      losses = []
      for _ in range(100):
        # Generate new input data
        noise = get_batch()
        loss, _, progress = sess.run([loss_op, optimize, summary],
                           {inputs_ph: noise})
        losses.append(loss)
        train_writer.add_summary(progress, step)
        step += 1
      print("Step {} loss {}".format(int(step), np.mean(losses)))

def get_batch():
  """Generate the adding problem dataset"""
  # Build the first sequence
  noise = np.random.normal(size=(BATCH_SIZE, TIME_STEPS)) / 2
  fft = np.fft.rfft(noise) 
  fft[:, TIME_STEPS // 2:] = 0
  noise = np.fft.irfft(fft)
  noise = np.clip(noise, -1, 1)
  return noise

if __name__ == "__main__":
  main()
