# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from indrnn_convattention import build_model
import datetime, wave, os, struct, linecache

input_data = tf.placeholder(tf.int32, shape=(None, None, 2)) # (B, T, c + s)
features   = tf.placeholder(tf.int32, shape=(None, None)) # (B, N_f)
valid_samp_cnt = tf.placeholder(tf.float32, shape=[])

BATCH_SIZE = 8
TIME_STEPS = 1000
DICT_SIZE  = 30

losses, accuracies, init_states, states_op, prs = build_model(
    input_data, valid_samp_cnt, features,
    BATCH_SIZE=BATCH_SIZE,
    DICT_SIZE=DICT_SIZE,
    TIME_STEPS=TIME_STEPS)

c_loss, f_loss = losses
c_acc, f_acc = accuracies

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

gradients, variables = zip(*optimizer.compute_gradients(c_loss + f_loss))
clipped_grads, _ = tf.clip_by_global_norm(gradients, 2.)
with tf.control_dependencies(update_ops):
    train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=global_step)

merged = tf.summary.merge([
    tf.summary.scalar('coarse_loss', c_loss),
    tf.summary.scalar('fine_loss', f_loss),
    tf.summary.scalar('total_loss', c_loss + f_loss)])

acc_summary = tf.summary.merge([
    tf.summary.scalar('coarse_accuracy', c_acc),
    tf.summary.scalar('fine_accuracy', f_acc),
    tf.summary.histogram('distribution', prs)])

def gen_batch(lineno=1):
    WEB_dir = "../data/WEB"
    sentence_batch = []
    voice_batch   = []
    
    while True:
       line = linecache.getline(os.path.join(WEB_dir, "transcript.txt"), lineno)
       if line == '':
           lineno = 1
           continue

       lineno += 1

       filename, sentence, duration = line.split('\t')

       vocab = "abcedfghijklmnopqrstuvwxyz"
       encoded = [vocab.find(ch) for ch in sentence.lower()]
       encoded = np.array([i if i >= 0 else len(vocab) for i in encoded])

       with wave.open(os.path.join(WEB_dir, filename + '.wav'), 'rb') as wavfile:
           num_frames = wavfile.getnframes()
           if wavfile.getnchannels() == 2:
               fmt = '{}h'.format(num_frames * 2)
               raw_data = wavfile.readframes(num_frames)
               voice = np.array(struct.unpack(fmt, raw_data))
               voice = np.reshape(voice, (-1, 2))[:, 0]
           else:
               fmt = '{}h'.format(num_frames)
               raw_data = wavfile.readframes(num_frames)
               voice = np.array(struct.unpack(fmt, raw_data))
               voice = np.reshape(voice, (-1))
       voice += 32768
       voice = np.stack([voice / 256, voice % 256], axis=-1)

       sentence_batch.append(encoded)
       voice_batch.append(voice)

       if len(sentence_batch) % BATCH_SIZE == 0:
           max_sentence_len = max(s.shape[0] for s in sentence_batch)
           orig_sentence_len = [row.shape[0] for row in sentence_batch]
           sentence_batch = [
               np.concatenate(
                    [row, np.zeros((max_sentence_len - row.shape[0]) + len(vocab))]
               ) for row in sentence_batch
           ]
           max_voice_len = max(s.shape[0] for s in voice_batch)
           orig_voice_len = [row.shape[0] for row in voice_batch]
           voice_batch = [
               np.concatenate(
                    [row, np.tile((128, 0), (max_voice_len - row.shape[0], 1))]
               ) for row in voice_batch
           ]
           yield (np.array(sentence_batch), 
                  np.array(voice_batch), 
                  np.array(orig_sentence_len), 
                  np.array(orig_voice_len))

           sentence_batch = []
           voice_batch = []

def main():
    saver = tf.train.Saver(sharded=True)

    with tf.Session(config=tf.ConfigProto()) as sess:

        train_writer = tf.summary.FileWriter('../train_logs/speak2', sess.graph)

        sess.run([tf.global_variables_initializer()])
        if input('Restore Model? ') == 'y':
            saver.restore(sess, input('Model Path: '))
            print('Restored. Resuming from iteration', global_step.eval())

        start_time = datetime.datetime.now() 
        for sentence, data, s_lens, d_lens in gen_batch():
            
            states = None
            for start in range(0, data.shape[1], TIME_STEPS):
                valid_cnt = np.sum(np.clip(d_lens - start, 0, TIME_STEPS))
                feed_dict = {input_data.name: data[:, start:start + TIME_STEPS],
                             features.name: sentence[:, start:start + TIME_STEPS],
                             valid_samp_cnt.name: valid_cnt}
                if states is not None:
                    feed_dict[states_op] = states

                _, iteration, Lc, Lf, summary, states = sess.run(
                    [train_op, global_step, c_loss, f_loss, merged, states_op],
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
