import numpy as np
import wave, os, struct, linecache

def gen_batch(BATCH_SIZE, lineno=1):
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
