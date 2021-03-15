import random
import numpy as np
import tensorflow as tf

from helper import get_filenames_from_dir, WordVectorHelper
from helper import CROSS_EMBEDDING_DIR, CE_VEC, \
    WORD2VEC_DIR, W2V_VEC, \
    SPEECH2VEC_DIR, S2V_VEC, EMBEDDING_DIMENSION


LONGEST_SENTENCE_LENGTH = 100
LONGEST_WORD_LENGTH = 50


def filter_labels(pred, gt, sen_length):
    # Inputs:   pred, gt: (bs, max_sen_length)
    #           sen_length (bs)
    # Return new_pred, new_gt: (bs * valid_length, 1)

    batch_size, _ = gt.shape

    new_pred, new_gt = tf.constant([]), tf.constant([])

    for i in range(batch_size):
        s_len = sen_length[i]
        new_pred = tf.concat([new_pred, pred[i, :s_len]], axis=0)
        new_gt = tf.concat([new_gt, gt[i, :s_len]], axis=0)

    return new_pred, new_gt


def load_corpus(dataset_dir, seq_length=100, is_training=True, dataset=None):
    # Read tfrecords data

    if is_training:
        train_set, _, _ = get_filenames_from_dir(dataset_dir)
        set = train_set
    else:
        set = dataset

    batch_data = []
    for file in set:
        # training: file = '<file>' etc.
        # evaluation: file = 'tfrecords/<file>.tfrecords
        path = file if file.endswith('.tfrecords') else (dataset_dir + file + '.tfrecords')

        record_iterator = tf.python_io.tf_record_iterator(path)

        data = []
        for record in record_iterator:
            example = tf.train.Example.FromString(record)

            file_name = (example.features.feature['file_name'].bytes_list.value[0])

            aframe = (example.features.feature['audio_frame'].bytes_list.value[0])
            aframe = np.frombuffer(aframe, np.float32)

            embedding = (example.features.feature['embedding'].bytes_list.value[0])
            embedding = np.frombuffer(embedding)

            label = (example.features.feature['label'].bytes_list.value[0])
            label = np.frombuffer(label, np.float32)

            data.append((aframe, embedding, label, file_name.decode("utf-8")))

        # Create batch of sequence-length frames
        tmp_batch_data = [data[x:x+seq_length] for x in range(0, len(data), seq_length)]

        for batch in tmp_batch_data:
            aframes, embeddings, labels, fn = batch[0]

            for (af, emb, gt, _) in batch[1:]:
                aframes = np.vstack((aframes, af))
                embeddings = np.vstack((embeddings, emb))
                labels = np.vstack((labels, gt))

            # print(aframes.shape, embeddings.shape, labels.shape)
            batch_data.append((aframes, embeddings, labels, aframes.shape[0], fn))

    return batch_data


def load_sent_corpus(dataset_dir, is_training=True, dataset=None):
    # Read tfrecords data

    if is_training:
        train_set, _, _ = get_filenames_from_dir(dataset_dir)
        set = train_set
    else:
        set = dataset

    sentences = []

    max_sent_length, fn = 0, ''

    for file in set:
        # training: file = '<file>' etc.
        # evaluation: file = 'tfrecords/<file>.tfrecords
        path = file if file.endswith('.tfrecords') else (dataset_dir + file + '.tfrecords')

        record_iterator = tf.python_io.tf_record_iterator(path)

        for record in record_iterator:
            example = tf.train.Example.FromString(record)

            file_name = (example.features.feature['file_name'].bytes_list.value[0])

            sent_length = (example.features.feature['sentence_length'].int64_list.value[0])

            aframes = (example.features.feature['audio_frames'].bytes_list.value[0])
            aframes = np.frombuffer(aframes, np.float32).reshape((sent_length, -1))

            words = (example.features.feature['words'].bytes_list.value[0])
            words = np.frombuffer(words).reshape((sent_length, -1))

            labels = (example.features.feature['labels'].bytes_list.value[0])
            labels = np.frombuffer(labels, np.float32).reshape((sent_length, -1))

            sentences.append((aframes, words, labels, sent_length, file_name.decode("utf-8")))

    return sentences


def load_word_corpus(dataset_dir, is_training=True, dataset=None):
    # Read tfrecords data

    if is_training:
        train_set, _, _ = get_filenames_from_dir(dataset_dir)
        set = train_set
    else:
        set = dataset

    words_list = []
    max_word_length, fn = 0, ''

    for file in set:
        # training: file = '<file>' etc.
        # evaluation: file = 'tfrecords/<file>.tfrecords
        path = file if file.endswith('.tfrecords') else (dataset_dir + file + '.tfrecords')

        record_iterator = tf.python_io.tf_record_iterator(path)

        for record in record_iterator:
            example = tf.train.Example.FromString(record)

            file_name = (example.features.feature['file_name'].bytes_list.value[0])

            word_length = (example.features.feature['word_length'].int64_list.value[0])

            aframes = (example.features.feature['audio_frames'].bytes_list.value[0])
            aframes = np.frombuffer(aframes, np.float32).reshape((word_length, -1))

            embeddings = (example.features.feature['embeddings'].bytes_list.value[0])
            embeddings = np.frombuffer(embeddings).reshape((word_length, -1))

            labels = (example.features.feature['labels'].bytes_list.value[0])
            labels = np.frombuffer(labels, np.float32).reshape((word_length, -1))

            # print(aframes.shape, words.shape, labels.shape, sent_length)

            # if word_length > max_word_length:
            #     max_word_length = word_length
            #     emb = embeddings[0]
            #     fn = file_name

            words_list.append((aframes, embeddings, labels, word_length, file_name.decode("utf-8")))

    # print(max_word_length, fn)
    # vec_helper = WordVectorHelper(CROSS_EMBEDDING_DIR + CE_VEC)
    # vec_helper.load_vec()
    # print(vec_helper.get_word_by_embedding(emb))

    return words_list


def pad(a, pad_length):
    if pad_length == 0:
        return a

    return np.pad(a, [(0, pad_length), (0, 0)], mode='constant')


def pad_batch(batch, pad_length):
    new_batch = []
    af_batch, word_batch, label_batch, sent_len_batch = [], [], [], []

    for s in batch:
        aframes, words, labels, l = s
        aframes = pad(aframes, pad_length - l)
        words = pad(words, pad_length - l)
        labels = pad(labels, pad_length - l)

        new_batch.append((aframes, words, labels))
        # print(aframes.shape, words.shape, labels.shape)

        af_batch.append(aframes)
        word_batch.append(words)
        label_batch.append(labels)
        sent_len_batch.append(l)

    return np.asarray(af_batch), np.asarray(word_batch), np.asarray(label_batch), np.asarray(sent_len_batch)


class TrainDataReader(object):
    def __init__(self, dataset_dir, batch_size, seq_length=100, data_unit=''):
        self.data_dir = dataset_dir
        self.batch_size = batch_size
        self.sequence_length = seq_length

        self.data_unit = data_unit

        self.aframes_batches = []
        self.word_emb_batches = []
        self.label_batches = []
        self.data_length_batches = []

        self.epoch = 0

    def make_batches(self):
        if self.data_unit == 'word':
            data = load_word_corpus(dataset_dir=self.data_dir, is_training=True)
            max_data_length = LONGEST_WORD_LENGTH

        elif self.data_unit == 'sentence':
            data = load_sent_corpus(dataset_dir=self.data_dir, is_training=True)
            max_data_length = LONGEST_SENTENCE_LENGTH

        else:
            data = load_corpus(self.data_dir, self.sequence_length, is_training=True)
            max_data_length = self.sequence_length

        batch_size = self.batch_size

        # Shuffle idx in len(data)
        num_data = len(data)
        shuffle_idx = [_ for _ in range(num_data)]
        random.shuffle(shuffle_idx)

        # Create batches
        num_batch = num_data // batch_size + 1
        for i in range(num_batch):
            start, end = i * batch_size, (i + 1) * batch_size
            end = end if end < num_data else num_data

            batch = []
            for j in (shuffle_idx[start:end]):
                aframes, embeddings, labels, length, _ = data[j]
                batch.append((aframes, embeddings, labels, length))

            # Add more data to batch
            if len(batch) < batch_size:
                idx_sublist = random.sample(shuffle_idx, batch_size - len(batch))
                for idx in idx_sublist:
                    aframes, embeddings, labels, length, _ = data[idx]
                    batch.append((aframes, embeddings, labels, length))

            aframes, embeddings, labels, length = pad_batch(batch, max_data_length)

            self.aframes_batches.append(aframes)
            self.word_emb_batches.append(embeddings)
            self.label_batches.append(labels)
            self.data_length_batches.append(length)

    def iter(self):
        while True:
            try:
                self.epoch += 1
                print('-------- Start epoch %d' % self.epoch)
                self.make_batches()
                for af, w, sl, y in zip(self.aframes_batches, self.word_emb_batches,
                                        self.data_length_batches, self.label_batches):
                    yield af, w, sl, y
            except StopIteration:
                continue

    def get_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, self.sequence_length, 4410]),
                                                  tf.TensorShape([self.batch_size, self.sequence_length,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, self.sequence_length, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, sl, y = batch_iterator.get_next()

        return af, w, sl, y

    def get_sentence_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH, 4410]),
                                                  tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, sl, y = batch_iterator.get_next()

        return af, w, sl, y

    def get_word_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH, 4410]),
                                                  tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, l, y = batch_iterator.get_next()

        return af, w, l, y


class EvalDataReader(object):
    def __init__(self, dataset_dir, batch_size, seq_length=100, data_unit=''):
        self.data_dir = dataset_dir
        self.batch_size = batch_size
        self.sequence_length = seq_length

        self.data_unit = data_unit

        self.aframes_batches = []
        self.word_emb_batches = []
        self.label_batches = []
        self.data_length_batches = []

    def make_batches(self, files):
        if self.data_unit == 'word':
            data = load_word_corpus(dataset_dir=self.data_dir, is_training=False, dataset=files)
            max_data_length = LONGEST_WORD_LENGTH

        elif self.data_unit == 'sentence':
            data = load_sent_corpus(self.data_dir, is_training=False, dataset=files)
            max_data_length = LONGEST_SENTENCE_LENGTH

        else:
            data = load_corpus(self.data_dir, self.sequence_length, is_training=False, dataset=files)
            max_data_length = self.sequence_length

        batch_size = self.batch_size

        # Shuffle idx in len(data)
        num_data = len(data)
        shuffle_idx = [_ for _ in range(num_data)]
        random.shuffle(shuffle_idx)

        # Create batches
        num_batch = num_data // batch_size + 1
        for i in range(num_batch):
            start, end = i * batch_size, (i + 1) * batch_size
            end = end if end < num_data else num_data

            batch = []
            for j in (shuffle_idx[start:end]):
                aframes, embeddings, labels, length, _ = data[j]
                batch.append((aframes, embeddings, labels, length))

            # Add more data to batch
            if len(batch) < batch_size:
                idx_sublist = random.sample(shuffle_idx, batch_size - len(batch))
                for idx in idx_sublist:
                    aframes, embeddings, labels, length, _ = data[idx]
                    batch.append((aframes, embeddings, labels, length))

            aframes, embeddings, labels, length = pad_batch(batch, max_data_length)

            # print(aframes.shape, words.shape, labels.shape)
            # (bs, seq_length, 4410), (bs, seq_length, 100), (bs, seq_length, 3)

            self.aframes_batches.append(aframes)
            self.word_emb_batches.append(embeddings)
            self.label_batches.append(labels)
            self.data_length_batches.append(length)

    def iter(self):
        for af, w, sl, y in zip(self.aframes_batches, self.word_emb_batches,
                                self.data_length_batches, self.label_batches):
            yield af, w, sl, y

    def get_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, self.sequence_length, 4410]),
                                                  tf.TensorShape([self.batch_size, self.sequence_length,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, self.sequence_length, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, sl, y = batch_iterator.get_next()

        return af, w, sl, y

    def get_sentence_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH, 4410]),
                                                  tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, LONGEST_SENTENCE_LENGTH, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, sl, y = batch_iterator.get_next()

        return af, w, sl, y

    def get_word_split(self):
        dataset = tf.data.Dataset.from_generator(self.iter,
                                                 (tf.float32, tf.float32, tf.int32, tf.float32),
                                                 (tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH, 4410]),
                                                  tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH,
                                                                  EMBEDDING_DIMENSION]),
                                                  tf.TensorShape([self.batch_size]),
                                                  tf.TensorShape([self.batch_size, LONGEST_WORD_LENGTH, 3])))

        batch_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        af, w, l, y = batch_iterator.get_next()

        return af, w, l, y

    def get_num_batches(self):
        return len(self.aframes_batches) - 1  # batch_length


def main():
    # train_reader = TrainDataReader('tfrecords/', 10)
    # train_reader.make_batches()

    data = load_corpus('tfrecords/')

    # sentences = load_sent_corpus('tfrecords_1/')
    # print(len(sentences))
    #
    # words = load_word_corpus('tfrecords_2/')
    # print(len(words))

    # max_sl, max_fn = 0, ''
    # for s in sentences:
    #     _, _, _, sl, fn = s
    #     if sl > max_sl:
    #         max_sl = sl
    #         max_fn = fn
    #
    # print(max_sl, max_fn)
