from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from helper import get_filenames_from_dir, EMBEDDING_DIMENSION

slim = tf.contrib.slim


def get_split(dataset_dir, is_training=True, batch_size=32, seq_length=100):

    if is_training:
        train_set, _, _ = get_filenames_from_dir(dataset_dir)

        paths = []
        for file in train_set:
            paths.append(dataset_dir + file + '.tfrecords')
        filename_queue = tf.train.string_input_producer(paths, shuffle=True)
    else:
        filename_queue = dataset_dir

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(  # take one frame
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'file_name': tf.FixedLenFeature([], tf.string),
            'audio_frame': tf.FixedLenFeature([], tf.string),
            'embedding': tf.FixedLenFeature([], tf.string),
        }
    )
    file_name = features['file_name']  # file name
    file_name.set_shape([])  # string, file name, further use

    audio_frame = tf.decode_raw(features['audio_frame'], tf.float32)  # decode the audio feature of one frame
    audio_frame.set_shape([4410])  # raw feature of audio considering interloculor information

    embedding = tf.decode_raw(features['embedding'], tf.float64)
    embedding.set_shape([EMBEDDING_DIMENSION])

    label = tf.decode_raw(features['label'], tf.float32)  # decode label of that frame
    label.set_shape([3])  # 3-D label

    print(audio_frame, embedding, label)

    # generate sequence, num_threads = 1, guarantee the generation of sequences is correct
    # i.e. frames of a sequence are in correct order and belong to same subject
    audio_frames, embeddings, labels, file_names = tf.train.batch(
        [audio_frame, embedding, label, file_name], seq_length, num_threads=1, capacity=1000
    )

    labels = tf.expand_dims(labels, 0)
    audio_frames = tf.expand_dims(audio_frames, 0)
    embeddings = tf.expand_dims(embeddings, 0)
    file_names = tf.expand_dims(file_names, 0)

    print(audio_frames, embeddings, labels, file_names)

    if is_training:  # generate mini_batch of sequences
        audio_frames, embeddings, labels, file_names = tf.train.shuffle_batch(
            [audio_frames, embeddings, labels, file_names], batch_size, 1000, 50, num_threads=1)
    else:
        audio_frames, embeddings, labels, file_names = tf.train.batch(
            [audio_frames, embeddings, labels, file_names], batch_size, num_threads=1, capacity=1000)

    print(audio_frames, embeddings, labels, file_names)

    frames = audio_frames[:, 0, :, :]
    labels = labels[:, 0, :]
    embeddings = embeddings[:, 0, :]
    file_names = file_names[:, 0, :]

    print(frames, embeddings, labels, file_names)

    masked_audio_samples = []
    masked_embeddings = []
    masked_labels = []

    for i in range(batch_size):  # make sure sequences in a batch all belong to the same subject
        mask = tf.equal(file_names[i][0], file_names[i])

        fs = tf.boolean_mask(frames[i], mask)
        es = tf.boolean_mask(embeddings[i], mask)
        ls = tf.boolean_mask(labels[i], mask)

        fs = tf.cond(tf.shape(fs)[0] < seq_length,
                     lambda: tf.pad(fs, [[0, seq_length - tf.shape(fs)[0]], [0, 0]], "CONSTANT"),
                     lambda: fs)

        es = tf.cond(tf.shape(es)[0] < seq_length,
                     lambda: tf.pad(es, [[0, seq_length - tf.shape(es)[0]], [0, 0]], "CONSTANT"),
                     lambda: es)

        ls = tf.cond(tf.shape(ls)[0] < seq_length,
                     lambda: tf.pad(ls, [[0, seq_length - tf.shape(ls)[0]], [0, 0]], "CONSTANT"),
                     lambda: ls)

        masked_audio_samples.append(fs)
        masked_embeddings.append(es)
        masked_labels.append(ls)

    masked_audio_samples = tf.stack(masked_audio_samples)
    masked_embeddings = tf.stack(masked_embeddings)
    masked_labels = tf.stack(masked_labels)

    print(masked_audio_samples, masked_embeddings, masked_labels)

    masked_audio_samples = tf.reshape(masked_audio_samples, (batch_size, seq_length, 4410))
    masked_embeddings = tf.reshape(masked_embeddings, (batch_size, seq_length, EMBEDDING_DIMENSION))
    masked_labels = tf.reshape(masked_labels, (batch_size, seq_length, 3))

    print(masked_audio_samples, masked_embeddings, masked_labels, file_names)
    return masked_audio_samples, masked_embeddings, masked_labels


# get_split('./tfrecords/')
