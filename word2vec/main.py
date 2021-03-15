"""
Train a text embedding spaces by using Word2Vec (German-corpus)
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/word2vec.ipynb
https://github.com/tensorflow/models/blob/ab8febd4f66ebf55673d828198d5899ccb447cf2/tutorials/embedding/word2vec.py#L377
"""

from __future__ import division, print_function, absolute_import

import collections
import os
import random
import time
import numpy as np
import tensorflow as tf

from preprocess import process_swc
from articles import Articles

flags = tf.app.flags
flags.DEFINE_string('method', 'articles', 'Method to preprocess text corpus')

# Word2Vec parameters
flags.DEFINE_integer('embedding_size', 100, '# Dimension of the embedding vector')
flags.DEFINE_integer('max_vocabulary_size', 50000, 'Total number of different words in the vocabulary')
flags.DEFINE_integer('min_occurrence', 7, 'Remove all words that does not appears at least n times')
flags.DEFINE_integer('skip_window', 4, 'How many words to consider left and right')
flags.DEFINE_integer('num_skips', 4, ' How many times to reuse an input to generate a label')
flags.DEFINE_float('max_occurrence_percentage', 1.0, 'Remove all words that appears in more than n% of articles')
flags.DEFINE_integer('num_sampled', 64, 'Number of negative examples to sample')
flags.DEFINE_integer('checkpoint_interval', 600, 'Checkpoint the model (i.e. save the parameters) every n')
flags.DEFINE_string('save_path', './ckpt/', 'Checkpoint directory')

# Training parameters
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('num_steps', 3000000, 'Number of training steps')
flags.DEFINE_integer('display_step', 10000, 'Display loss every n steps')
flags.DEFINE_integer('eval_step', 200000, 'Evaluate nearest neighbors of chosen words every n steps')

FLAGS = flags.FLAGS

# Evaluation Parameters
eval_words = ['vier', 'august', 'automat', 'apotheker', 'haus', 'frankfurt', 'deutsche']


####################
# USING TEXT_WORDS
####################
class TextWords(object):
    def __init__(self):
        self.text_words = process_swc()
        self.data = list()
        self.word2id = dict()
        self.id2word = dict()
        self.vocabulary_size = 0

    def build_dictionary(self):
        # Build the dictionary and replace rare words with UNK token
        count = [('UNK', -1)]

        # Retrieve the most common words
        count.extend(collections.Counter(self.text_words).most_common(FLAGS.max_vocabulary_size - 1))

        # Remove samples with less than 'min_occurrence' occurrences
        for i in range(len(count) - 1, -1, -1):
            if count[i][1] < FLAGS.min_occurrence:
                count.pop(i)
            else:
                # The collection is ordered, so stop when 'min_occurrence' is reached
                break

        # Compute the vocabulary size
        self.vocabulary_size = len(count)

        # Assign an id to each word
        for i, (word, _) in enumerate(count):
            self.word2id[word] = i

        unk_count = 0
        for word in self.text_words:
            index = self.word2id.get(word, 0)
            if index == 0:
                unk_count += 1
            self.data.append(index)

        count[0] = ('UNK', unk_count)
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        print('Words count:', len(self.text_words))
        print('Unique words:', len(set(self.text_words)))
        print('Vocabulary size:', self.vocabulary_size)
        print('Most common words:', count[:10])
        print('Least common words:', count[-10:])

    def get_data(self):
        return self.data

    def get_word2id(self):
        return self.word2id

    def get_id2word(self):
        return self.id2word

    def get_vocab_size(self):
        return self.vocabulary_size


def next_batch_textwords(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # Get window size (words left/right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)  # buffer is a double-ended queue
    if data_index + span > len(data):
        data_index = 0
    buffer.extend((data[data_index:data_index + span]))
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

            # j = 0 -> j = num_skips,
            # i = 0, batch[0:(num_skips)]
            # i = 1, batch[num_skips:(2 * num_skips)]

        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)

    return batch, labels


####################
# USING ARTICLES
####################
def build_training_data(window=FLAGS.skip_window):
    x = []
    y = []
    data = []

    for a in articles:
        for sent in a:
            for idx, word in enumerate(sent):
                for context_word in sent[max(idx - window, 0): min((idx + window), len(sent) + 1)]:
                    if context_word != word:
                        data.append([word, context_word])
                        x.append(word2id[word])
                        y.append(word2id[context_word])

    return x, y


def next_batch_articles(batch_size, skip_window):
    global next_word_idx

    assert batch_size % FLAGS.num_skips == 0
    assert FLAGS.num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size // FLAGS.num_skips):
        start = next_word_idx
        word = input_word[start]

        next_word_idx += 1
        while input_word[next_word_idx] == word:
            next_word_idx += 1
            if next_word_idx == len(input_word):
                next_word_idx = 0

        end = next_word_idx if next_word_idx != 0 else len(input_word)
        if FLAGS.num_skips <= (end - start):
            words_to_use = random.sample(context_word[start:end], FLAGS.num_skips)
        else:
            words_to_use = [random.choice(context_word[start:end]) for _ in range(FLAGS.num_skips)]

        for j, cw in enumerate(words_to_use):
            batch[i * FLAGS.num_skips + j] = word
            labels[i * FLAGS.num_skips + j, 0] = cw

    return batch, labels


####################
# TRAINING
####################
def next_batch(method=FLAGS.method):
    if method == 'articles':
        batch, labels = next_batch_articles(FLAGS.batch_size, FLAGS.skip_window)
    else:
        batch, labels = next_batch_textwords(FLAGS.batch_size, FLAGS.num_skips, FLAGS.skip_window)

    return batch, labels


def train():
    dirname = os.path.dirname(os.path.realpath(__file__))

    # Init the variables (assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Testing data
        x_test = np.array([word2id[w] for w in eval_words])

        average_loss = 0
        last_checkpoint_time = 0
        for step in range(1, FLAGS.num_steps + 1):
            # Get a new batch of data
            batch_x, batch_y = next_batch()

            # Run training
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            average_loss += loss

            if step % FLAGS.display_step == 0 or step == 1:
                if step > 1:
                    average_loss /= FLAGS.display_step
                print('Step ' + str(step) + ', Average Loss = ' + '{:.4f}'.format(average_loss))
                average_loss = 0

            if step % FLAGS.eval_step == 0 or step == 1:
                print('Evaluation...')
                sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
                for i in range(len(eval_words)):
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = '"%s" nearest neighbors:' % eval_words[i]
                    for k in range(top_k):
                        log_str = '%s %s,' % (log_str, id2word[nearest[k]])

                    print(log_str)

            now = time.time()
            if now - last_checkpoint_time > FLAGS.checkpoint_interval:

                path = saver.save(sess, os.path.join(dirname + FLAGS.save_path, 'model.ckpt'))
                print('---- Model saved in path: %s ----' % path)
                last_checkpoint_time = now

        path = saver.save(sess, os.path.join(dirname + FLAGS.save_path, 'model.ckpt'))
        print('---- Model saved in path: %s ----' % path)


def save_model(dimension):
    new_saver = tf.compat.v1.train.Saver()

    with tf.Session() as sess:
        checkpoint_dir = os.path.dirname(os.path.realpath(__file__)) + FLAGS.save_path + 'model.ckpt'
        new_saver.restore(sess, checkpoint_dir)
        print('Model restored.')
        embedding_weight_matrix = sess.run(embedding)

        w2v = dict()
        for idx, vector in enumerate(embedding_weight_matrix):
            word = id2word[idx]
            w2v[word] = vector

        dirname = os.path.dirname(__file__)
        args = 'min-o=%s_max-op=%s__window=%s_' % (str(FLAGS.min_occurrence), str(FLAGS.max_occurrence_percentage),
                                                   str(FLAGS.skip_window))

        f = open(dirname + '/vec/' + args + str(dimension) + '.vec', 'w+')
        f.write('%d %d\n' % (len(w2v), dimension))
        for key, values in w2v.items():
            f.write('%s %s\n' % (key, ' '.join(format(x, '.5f') for x in values)))

        f.close()

# Initialisation
if FLAGS.method == 'articles':
    document = Articles(params=FLAGS)
    document.build_dictionary()

    vocabulary_size = document.get_vocab_size()
    word2id = document.get_word2id()
    id2word = document.get_id2word()
    articles = document.get_articles()

    next_word_idx = 0
    input_word, context_word = build_training_data()
    next_batch_articles(FLAGS.batch_size, FLAGS.skip_window)
else:
    tw = TextWords()
    tw.build_dictionary()
    word2id = tw.get_word2id()
    id2word = tw.get_id2word()
    data = tw.get_data()
    vocabulary_size = tw.get_vocab_size()

    # Generate training batch for the skip-gram model
    data_index = 0


# Input data
X = tf.compat.v1.placeholder(tf.int32, shape=[None])

# Input label
Y = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])

with tf.device('/cpu:0'):
    # Create the embedding variable (each row represent a word embedding vector)
    embedding = tf.Variable(tf.random.normal([vocabulary_size, FLAGS.embedding_size]))

    # Lookup the corresponding embedding vectors for each sample in X
    X_embed = tf.nn.embedding_lookup(embedding, X)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, FLAGS.embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))  # noise contrastive estimation loss

# Compute the average NCE loss for the batch
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=FLAGS.num_sampled,
                   num_classes=vocabulary_size)
)

# Define optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation
# Compute the cosine similarity between input data embedding and every embedding vectors
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)


########################
# Call train()
########################
tic = time.time()
print('Start training w2v with arguments: min_occur = %d, max_occur_percent = %.2f, window = %d, num_skips = %d, es = %d'
      % (FLAGS.min_occurrence, FLAGS.max_occurrence_percentage, FLAGS.skip_window, FLAGS.num_skips, FLAGS.embedding_size))

train()
save_model(dimension=FLAGS.embedding_size)

tic = time.time() - tic
print('Finish training in {%d} minutes, {%d} seconds' % (tic // 60, tic % 60))
