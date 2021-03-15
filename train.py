from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import losses
import models

from data_utils import TrainDataReader, filter_labels
from data_provider import get_split
from tensorflow.python.platform import tf_logging as logging


# Update tfrecord directory
TFRECORD_DIR = 'tfrecords/data/'
SENT_DATASET_DIR = 'tfrecords/sentences/'
WORD_DATASET_DIR = 'tfrecords/words/'

slim = tf.contrib.slim

# Create FLAGS
flags = tf.app.flags
flags.DEFINE_string('dataset_dir', TFRECORD_DIR, 'The tfrecords directory')  # tfrecord directory
flags.DEFINE_string('sent_dataset_dir', SENT_DATASET_DIR, 'The tfrecords directory for sentences')
flags.DEFINE_string('word_dataset_dir', WORD_DATASET_DIR, 'The tfrecords directory')
flags.DEFINE_string('train_dir', 'checkpoints/',
                    'Directory where to write event logs and checkpoint.')  # model save path

# Training parameters
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 5, 'The batch size to use.')
flags.DEFINE_integer('hidden_units', 256, 'Recurrent network hidden units.')

flags.DEFINE_string('model', 'audio_model2', 'Which model is going to be used: audio, video, or both')
flags.DEFINE_integer('sequence_length', 100, 'Number of audio frames in one input')

flags.DEFINE_string('data_unit', None, '{word|sentence} as data input')
flags.DEFINE_boolean('liking', True, 'Liking dimension is calculated in the losses function or not')
FLAGS = flags.FLAGS


def train():
    tf.set_random_seed(1)

    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        audio_frames, word_embeddings, ground_truth = get_split(FLAGS.dataset_dir, True,
                                                                FLAGS.batch_size, seq_length=FLAGS.sequence_length)

        # Define model graph.
        with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout], is_training=True):
            prediction = models.get_model(FLAGS.model)(audio_frames,
                                                       emb=tf.cast(word_embeddings, tf.float32),
                                                       hidden_units=FLAGS.hidden_units)

        optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.99)

        count = 0
        for i, name in enumerate(['arousal', 'valence', 'liking']):
            count += 1
            
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,))

            loss = losses.concordance_cc(pred_single, gt_single)
            tf.summary.scalar('losses/{} loss'.format(name), loss)
            
            mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            tf.summary.scalar('losses/mse {} loss'.format(name), mse)
            
            tf.losses.add_loss(loss / count)
        
        # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)
        
        with tf.Session(graph=g) as sess:
            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)
            
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                FLAGS.train_dir,
                                save_summaries_secs=60,
                                save_interval_secs=120)


if __name__ == '__main__':
    train()
