from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import shutil
import time
from pathlib import Path

import tensorflow as tf
import numpy as np

import models
import data_metrics
from data_utils import EvalDataReader, filter_labels
from data_provider import get_split

# Run evaluation with CPU
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

slim = tf.contrib.slim

DATASET_DIR = 'tfrecords/'
SENT_DATASET_DIR = 'tfrecords_1/'
WORD_DATASET_DIR = 'tfrecords_2/'
CKPT_DIR = 'checkpoints/'
TMP_DIR = 'tmp/'

LOG_PATH = 'ManuallyStore/log_eval.txt'

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', DATASET_DIR, 'The tfrecords directory.')  # specify the dataset path
flags.DEFINE_string('sent_dataset_dir', SENT_DATASET_DIR, 'The tfrecords directory for sentences')
flags.DEFINE_string('word_dataset_dir', WORD_DATASET_DIR, 'The tfrecords directory for words')
flags.DEFINE_string('checkpoint_dir', CKPT_DIR, 'The checkpoint directory.')  # specify the saved model path

flags.DEFINE_integer('batch_size', 1, 'The batch size to use.')
flags.DEFINE_integer('hidden_units', 256, 'Recurrent network hidden units.')
flags.DEFINE_string('model', 'audio_model2', 'Which model is going to be used: audio, video, or both')
flags.DEFINE_integer('sequence_length', 100, 'Number of audio frames in one input')

flags.DEFINE_integer('eval_interval_secs', 75, 'The seconds to wait until next evaluation.')
flags.DEFINE_string('portion', 'Devel', '{Devel|Test} to evaluation on validation or test set.')
flags.DEFINE_string('data_unit', None, '{word|sentence} as data input')
flags.DEFINE_boolean('liking', True, 'Liking dimension is calculated in the losses function or not')

# tf.app.flags.DEFINE_string('log_dir', 'ckpt/eval', 'The checkpoint/evaluation directory.')
tf.app.flags.DEFINE_string('store_best_path', './ManuallyStore/current_best.ckpt', 'where to manually store model')

FLAGS = flags.FLAGS


def evaluate(file2eval, model_path):

    with tf.Graph().as_default():
        filename_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.string])

        # Load dataset
        eval_reader = EvalDataReader(dataset_dir=FLAGS.dataset_dir,
                                     batch_size=FLAGS.batch_size,
                                     seq_length=FLAGS.sequence_length)
        eval_reader.make_batches([file2eval])
        audio_frames, word_embeddings, data_length, ground_truth = eval_reader.get_split()

        num_batches = eval_reader.get_num_batches()

        # Define model graph.
        with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout], is_training=False):
            prediction = models.get_model(FLAGS.model)(audio_frames,
                                                       emb=word_embeddings,
                                                       hidden_units=FLAGS.hidden_units)

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            evaluated_predictions = None
            evaluated_labels = None

            print('Evaluating file : {}'.format(file2eval))

            sess.run(filename_queue.enqueue(file2eval))

            for _ in range(num_batches):
                pred, gt, s_len = sess.run([prediction, ground_truth, data_length])

                pred_batch, gt_batch = [], []
                for i, _ in enumerate(['arousal', 'valence', 'liking']):
                    pred_single, gt_single = filter_labels(pred[:, :, i],
                                                           gt[:, :, i],
                                                           s_len)  # batch * seq * 1

                    pred_batch.append(pred_single)
                    gt_batch.append(gt_single)

                pred_batch = tf.convert_to_tensor(pred_batch)
                gt_batch = tf.convert_to_tensor(gt_batch)

                if evaluated_predictions is not None:
                    evaluated_predictions = tf.concat([evaluated_predictions, pred_batch], axis=1)
                    evaluated_labels = tf.concat([evaluated_labels, gt_batch], axis=1)
                else:
                    evaluated_predictions = pred_batch
                    evaluated_labels = gt_batch

            evaluated_predictions = evaluated_predictions.eval()
            evaluated_labels = evaluated_labels.eval()

            for i in range(sess.run(filename_queue.size())):
                sess.run(filename_queue.dequeue())
            if sess.run(filename_queue.size()) != 0:
                raise ValueError('Queue not empty!')

    evaluated_predictions = evaluated_predictions.transpose()
    evaluated_labels = evaluated_labels.transpose()

    print(evaluated_predictions.shape, evaluated_labels.shape)

    return evaluated_predictions, evaluated_labels


def evaluate_2(file2eval, model_path):
    with tf.Graph().as_default():

        filename_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.string])

        # Load dataset.
        audio_frames, word_embeddings, labels = get_split(filename_queue, False,
                                                          FLAGS.batch_size, seq_length=FLAGS.sequence_length)

        # Define model graph.
        with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout], is_training=False):
            predictions = models.get_model(FLAGS.model)(audio_frames,
                                                       emb=tf.cast(word_embeddings, tf.float32),
                                                       hidden_units=FLAGS.hidden_units)

        coord = tf.train.Coordinator()
        saver = tf.train.Saver(slim.get_variables_to_restore())

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            tf.train.start_queue_runners(sess=sess, coord=coord)

            evaluated_predictions = []
            evaluated_labels = []

            nexamples = _get_num_examples(file2eval)
            num_batches = int(math.ceil(nexamples / (float(FLAGS.sequence_length))))

            print('Evaluating file : {}'.format(file2eval))
            sess.run(filename_queue.enqueue(file2eval))
            sess.run(filename_queue.enqueue(file2eval))

            for i in range(num_batches):
                prediction_, label_ = sess.run([predictions, labels])
                evaluated_predictions.append(prediction_[0])
                evaluated_labels.append(label_[0])

            print(np.vstack(evaluated_predictions).shape)
            evaluated_predictions = np.vstack(evaluated_predictions)[:nexamples]
            print(np.vstack(evaluated_predictions).shape)
            evaluated_labels = np.vstack(evaluated_labels)[:nexamples]

            for i in range(sess.run(filename_queue.size())):
                sess.run(filename_queue.dequeue())
            if sess.run(filename_queue.size()) != 0:
                raise ValueError('Queue not empty!')
            coord.request_stop()

    return evaluated_predictions, evaluated_labels


def _get_num_examples(tf_file):
    c = 0
    for _ in tf.python_io.tf_record_iterator(tf_file):
        c += 1

    return c


def copy2temporary(model_path):
    shutil.copy(model_path + '.data-00000-of-00001', '{}temporary.ckpt.data-00000-of-00001'.format(TMP_DIR))
    shutil.copy(model_path + '.index', '{}temporary.ckpt.index'.format(TMP_DIR))
    shutil.copy(model_path + '.meta', '{}temporary.ckpt.meta'.format(TMP_DIR))

    return '{}temporary.ckpt'.format(TMP_DIR)


# if you want to save the best model
def copy2best(model_path, inx):
    shutil.copy(model_path + '.data-00000-of-00001', './Best_Audio_{}.ckpt.data-00000-of-00001'.format(inx))
    shutil.copy(model_path + '.index', './Best_Audio_{}.ckpt.index'.format(inx))
    shutil.copy(model_path + '.meta', './Best_Audio_{}.ckpt.meta'.format(inx))


def deltemporary(model_path):
    os.remove(model_path + '.data-00000-of-00001')
    os.remove(model_path + '.index')
    os.remove(model_path + '.meta')


def main(_):
    if FLAGS.data_unit == 'word':
        dataset_dir = Path(FLAGS.word_dataset_dir)
    elif FLAGS.data_unit == 'sentence':
        dataset_dir = Path(FLAGS.sent_dataset_dir)
    else:
        dataset_dir = Path(FLAGS.dataset_dir)

    best, inx = 0.62, 1
    cnt = 0

    while True:
        if FLAGS.portion == 'Test':
            model_path = FLAGS.store_best_path
        else:
            model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            print('Current latest model: ' + model_path)

            model_path = copy2temporary(model_path)

        predictions, labels = None, None

        eval_model = data_metrics.metric_graph()
        eval_arousal = eval_model.eval_metric_arousal
        eval_valence = eval_model.eval_metric_valence
        eval_liking = eval_model.eval_metric_liking

        files = os.listdir(str(dataset_dir))
        portion_files = [str(dataset_dir / x) for x in files if FLAGS.portion in x]

        for tf_file in portion_files:
            predictions_file, labels_file = evaluate_2(str(tf_file), model_path)

            print(tf_file)

            if predictions is not None and labels is not None:
                predictions = np.vstack((predictions, predictions_file))
                labels = np.vstack((labels, labels_file))
            else:
                predictions = predictions_file
                labels = labels_file

        print(predictions.shape)
        print(labels.shape)

        with tf.Session() as sess:

            e_arousal, e_valence, e_liking = sess.run([eval_arousal, eval_valence, eval_liking],
                                                      feed_dict={
                                                          eval_model.eval_predictions: predictions,
                                                          eval_model.eval_labels: labels
                                                      })
            eval_res = np.array([e_arousal, e_valence, e_liking])

            if FLAGS.liking:
                eval_loss = 1 - (np.sum(eval_res) / eval_res.shape[0])
            else:
                eval_loss = 2 - eval_res[0] - eval_res[1]

            print('Evaluation: %d, loss: _%.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                  % (cnt, eval_loss, eval_res[0], eval_res[1], eval_res[2]))

            cnt += 1

        if eval_loss < best:
            print('================================================================================')
            if FLAGS.portion == 'Devel':
                copy2best(model_path, inx)

            log = open(LOG_PATH, 'a')
            log.write('Evaluated Model %d: %s \n' % (inx, model_path))
            log.write('Evaluated loss: %.4f, arousal: %.4f, valence: %.4f, liking: %.4f\n'
                      % (eval_loss, eval_res[0], eval_res[1], eval_res[2]))
            log.write('========================================\n')
            inx += 1
            log.close()

        else:
            if FLAGS.portion == 'Devel':
                print(model_path)
                deltemporary(model_path)

        print('Finished evaluation! Now waiting for {} secs'.format(FLAGS.eval_interval_secs))
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()