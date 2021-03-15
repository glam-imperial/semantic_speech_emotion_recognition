import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


class AttributeDict(dict):
    """
        Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    """

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def concordance_cc2(prediction, ground_truth):
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/mean_pred': slim.metrics.streaming_mean(prediction),
        'eval/mean_lab': slim.metrics.streaming_mean(ground_truth),
        'eval/cov_pred': slim.metrics.streaming_covariance(prediction, prediction),
        'eval/cov_lab': slim.metrics.streaming_covariance(ground_truth, ground_truth),
        'eval/cov_lab_pred': slim.metrics.streaming_covariance(prediction, ground_truth)
    })

    metrics = dict()
    for name, value in names_to_values.items():
        metrics[name] = value

    mean_pred = metrics['eval/mean_pred']
    var_pred = metrics['eval/cov_pred']
    mean_lab = metrics['eval/mean_lab']
    var_lab = metrics['eval/cov_lab']
    var_lab_pred = metrics['eval/cov_lab_pred']

    denominator = (var_pred + var_lab + (mean_pred - mean_lab) ** 2)

    ccc2 = (2 * var_lab_pred) / denominator

    return ccc2, names_to_values, names_to_updates


def metric_graph():
    with tf.variable_scope('CCC'):
        pred = tf.placeholder(tf.float32, [None, 3], name='pred')
        label = tf.placeholder(tf.float32, [None, 3], name='label')

        metric = {0: 0.0, 1: 0.0, 2: 0.0}

        for i in [0, 1, 2]:

            pred_mean, pred_var = tf.nn.moments(pred[:, i], [0])
            gt_mean, gt_var = tf.nn.moments(label[:, i], [0])

            mean_cent_prod = tf.reduce_mean((pred[:, i] - pred_mean) * (label[:, i] - gt_mean))
            # cov = tf.contrib.metrics.streaming_covariance(pred[:, i], label[:, i])[0]

            metric[i] = 2. * mean_cent_prod / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

    return AttributeDict(
        eval_predictions=pred,
        eval_labels=label,
        eval_metric_arousal=metric[0],
        eval_metric_valence=metric[1],
        eval_metric_liking=metric[2]
    )


def metric_calculation(pred, label):
    metric = {0: 0.0, 1: 0.0, 2: 0.0}

    for i in [0, 1, 2]:

        pred_mean, pred_var = np.mean(pred[:, i]), np.var(pred[:, i])
        gt_mean, gt_var = np.mean(label[:, i]), np.var(label[:, i])

        mean_cent_prod = np.mean((pred[:, i] - pred_mean) * (label[:, i] - gt_mean))

        metric[i] = 2. * mean_cent_prod / (pred_var + gt_var + np.square(pred_mean - gt_mean))

    return metric[0], metric[1], metric[2]
