"""
    Based on https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/nnUNet/runtime/metrics.py

"""

import tensorflow as tf
import numpy as np


class Dice(tf.keras.metrics.Metric):
    def __init__(self, n_class, index=None, **kwargs):
        super().__init__(**kwargs)
        self.n_class = n_class
        self.index = index
        self.dice = self.add_weight(
            name='dice',
            initializer='zeros',
            aggregation=tf.VariableAggregation.SUM,
        )
        self.steps = self.add_weight(
            name='steps',
            initializer='zeros',
            aggregation=tf.VariableAggregation.SUM
        )

    def update_state(self, y_true, y_pred, **kwargs):
        self.steps.assign_add(1)
        if self.index is None:
            self.dice.assign_add(tf.reduce_mean(self.compute_stats(y_true, y_pred)[1:]))
        else:
            self.dice.assign_add(self.compute_stats(y_true, y_pred)[self.index])

    def result(self):
        # dice_sum = hvd.allreduce(self.dice, op=hvd.mpi_ops.Sum)
        # steps_sum = hvd.allreduce(self.steps, op=hvd.mpi_ops.Sum)
        return self.dice / self.steps

    def compute_stats(self, y_true, y_pred):
        scores = tf.TensorArray(tf.float32, size=self.n_class)
        pred_classes = tf.argmax(y_pred, axis=-1)
        for i in range(0, self.n_class):
            if tf.math.count_nonzero(y_true == i) == 0:
                scores = scores.write(i, 1. if tf.math.count_nonzero(pred_classes == i) == 0 else 0.)
                continue
            true_pos, false_neg, false_pos = self.get_stats(y_true, pred_classes, i)
            denom = tf.cast(2 * true_pos + false_pos + false_neg, dtype=tf.float32)
            score_cls = tf.cast(2 * true_pos, tf.float32) / denom
            scores = scores.write(i, score_cls)
        return scores.stack()

    @staticmethod
    def get_stats(target, preds, class_idx):
        true_pos = tf.math.count_nonzero(tf.logical_and(preds == class_idx, target == class_idx))
        false_neg = tf.math.count_nonzero(tf.logical_and(preds != class_idx, target == class_idx))
        false_pos = tf.math.count_nonzero(tf.logical_and(preds == class_idx, target != class_idx))
        return true_pos, false_neg, false_pos

