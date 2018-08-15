#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: ctc_estimator.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/10
#   description:
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.keras as keras

from ..util import label_util
from .cnn_extractor import LipNet
from .tcn import tcn
from .base_estimator import BaseEstimator


class CtcEstimator(BaseEstimator):
    """lipnet model: cnn+rnn+ctc
    """

    def __init__(self, model_parms, run_config):
        super(CtcEstimator, self).__init__(model_parms, run_config)

    def model_fn(self, features, labels, mode, params):
        """model define fn

        Args:
            features: 5-D Tensor. videos of shape (batch_size, T, H, W, C)
            labels: 1-D Tensor. labels of shape (batch_size,). For example: [ 'ab haha', 'hha fd fd']
            mode: tf.estimator.ModeKeys. PREDICT or TRAIN or EVAL.
            params: Dict. params of the Estimator.

        Returns: tf.estimator.EstimatorSpec

        """

        feature_len = params.get('feature_len', 256)
        gru_layer_num = params.get('gru_layer_num', 2)
        gru_units = params.get('gru_unit', 256)
        beam_width = params.get('beam_width', 4)
        learning_rate = params.get('learning_rate', 0.001)
        use_tcn = params.get('use_tcn', False)

        in_training = mode == tf.estimator.ModeKeys.TRAIN

        video = features
        feature_extractor = LipNet(
            feature_len=feature_len,
            training=in_training,
            scope='cnn_feature_extractor')
        output_size = params.get('output_size', 28)
        net = feature_extractor.build(video)  # BxTxN

        # rnn
        if use_tcn:
            with tf.variable_scope('tcn'):
                num_channels = [gru_units for i in range(gru_layer_num)]
                net = tcn.TemporalConvNet(num_channels, 3, 0.25)(
                    net, training=in_training) # BxTxN
                net_reverse = tf.reverse(net, axis=[1])
                net_reverse = tcn.TemporalConvNet(num_channels, 3, 0.25)(
                    net_reverse, training=in_training) # BxTxN
                net = tf.concat([net, net_reverse], 2) # BxTx2N
                # net = wn_tcn.TemporalConvNet(num_channels, 1, 3, 0.25)(
                # net, training=in_training)
        else:
            with tf.variable_scope('brnn'):
                for i in range(gru_layer_num):
                    net = keras.layers.Bidirectional(
                        keras.layers.GRU(
                            gru_units,
                            return_sequences=True,
                            kernel_initializer='Orthogonal',
                            name='gru_{}'.format(i)),
                        merge_mode='concat',
                        name='gru_concat_{}'.format(i))(net)

        with tf.variable_scope('fc1'):
            logits = keras.layers.Dense(
                output_size, kernel_initializer='he_normal',
                name='dense1')(net)
            logits = tf.transpose(logits, (1, 0, 2))  #time major

        # decode logits
        batch_size = tf.expand_dims(tf.shape(video)[0], 0)
        input_length = tf.expand_dims(tf.shape(video)[1], 0)
        sequence_length = tf.tile(input_length, batch_size)
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(
            logits,
            sequence_length,
            beam_width=beam_width,
            merge_repeated=False)

        predictions = decoded[0]

        predicted_char_list = label_util.indices2string(
            predictions
        )  # (batch_size, string_len). [ ['a', 'b'], ['a','b','c'] ]
        predicted_string = label_util.char_list2string(
            predicted_char_list)  # ['ab', 'abc']

        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_output = {
                'predictions':
                tf.sparse_tensor_to_dense(predictions, default_value=-1),
                'predicted_string':
                predicted_string
            }
            export_outputs = {
                'predictions':
                tf.estimator.export.PredictOutput(predict_output)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predict_output,
                export_outputs=export_outputs)

        # process label
        labels = label_util.string2char_list(labels)
        numeric_label = label_util.string2indices(labels)
        numeric_label = tf.cast(numeric_label, tf.int32)

        # calculate metrics
        cer, wer = self.cal_metrics(labels, predicted_char_list)

        loss = tf.nn.ctc_loss(numeric_label, logits, sequence_length)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(
                loss=cost, global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook(
            {
                'loss': cost,
                'cer': tf.reduce_mean(cer),
                'wer': tf.reduce_mean(wer),
                'predicted': predicted_string[:5],
                'labels': label_util.char_list2string(labels)[:5]
            },
            every_n_iter=100)

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('cer', tf.reduce_mean(cer))
        tf.summary.scalar('wer', tf.reduce_mean(wer))

        if mode == tf.estimator.ModeKeys.TRAIN:
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cost,
                train_op=train_op,
                training_hooks=[logging_hook])
            return estimator_spec

        eval_metric_ops = {
            'cost': tf.metrics.mean(loss),
            'cer': tf.metrics.mean(cer),
            'wer': tf.metrics.mean(wer)
        }
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cost,
            train_op=train_op,
            training_hooks=[logging_hook],
            eval_metric_ops=eval_metric_ops)
        return estimator_spec
