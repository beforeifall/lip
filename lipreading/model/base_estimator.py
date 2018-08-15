#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: base_estimator.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/09
#   description:
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.keras as keras
from .cnn_extractor import LipNet
from ..util import label_util


class BaseEstimator(object):
    """base estimator for lipreading

    Args:
        model_parms: Dict. parameters to build model_fn
        run_config: RunConfig. config for `Estimator`
    """

    def __init__(self, model_parms, run_config):
        super(BaseEstimator, self).__init__()
        self.model_parms = model_parms
        self.run_config = run_config
        self.estimator = tf.estimator.Estimator(
            self.model_fn, params=self.model_parms, config=self.run_config)

    def train_and_evaluate(self,
                           train_input_fn,
                           eval_input_fn,
                           max_steps=100000,
                           eval_steps=100,
                           throttle_secs=200):
        """train and eval.

        Args:
            train_input_fn: Input fn for Train.
            eval_input_fn: Input fn for Evaluation.

        Kwargs:
            max_steps: Max training steps.
            eval_steps: Steps to evaluate.
            throttle_secs: Evaluate interval. evaluation will perform only when new checkpoints exists.

        """
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=max_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            throttle_secs=throttle_secs,
            steps=eval_steps)

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def evaluate(self, eval_input_fn, steps=None, checkpoint_path=None):
        """evaluate and print

        Args:
            eval_input_fn: Input function.

        Kwargs:
            steps: Evaluate steps
            checkpoint_path: Checkpoint to evaluate.

        Returns: Evaluate results.

        """
        return self.estimator.evaluate(
            eval_input_fn, steps=steps, checkpoint_path=checkpoint_path)

    def predict(self, predict_input_fn, checkpoint_path=None):
        """predict new examples
        Args:
            predict_input_fn: Input fn.

        """
        predictions = self.estimator.predict(
            predict_input_fn, checkpoint_path=checkpoint_path)
        for prediction in predictions:
            print prediction

    def model_fn(self, features, labels, mode, params):
        """the model_fn to estimator

        Args:
            features: Tensor. videos of shape (batch_size, T, H, W, C)
            labels: 1-D Tensor. labels of shape (batch_size,). For example: [ 'ab haha', 'hha fd fd']
            mode: tf.estimator.ModeKeys. PREDICT or TRAIN or EVAL.
            params: Dict. params of the Estimator.

        Returns: `EstimatorSpec`

        """
        raise NotImplementedError('model function is not implemented')

    def cal_metrics(self, labels, predictions):
        """ calculate cer, wer

        Args:
            labels: `2-D` string `SparseTensor`. Shape: (batch_size, sentence_length). example: [['l', 'a','y',' ', 'c'], [ 'p', 'a'] ]
            predictions: `2-D` string `SparseTensor`. The same as labels but predicted values.

        Returns: tuple of length 2: (cer, wer). Note The cer and wer are not averaged along batch_size. cer and wer are `Tensor` of shape (batch_size,)
        """
        cer = tf.edit_distance(predictions, labels)  #character error rate
        wer = tf.edit_distance(
            label_util.char2word(predictions),
            label_util.char2word(labels))  # word error rate
        return cer, wer

    @staticmethod
    def get_runConfig(model_dir,
                      save_checkpoints_steps,
                      multi_gpu = False,
                      keep_checkpoint_max=100):
        """ get RunConfig for Estimator.
        Args:
            model_dir: The directory to save and load checkpoints.
            save_checkpoints_steps: Step intervals to save checkpoints.
            keep_checkpoint_max: The max checkpoints to keep.
        Returns: Runconfig.

        """
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        if multi_gpu:
            distribution = tf.contrib.distribute.MirroredStrategy()
        else:
            distribution = None
        return tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
            keep_checkpoint_max=100,
            train_distribute=distribution,
            session_config=sess_config)
