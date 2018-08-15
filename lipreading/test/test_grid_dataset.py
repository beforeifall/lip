#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: test_grid_dataset.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/08
#   description:
#
#================================================================

import os
import time
import unittest
import tensorflow as tf

from lipreading.dataset.grid_dataset import grid_tfrecord_input_fn

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


class GridDatasetTest(unittest.TestCase):
    def testDataset(self):
        file_name_pattern = os.path.join(
            CURRENT_FILE_DIRECTORY,
            '../../data/tf-records/GRID/unseen_test.tfrecord')
        dataset = grid_tfrecord_input_fn(
            file_name_pattern=file_name_pattern,
            num_epochs=None,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=50,
            num_threads=8)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        print(features, labels)
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                for i in range(10):
                    t1 = time.time()
                    feature, label = sess.run([features, labels])
                    t = time.time() - t1
                    print('feature: {}, labels: {}. time: {}s'.format(
                        feature.shape, label.shape, t))


if __name__ == "__main__":
    unittest.main()
