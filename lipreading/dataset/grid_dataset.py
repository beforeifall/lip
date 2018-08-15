#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: grid_dataset.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/15
#   description:
#
#================================================================

import os
import time
import tensorflow as tf
from ..util.label_util import string2char_list

VIDEO_LENGTH = 75
VIDEO_FRAME_SHAPE = (50, 100)
GRID_EXAMPLE_FEATURES = {
    'video': tf.FixedLenFeature([VIDEO_LENGTH], tf.string),
    'label': tf.FixedLenFeature([], tf.string),
    'align': tf.VarLenFeature(tf.int64)
}


def parse_single_example(serialized_record):
    """parse serialized_record to tensors

    Args:
        serialized_record (TODO): TODO

    Returns: TODO

    """
    features = tf.parse_single_example(serialized_record,
                                       GRID_EXAMPLE_FEATURES)
    #parse x
    video = features['video']
    i = tf.constant(0)
    video_length = tf.shape(video)[0]
    images = tf.TensorArray(dtype=tf.uint8, size=video_length)

    c = lambda i, images: tf.less(i, video_length)
    b = lambda i, images: [tf.add(i, 1), images.write(i, tf.image.resize_images(tf.image.decode_jpeg(video[i], channels=3), size=VIDEO_FRAME_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))]
    i, images = tf.while_loop(
        c,
        b, [i, images],
        back_prop=False,
        # parallel_iterations=VIDEO_LENGTH
        )
    x = images.stack()
    x = tf.cast(x, tf.float32)
    x /= 255.0

    #parse y
    y = features['label']
    return x, y


def grid_tfrecord_input_fn(file_name_pattern,
                           mode=tf.estimator.ModeKeys.EVAL,
                           num_epochs=1,
                           batch_size=32,
                           num_threads=4):
    """TODO: Docstring for grid_tfrecord_input_fn.

    Args:
        file_name_pattern (TODO): TODO

    Kwargs:
        mode (TODO): TODO
        num_epochs (TODO): TODO
        batch_size (TODO): TODO
        num_threads:

    Returns: TODO

    """
    file_names = tf.matching_files(file_name_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_names)
    dataset = dataset.map(parse_single_example, num_parallel_calls=num_threads)

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100 * batch_size + 1)

    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=10)
    return dataset
