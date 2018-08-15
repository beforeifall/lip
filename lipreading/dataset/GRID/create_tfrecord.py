#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: create_tfrecord.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/06/06
#   description:
#
#================================================================

import os
import cv2
import glob
import time
import argparse
import numpy as np
import tensorflow as tf

from tf_record import feature


def _convert2example(video, label, align):
    """get the tf example given the video and label

    An example of label and align:
        label: 'sil bin blue at f two now sil'
        align: [0, 23, 29, 34, 35, 41, 47, 53]

    Args:
        video: A list, each element is encoded image.
        label: String. The corresponding sentence of the video.
        align: A list, each element is the begining frame id of a word.

    Returns:
        A tf example

    """
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'video': feature.bytes_list_feature(video),
                'label': feature.bytes_feature(tf.compat.as_bytes(label)),
                'align': feature.int64_list_feature(align)
            }))


def _parse_align_file(align_file_path):
    """Parse label and align time

    Args:
        align_file_path: Path of align file.

    Returns:
        A tuple of two elements. The first element is a string which
        represent the sentence. The second element is a list, each element
        of which is the begining frame id of a word.

    For example: given align file with content
    ```
    0 23750 sil
    23750 29500 bin
    29500 34000 blue
    34000 35500 at
    35500 41000 f
    41000 47250 two
    47250 53000 now
    53000 74500 sil
    ```
    The function returns
    ( 'sil bin blue at f two now sil',
      [0, 23,29, 34, 35, 41, 47, 53]
    )
    """
    label = ''
    align = []
    for line in open(align_file_path):
        line = line.rstrip('\n')
        begin, end, word = line.split()
        label += word + ' '
        align.append(int(begin) / 1000)
    label.rstrip()
    return label, align


def read_example_from_path(video_path, align_path):
    """Read an example from path

    Args:
        video_path: The path contain mouth sequence.
        align_path The path of align file.

    Returns:
        tf.train.Example
    """
    mouth_frame_filenames = glob.glob(os.path.join(video_path, '*.jpg'))
    mouth_frame_filenames = sorted(mouth_frame_filenames)

    img = cv2.imread(mouth_frame_filenames[0])  # read image  in BGR
    video = []
    for idx, filename in enumerate(mouth_frame_filenames):
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_jpg = fid.read()
        video.append(encoded_jpg)

    label, align = _parse_align_file(align_path)
    return _convert2example(video, label, align)


def write_tfrecord(split_file,
                   tfrecord_save_path,
                   video_dir,
                   align_dir,
                   seed=10000):
    """write tf record file given split.

    Args:
        split_file: The txt file contains the videos to be written.
        tfrecord_save_path: The path (include filename) to save tfrecord.
        video_dir: The video dir contains the videos (extracted mouth sequence).
        align_dir: The align dir contains the align files.

    """
    video_list = [line.rstrip('\n') for line in open(split_file)]
    np.random.seed(seed)
    np.random.shuffle(video_list)

    video_n = len(video_list)

    with tf.python_io.TFRecordWriter(tfrecord_save_path) as writer:
        t = 0
        for idx, video in enumerate(video_list):
            video_path = os.path.join(video_dir, video)
            align_path = os.path.join(align_dir, video + '.align')

            t1 = time.time()
            example = read_example_from_path(video_path, align_path)
            writer.write(example.SerializeToString())
            t2 = time.time()
            t = t * idx / (idx + 1) + (t2 - t1) / (idx + 1)

            if idx % 1000 == 0:
                print('Begin writing video {}/{}. {:.2f}ms/video'.format(idx, video_n, t*1000))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_file', help='the txt file contain video list')
    parser.add_argument('--save_file', help='the file path to save tfrecord')
    parser.add_argument('--video_dir', help='the path of video')
    parser.add_argument('--align_dir', help='the path of align')
    return parser.parse_args()


def main():
    args = parse_args()
    write_tfrecord(args.split_file, args.save_file, args.video_dir,
                   args.align_dir)


if __name__ == "__main__":
    main()
