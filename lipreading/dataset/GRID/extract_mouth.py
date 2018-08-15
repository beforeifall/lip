#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: extract_mouth.py
#   author: rizkiarm
#   email: https://github.com/rizkiarm/LipNet
#   created date: 2018/06/03
#   description: copied with minor modify from https://github.com/rizkiarm/LipNet/blob/master/scripts/extract_mouth_batch.py
#
#================================================================
'''
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.
Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]
    pattern: *.avi, *.mpg, etc
Example:
    python scripts/extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat
    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''

from util.video import Video
import os, fnmatch, sys, errno
from skimage import io
from multiprocessing import Pool

# FACE_PREDICTOR_PATH = sys.argv[4]
FACE_PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def find_files(directory, pattern):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filenames.append(filename)
    return filenames


def extract_mouth(filepath):
    filepath_wo_ext = os.path.splitext(filepath)[0]
    # target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
    target_dir = filepath_wo_ext.replace('video', 'mouth')
    if os.path.exists(target_dir):
        print('{} already processed'.format(filepath))
        return

    print "Processing: {}".format(filepath)
    video = Video(
        vtype='face',
        face_predictor_path='./shape_predictor_68_face_landmarks.dat'
    ).from_video(filepath)

    mkdir_p(target_dir)

    i = 1
    for frame in video.mouth:
        io.imsave(os.path.join(target_dir, "{0:03d}.jpg".format(i)), frame)
        i += 1


def main():
    SOURCE_PATH = sys.argv[1]
    SOURCE_EXTS = sys.argv[2]
    TARGET_PATH = sys.argv[3]
    filenames = find_files(SOURCE_PATH, SOURCE_EXTS)
    p = Pool(4)
    p.map(extract_mouth, filenames)
    for filename in filenames:
        extract_mouth(filename)


if __name__ == "__main__":
    main()
