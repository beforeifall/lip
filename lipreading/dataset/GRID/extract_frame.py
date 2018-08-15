#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: extract_frame.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/05/26
#   description:
#
#================================================================

import os
import glob
from multiprocessing import Pool

VIDEO_DIR = 'video'
FRAME_DIR = 'frame'


def deframe(file_path):
    """deframe one video to frames.

    the frame will save to path(replace `video` with `frame` in file_path)
    for example, video `decompressed/video/s1/lgbs7s.mpg` will deframe to `decompressed/frame/s1/lgbs7s/%03d.jpg`

    Args:
        file_path: String. The video file path, e.g. `decompressed/video/s1/lgbs7s.mpg`

    Returns: TODO

    """
    video_filename, ext = os.path.splitext(file_path)
    frame_save_path = video_filename.replace(VIDEO_DIR, FRAME_DIR)
    frame_save_fmt = os.path.join(frame_save_path, '%03d.jpg')
    os.system('mkdir -p {}'.format(frame_save_path))
    os.system('ffmpeg -i {} {}'.format(file_path, frame_save_fmt))


def glob_video(path_prefix='./decompressed/video', video_ext='.mpg'):
    """find all the video

    Args:
        path_prefix (TODO): TODO

    Kwargs:
        video_ext (TODO): TODO

    Returns: TODO

    """
    path_pattern = os.path.join(path_prefix, "*/*{}".format(video_ext))
    print path_pattern
    return glob.glob(path_pattern)


def check_deframe(frame_dir='./decompressed/frame', frame_num=75):
    """check which video decompressed failed

    Args:
        frame_dir: the directory save frames.

    Returns: TODO

    """
    videos = glob.glob(os.path.join(frame_dir, '*/*'))
    for video in videos:
        frame_list = glob.glob(os.path.join(video, '*.jpg'))
        if len(frame_list) != frame_num:
            print('{} only has {} frames'.format(video, len(frame_list)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--deframe', help='True if perform deframe video', action='store_true')
    args = parser.parse_args()
    return args


def main():
    config = {
        'video_path': './decompressed/video',
        'video_ext': '.mpg',
        'deframe_pool': 4
    }
    args = parse_args()
    if args.deframe:
        video_list = glob_video(config['video_path'], config['video_ext'])
        p = Pool(config['deframe_pool'])
        p.map(deframe, video_list)

    check_deframe()


if __name__ == "__main__":
    main()
