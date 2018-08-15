#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: split_train_val.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/06/06
#   description: split database
#
#================================================================

import os
import glob
import numpy as np

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

MISSING_SPEAKER = 21
SPEAKERS = np.arange(1, 35)


def overlapped_split(db_base_dir,
                     split_save_dir,
                     test_num_per_speaker=255,
                     seed=10000):
    """ For each speaker, random select test_num_per_speaker to test_set. The remaining to train set.

    After split. two txt files will be saved in `split_save_dir`: overlapped_train.txt, overlapped_test.txt

    Args:
        db_base_dir: The base directory of GRID.
        split_save_dir: The saved directory of split text file.

    Kwargs:
        test_num_per_speaker: The number samples selected each speaker to test set.
        seed: The shuffle seed.

    """

    speakers = np.delete(SPEAKERS, MISSING_SPEAKER - 1)
    speaker_dirs = [
        os.path.join(db_base_dir, 's{}'.format(speaker))
        for speaker in speakers
    ]
    train_video = []
    test_video = []
    np.random.seed(seed)
    for speaker_dir in speaker_dirs:
        speaker_video_paths = glob.glob(speaker_dir + '/*')
        speaker_video = []
        for path in speaker_video_paths:
            base, video_name = os.path.split(path)
            base, speaker = os.path.split(base)
            speaker_video.append(os.path.join(speaker, video_name))

        np.random.shuffle(speaker_video)
        test_video.extend(speaker_video[0:test_num_per_speaker])
        train_video.extend(speaker_video[test_num_per_speaker:])

    #dump to text
    train_save_file = os.path.join(split_save_dir, 'overlapped_train.txt')
    test_save_file = os.path.join(split_save_dir, 'overlapped_test.txt')
    np.savetxt(train_save_file, train_video, fmt='%s')
    np.savetxt(test_save_file, test_video, fmt='%s')


def unseen_split(db_base_dir,
                 split_save_dir,
                 unseen_speakers=[1, 2, 20, 22],
                 seed=10000):
    """ Selected speakers are chosen to test set. The remaining speakers to train set.

    Args:
        db_base_dir: The base directory of GRID.
        split_save_dir: The saved directory of split text file.

    Kwargs:
        unseen_speakers: The speakers chosen as test.
        seed: The shuffle seed.

    Returns: TODO

    """
    unseen_speakers = np.array(unseen_speakers)
    seen_speakers = np.delete(SPEAKERS,
                              np.append(unseen_speakers, MISSING_SPEAKER) - 1)
    unseen_speaker_dirs = [
        os.path.join(db_base_dir, 's{}'.format(speaker))
        for speaker in unseen_speakers
    ]
    seen_speaker_dirs = [
        os.path.join(db_base_dir, 's{}'.format(speaker))
        for speaker in seen_speakers
    ]

    train_video = []
    test_video = []
    np.random.seed(seed)
    for idx, speaker_dirs in enumerate([seen_speaker_dirs, unseen_speaker_dirs]):
        for speaker_dir in speaker_dirs:
            speaker_video_paths = glob.glob(speaker_dir + '/*')
            speaker_video = []
            # strip abs dirs
            for path in speaker_video_paths:
                base, video_name = os.path.split(path)
                base, speaker = os.path.split(base)
                speaker_video.append(os.path.join(speaker, video_name))
            np.random.shuffle(speaker_video)
            # append video
            if idx == 0:
                train_video.extend(speaker_video)
            else:
                test_video.extend(speaker_video)

    #dump to text
    train_save_file = os.path.join(split_save_dir, 'unseen_train.txt')
    test_save_file = os.path.join(split_save_dir, 'unseen_test.txt')
    np.savetxt(train_save_file, train_video, fmt='%s')
    np.savetxt(test_save_file, test_video, fmt='%s')


def main():
    db_base_dir = os.path.join(CURRENT_FILE_DIRECTORY,
                               '../../decompressed/mouth')
    split_save_dir = os.path.join(CURRENT_FILE_DIRECTORY, '../../db-split')
    os.system('mkdir -p {}'.format(split_save_dir))
    overlapped_split(db_base_dir, split_save_dir)
    unseen_split(db_base_dir, split_save_dir)


if __name__ == "__main__":
    main()
