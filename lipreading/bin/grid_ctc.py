#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: grid_ctc.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/10
#   description:
#
#================================================================

import os
import argparse
import tensorflow as tf

from lipreading.dataset.grid_dataset import grid_tfrecord_input_fn
from lipreading.model.ctc_estimator import CtcEstimator

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='either train, eval, predict')
    parser.add_argument('type', help='either `unseen` or `overlapped`')

    # train
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='steps interval to save checkpoint')
    parser.add_argument('--model_dir', help='directory to save checkpoints')

    # eval
    parser.add_argument(
        '--eval_steps', type=int, default=100, help='steps to eval')
    # eval and predict
    parser.add_argument(
        '--ckpt_path', help='checkpoints to evaluate/predict', default=None)

    # misc
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='')
    parser.add_argument('-bw', '--beam_width', type=int, default=4)
    parser.add_argument(
        '-use_tcn',
        type=bool,
        default=True,
        help='whether replace rnn with tcn')
    return parser.parse_args()


def main():
    args = arg_parse()
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    multi_gpu = len(args.gpu.split(',')) > 1
    # build estimator
    run_config = CtcEstimator.get_runConfig(
        args.model_dir,
        args.save_steps,
        multi_gpu=multi_gpu,
        keep_checkpoint_max=100)

    if args.use_tcn:
        model_parms = {
            'feature_len': 256,
            'gru_layer_num': 4,
            'gru_unit': 256,
            'use_tcn': True,
            'beam_width': args.beam_width,
            'learning_rate': 0.01
        }
    else:
        model_parms = {
            'feature_len': 256,
            'gru_layer_num': 2,
            'gru_unit': 256,
            'use_tcn': False,
            'beam_width': args.beam_width,
            'learning_rate': 0.001
        }

    model = CtcEstimator(model_parms, run_config)

    # build input
    train_file = os.path.join(
        CURRENT_FILE_DIRECTORY,
        '../../data/tf-records/GRID/{}_train.tfrecord'.format(args.type))
    test_file = os.path.join(
        CURRENT_FILE_DIRECTORY,
        '../../data/tf-records/GRID/{}_test.tfrecord'.format(args.type))

    train_input_params = {
        'num_epochs': 100,
        'batch_size': 50,
        'num_threads': 4,
        'file_name_pattern': train_file
    }
    eval_input_params = {
        'num_epochs': 1,
        'batch_size': 50,
        'num_threads': 4,
        'file_name_pattern': test_file
    }
    train_input_fn = lambda: grid_tfrecord_input_fn(mode=tf.estimator.ModeKeys.TRAIN, **train_input_params)
    eval_input_fn = lambda: grid_tfrecord_input_fn(mode=tf.estimator.ModeKeys.EVAL, **eval_input_params)

    #begin train,eval,predict
    if args.mode == 'train':
        model.train_and_evaluate(
            train_input_fn, eval_input_fn, eval_steps=args.eval_steps)
    elif args.mode == 'eval':
        res = model.evaluate(
            eval_input_fn,
            steps=args.eval_steps,
            checkpoint_path=args.ckpt_path)
        print(res)
    elif args.mode == 'predict':
        model.predict(eval_input_fn, checkpoint_path=args.ckpt_path)
    else:
        raise ValueError(
            'arg mode should be one of "train", "eval", "predict"')


if __name__ == "__main__":
    main()
