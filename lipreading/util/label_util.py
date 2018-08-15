#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: __init__.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/10
#   description:
#
#================================================================

import string
import tensorflow as tf

DICT = list(string.ascii_lowercase) + [' ', '_']


def indices2string(predictions):
    """map numeric int64 indices to char values.
    For example:
        [ 0, 1, 2] -> [ 'a', 'b', 'c']

    Args:
        predictions: Tensor or SparseTenser. int64 indices to convert

    Returns: The same type of input. map the corresponding values in the DICT.
    """
    if predictions.dtype != tf.int64:
        predictions = tf.cast(predictions, tf.int64)
    index2string_table = tf.contrib.lookup.index_to_string_table_from_tensor(
        DICT, default_value='_')
    return index2string_table.lookup(predictions)


def string2indices(s):
    """ map the char values in `s` to numeric int64 indices
    For example:
        [ 'a', 'b', 'c'] -> [ 0, 1, 2]

    Args:
        s: Tensor or SparseTensor.

    Returns: The same type of input. the Dtype is `tf.int64`

    """
    string2index_table = tf.contrib.lookup.index_table_from_tensor(
        DICT, num_oov_buckets=1, default_value=-1)
    return string2index_table.lookup(s)


def string_join(string_tensor):
    """TODO: Docstring for string_join.

    Args:
        string_tensor: 1-D or 2-D string Tensor. join the strings in the last dimension

    Returns: 0-D or 1-D string Tensor. The joined string.

    """
    i = tf.constant(0)
    string_num = tf.shape(string_tensor)[-1]
    if tf.rank(string_tensor) == 1:
        joined_string = ''
    else:
        joined_string = tf.tile([''],
                                tf.expand_dims(tf.shape(string_tensor)[0], 0))

    c = lambda i, joined_string: tf.less(i, string_num)
    b = lambda i, joined_string: [tf.add(i,1), joined_string+string_tensor[...,i]]
    i, joined_string = tf.while_loop(c, b, [i, joined_string], back_prop=False)
    return joined_string


def char_list2string(char_list):
    """convert a char based list to string. For example:
    [ 'a', 'b'] -> [ 'ab']

    Args:
        char_list: Tenosr or SparseTensor of rank 2.

    Returns: Tensor of rank 1, dtype `string`.

    """
    #char_list: SparseTensor: [ ['a', 'b'], ['a','b','c'] ]
    if isinstance(char_list, tf.SparseTensor):
        dense_list = tf.sparse_tensor_to_dense(
            char_list, '')  # Tensor: [[ 'a' , 'b', ''], [ 'a', 'b', 'c'] ]
    else:
        dense_list = char_list

    return string_join(dense_list)


def string2char_list(string):
    """convert a string to char based list. For example:
    [ 'ab'] -> [ 'a', 'b']

    Args:
        string: `1-D` string `Tensor`. The string to convert

    Returns: `2-D` string `SparseTensor`. The string are split to list of chars.
    """
    return tf.string_split(string, delimiter='')


def char2word(predicted_char_list):
    """ convert char based string to word based string. Suppose the words are divided with space ' '.
    For example: [ ['a', 'b', ' ', 'd']] convert to [['ab', 'd'] ]

    Args:
        predicted_char_list: SparseTensor. dense_shape (batch_size, string_char_len)

    Returns: SparseTensor. dense_shape (batch_size, string_word_len)

    """
    char_string = char_list2string(predicted_char_list)  # [['ab d']]
    word_list = tf.string_split(char_string)  # [[ 'ab', 'd']]
    return word_list
