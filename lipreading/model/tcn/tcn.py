#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tcn.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/08/13
#   description: ref to https://medium.com/the-artificial-impos
#                tor/notes-understanding-tensorflow-part-3-7f66
#                33fcc7c7
#
#================================================================

import tensorflow as tf
from .nn import CausalConv1D


class TemporalBlock(tf.layers.Layer):
    def __init__(self,
                 n_outputs,
                 kernel_size,
                 strides,
                 dilation_rate,
                 dropout=0.2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs)
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs,
            kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=tf.nn.relu,
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs,
            kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=tf.nn.relu,
            name="conv2")
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(
            self.dropout,
            [tf.constant(1),
             tf.constant(1),
             tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(
            self.dropout,
            [tf.constant(1),
             tf.constant(1),
             tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)


class TemporalConvNet(tf.layers.Layer):
    def __init__(self,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable,
            dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name,
            **kwargs)
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(
                    out_channels,
                    kernel_size,
                    strides=1,
                    dilation_rate=dilation_size,
                    dropout=dropout,
                    name="tblock_{}".format(i)))

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs
