#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: cnn_extractor.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/10
#   description:
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.keras as keras


class CNN(object):
    """base cnn model. Extract feature of the video_tensor.

    Input:
        video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.
    Output: Tensor of shape(T, feature_len)

    """

    def __init__(self, feature_len, training, scope='cnn_feature_extractor'):
        self.feature_len = feature_len
        self.training = training
        self.scope = scope

    def build():
        raise NotImplementedError('CNN not NotImplemented.')


class LipNet(CNN):
    """lipnet cnn feature extractor"""

    def __init__(self, *args, **kwargs):
        super(LipNet, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope):
            self.zero1 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero1')(video_tensor)
            self.conv1 = keras.layers.Conv3D(
                32, (3, 5, 5),
                strides=(1, 2, 2),
                kernel_initializer='he_normal',
                name='conv1')(self.zero1)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
                # self.conv1, training=self.training)
            self.batc1 = tf.layers.batch_normalization(self.conv1, training=self.training, name= 'batc1') 
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)

            self.zero2 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero2')(self.maxp1)
            self.conv2 = keras.layers.Conv3D(
                64, (3, 5, 5),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv2')(self.zero2)
            # self.batc2 = keras.layers.BatchNormalization(name='batc2')(
                # self.conv2, training=self.training)
            self.batc2 = tf.layers.batch_normalization(self.conv2, training=self.training, name= 'batc2') 
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxp2 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max2')(self.drop2)

            self.zero3 = keras.layers.ZeroPadding3D(
                padding=(1, 1, 1), name='zero3')(self.maxp2)
            self.conv3 = keras.layers.Conv3D(
                96, (3, 3, 3),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv3')(self.zero3)
            # self.batc3 = keras.layers.BatchNormalization(name='batc3')(
                # self.conv3, training=self.training)
            self.batc3 = tf.layers.batch_normalization(self.conv3, training=self.training, name= 'batc3') 
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max3')(self.drop3)

            # prepare output
            self.conv4 = keras.layers.Conv3D(
                self.feature_len, (1, 1, 1),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv4')(self.maxp3) 
            self.output = keras.layers.TimeDistributed(
                keras.layers.GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(self.conv4) #shape: (T, feature_len) 
            return self.output
