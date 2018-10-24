# code for transfer learning of vgg-16 encoder and trainable fcn decoder
# @author : Abhishek R S

import os
import h5py
import numpy as np
import tensorflow as tf

'''
FCN
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition]
  (https://arxiv.org/abs/1409.1556)

- [FCN](https://arxiv.org/pdf/1411.4038)

# Pretrained model weights
- [Download pretrained vgg-16 model]
  (https://github.com/fchollet/deep-learning-models/releases/)

'''

class FCN:
    def __init__(self, pretrained_weights, is_training, data_format='channels_first', num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, 'r')
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = 'SAME'
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._encoder_pool_kernel = None
        self._encoder_pool_strides = None
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        '''
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU as it exploits faster GPU memory access
        '''

        if self._data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._encoder_pool_kernel = [1, 1, 2, 2]
            self._encoder_pool_strides = [1, 1, 2, 2]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = 'NHWC'
            self._encoder_pool_kernel = [1, 2, 2, 1]
            self._encoder_pool_strides = [1, 2, 2, 1]
            self._feature_map_axis = -1

    # build vgg-16 encoder
    def vgg16_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        if self._data_format == 'channels_last':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 1
        self.conv1_1 = self._conv_block(features, 'block1_conv1')
        self.conv1_2 = self._conv_block(self.conv1_1, 'block1_conv2')
        self.pool1 = self._maxpool_layer(self.conv1_2, name='pool1')

        # Stage 2
        self.conv2_1 = self._conv_block(self.pool1, 'block2_conv1')
        self.conv2_2 = self._conv_block(self.conv2_1, 'block2_conv2')
        self.pool2 = self._maxpool_layer(self.conv2_2, name='pool2')

        # Stage 3
        self.conv3_1 = self._conv_block(self.pool2, 'block3_conv1')
        self.conv3_2 = self._conv_block(self.conv3_1, 'block3_conv2')
        self.conv3_3 = self._conv_block(self.conv3_2, 'block3_conv3')
        self.pool3 = self._maxpool_layer(self.conv3_3, name='pool3')

        # Stage 4
        self.conv4_1 = self._conv_block(self.pool3, 'block4_conv1')
        self.conv4_2 = self._conv_block(self.conv4_1, 'block4_conv2')
        self.conv4_3 = self._conv_block(self.conv4_2, 'block4_conv3')
        self.pool4 = self._maxpool_layer(self.conv4_3, name='pool4')

        # Stage 5
        self.conv5_1 = self._conv_block(self.pool4, 'block5_conv1')
        self.conv5_2 = self._conv_block(self.conv5_1, 'block5_conv2')
        self.conv5_3 = self._conv_block(self.conv5_2, 'block5_conv3')
        self.pool5 = self._maxpool_layer(self.conv5_3, name='pool5')

    # define fcn8 decoder
    def fcn8(self):
        self.conv6 = self._get_conv2d_layer(
            self.pool5, 4096, [7, 7], [1, 1], name='conv6')
        self.bn6 = self._get_batchnorm_layer(self.conv6, name='bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name='elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name='dropout6')

        self.conv7 = self._get_conv2d_layer(
            self.dropout6, 4096, [1, 1], [1, 1], name='conv7')
        self.bn7 = self._get_batchnorm_layer(self.conv7, name='bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name='elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name='dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self._num_classes, [
                                            1, 1], [1, 1], name='conv8')
        self.bn8 = self._get_batchnorm_layer(self.conv8, name='bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name='elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(
            self.elu8, self._num_classes, [4, 4], [2, 2], name='conv_tr1')
        self.conv9 = self._get_conv2d_layer(self.pool4, self._num_classes, [
                                            1, 1], [1, 1], name='conv9')
        self.fuse1 = tf.add(self.conv_tr1, self.conv9, name='fuse1')

        self.conv_tr2 = self._get_conv2d_transpose_layer(
            self.fuse1, self._num_classes, [4, 4], [2, 2], name='conv_tr2')
        self.conv10 = self._get_conv2d_layer(self.pool3, self._num_classes, [
                                             1, 1], [1, 1], name='conv10')
        self.fuse2 = tf.add(self.conv_tr2, self.conv10, name='fuse2')

        self.conv_tr3 = self._get_conv2d_transpose_layer(
            self.fuse2, self._num_classes, [16, 16], [8, 8], name='conv_tr3')
        self.logits = self._get_conv2d_layer(self.conv_tr3, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # define fcn16 decoder
    def fcn16(self):
        self.conv6 = self._get_conv2d_layer(
            self.pool5, 4096, [7, 7], [1, 1], name='conv6')
        self.bn6 = self._get_batchnorm_layer(self.conv6, name='bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name='elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name='dropout6')

        self.conv7 = self._get_conv2d_layer(
            self.dropout6, 4096, [1, 1], [1, 1], name='conv7')
        self.bn7 = self._get_batchnorm_layer(self.conv7, name='bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name='elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name='dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self._num_classes, [
                                            1, 1], [1, 1], name='conv8')
        self.bn8 = self._get_batchnorm_layer(self.conv8, name='bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name='elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(
            self.elu8, self._num_classes, [4, 4], [2, 2], name='conv_tr1')
        self.conv9 = self._get_conv2d_layer(self.pool4, self._num_classes, [
                                            1, 1], [1, 1], name='conv9')
        self.fuse1 = tf.add(self.conv_tr1, self.conv9, name='fuse1')

        self.conv_tr2 = self._get_conv2d_transpose_layer(
            self.fuse1, self._num_classes, [32, 32], [16, 16], name='conv_tr2')
        self.logits = self._get_conv2d_layer(self.conv_tr2, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # define fcn32 decoder
    def fcn32(self):
        self.conv6 = self._get_conv2d_layer(
            self.pool5, 4096, [7, 7], [1, 1], name='conv6')
        self.bn6 = self._get_batchnorm_layer(self.conv6, name='bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name='elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name='dropout6')

        self.conv7 = self._get_conv2d_layer(
            self.dropout6, 4096, [1, 1], [1, 1], name='conv7')
        self.bn7 = self._get_batchnorm_layer(self.conv7, name='bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name='elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name='dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self._num_classes, [
                                            1, 1], [1, 1], name='conv8')
        self.bn8 = self._get_batchnorm_layer(self.conv8, name='bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name='elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(
            self.elu8, self._num_classes, [64, 64], [32, 32], name='conv_tr1')
        self.logits = self._get_conv2d_layer(self.conv_tr1, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # return convolution2d layer
    def _get_conv2d_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv'):
        return tf.layers.conv2d(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return transposed_convolution2d layer
    def _get_conv2d_transpose_layer(self, input_tensor, num_filters, kernel_size, strides, name='conv_tr'):
        return tf.layers.conv2d_transpose(inputs=input_tensor, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return ELU activation function
    def _get_elu_activation(self, input_tensor, name='elu'):
        return tf.nn.elu(input_tensor, name=name)

    # return dropout layer
    def _get_dropout_layer(self, input_tensor, rate=0.5, name='dropout'):
        return tf.layers.dropout(inputs=input_tensor, rate=rate, training=self._is_training, name=name)

    # return batch_normalization layer
    def _get_batchnorm_layer(self, input_tensor, name='bn'):
        return tf.layers.batch_normalization(input_tensor, axis=self._feature_map_axis, training=self._is_training, name=name)

    #-------------------------------------#
    # pretrained vgg-16 encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_block(self, input_layer, name):
        W = tf.constant(self._weights_h5[name][name + '_W_1:0'])
        b = self._weights_h5[name][name + '_b_1:0']
        b = tf.constant(np.reshape(b, (b.shape[0])))

        x = tf.nn.conv2d(input_layer, filter=W, strides=self._encoder_conv_strides,
                         padding=self._padding, data_format=self._encoder_data_format, name=name + '_conv')
        x = tf.nn.bias_add(
            x, b, data_format=self._encoder_data_format, name=name + '_bias')
        x = tf.nn.relu(x, name=name + '_relu')

        return x

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize=self._encoder_pool_kernel, strides=self._encoder_pool_strides,
                              padding=self._padding, data_format=self._encoder_data_format, name=name)

        return pool
