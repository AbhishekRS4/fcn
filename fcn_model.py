# code for transfer learning of vgg-16 encoder and trainable fcn decoder
# @author : Abhishek R S

import os
import numpy as np
import tensorflow as tf

class FCN:
    def __init__(self, vgg16_npy_path, num_kernels, conv_kernel_size, conv_same_kernel_size, conv_strides, conv_tr_kernel_size, conv_tr_strides, training_pl, data_format = 'channels_first', num_classes = 2):
        self.data_dict = np.load(vgg16_npy_path, encoding = 'latin1').item()
        self.num_kernels = num_kernels
        self.conv_kernel_size = conv_kernel_size
        self.conv_same_kernel_size = conv_same_kernel_size
        self.conv_strides = conv_strides
        self.conv_tr_kernel_size = conv_tr_kernel_size
        self.conv_tr_strides = conv_tr_strides
        self.training = training_pl
        self.data_format = data_format
        self.num_classes = num_classes
        self.padding = 'SAME'
        self.vgg_strides = [1, 1, 1, 1]
        self.bn_axis = None
        self.vgg_data_format = None
        self.pool_kernel = None
        self.pool_strides = None
        self.initializer = tf.contrib.layers.xavier_initializer_conv2d()
 
        '''
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU as it exploits faster GPU memory access
        '''

        if data_format == 'channels_first':
            self.vgg_data_format = 'NCHW'
            self.pool_kernel = [1, 1, 2, 2]
            self.pool_strides = [1, 1, 2, 2] 
            self.bn_axis = 1
        else:
            self.vgg_data_format = 'NHWC'
            self.pool_kernel = [1, 2, 2, 1]
            self.pool_strides = [1, 2, 2, 1] 
            self.bn_axis = -1
            
        print('VGG-16 weights loaded')

    def vgg_encoder(self, features):
        '''
        load pretrained weights to build the VGG-16 encoder
        input image - bgr image
        '''

        # build the vgg-16 encoder
        self.conv1_1 = self._conv_layer(features, 'conv1_1')
        self.conv1_2 = self._conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self._conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self._conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self._conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self._conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self._conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self._conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self._conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')

    # define the fcn8 decoder
    def fcn8(self):
        self.conv6 = self._get_conv2d_layer(self.pool5, self.num_kernels[4], self.conv_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv6')
        self.bn6 = self._get_batch_norm_layer(self.conv6, name = 'bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name = 'elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name = 'dropout6')

        self.conv7 = self._get_conv2d_layer(self.dropout6, self.num_kernels[4], self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv7')
        self.bn7 = self._get_batch_norm_layer(self.conv7, name = 'bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name = 'elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name = 'dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv8')
        self.bn8 = self._get_batch_norm_layer(self.conv8, name = 'bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name = 'elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(self.elu8, self.num_classes, self.conv_tr_kernel_size[0], self.conv_tr_strides[0], self.padding, self.data_format, name = 'conv_tr1')
        self.conv9 = self._get_conv2d_layer(self.pool4, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv9')
        self.fuse1 = tf.add(self.conv_tr1, self.conv9, name = 'fuse1')

        self.conv_tr2 = self._get_conv2d_transpose_layer(self.fuse1, self.num_classes, self.conv_tr_kernel_size[0], self.conv_tr_strides[0], self.padding, self.data_format, name = 'conv_tr2')
        self.conv10 = self._get_conv2d_layer(self.pool3, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv10')
        self.fuse2 = tf.add(self.conv_tr2, self.conv10, name = 'fuse2')
                
        self.conv_tr3 = self._get_conv2d_transpose_layer(self.fuse2, self.num_classes, self.conv_tr_kernel_size[1], self.conv_tr_strides[1], self.padding, self.data_format, name = 'conv_tr3')
        self.logits = self._get_conv2d_layer(self.conv_tr3, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'logits')
              
    # define the fcn16 decoder
    def fcn16(self):
        self.conv6 = self._get_conv2d_layer(self.pool5, self.num_kernels[4], self.conv_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv6')
        self.bn6 = self._get_batch_norm_layer(self.conv6, name = 'bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name = 'elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name = 'dropout6')

        self.conv7 = self._get_conv2d_layer(self.dropout6, self.num_kernels[4], self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv7')
        self.bn7 = self._get_batch_norm_layer(self.conv7, name = 'bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name = 'elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name = 'dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv8')
        self.bn8 = self._get_batch_norm_layer(self.conv8, name = 'bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name = 'elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(self.elu8, self.num_classes, self.conv_tr_kernel_size[0], self.conv_tr_strides[0], self.padding, self.data_format, name = 'conv_tr1')
        self.conv9 = self._get_conv2d_layer(self.pool4, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv9')
        self.fuse1 = tf.add(self.conv_tr1, self.conv9, name = 'fuse1')
        
        self.conv_tr2 = self._get_conv2d_transpose_layer(self.fuse1, self.num_classes, self.conv_tr_kernel_size[2], self.conv_tr_strides[2], self.padding, self.data_format, name = 'conv_tr2')
        self.logits = self._get_conv2d_layer(self.conv_tr2, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'logits')
    
    # define the fcn32 decoder
    def fcn32(self):
        self.conv6 = self._get_conv2d_layer(self.pool5, self.num_kernels[4], self.conv_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv6')
        self.bn6 = self._get_batch_norm_layer(self.conv6, name = 'bn6')
        self.elu6 = self._get_elu_activation(self.bn6, name = 'elu6')
        self.dropout6 = self._get_dropout_layer(self.elu6, name = 'dropout6')

        self.conv7 = self._get_conv2d_layer(self.dropout6, self.num_kernels[4], self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv7')
        self.bn7 = self._get_batch_norm_layer(self.conv7, name = 'bn7')
        self.elu7 = self._get_elu_activation(self.bn7, name = 'elu7')
        self.dropout7 = self._get_dropout_layer(self.elu7, name = 'dropout7')

        self.conv8 = self._get_conv2d_layer(self.dropout7, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'conv8')
        self.bn8 = self._get_batch_norm_layer(self.conv8, name = 'bn8')
        self.elu8 = self._get_elu_activation(self.bn8, name = 'elu8')

        self.conv_tr1 = self._get_conv2d_transpose_layer(self.elu8, self.num_classes, self.conv_tr_kernel_size[3], self.conv_tr_strides[3], self.padding, self.data_format, name = 'conv_tr1')
        self.logits = self._get_conv2d_layer(self.conv_tr1, self.num_classes, self.conv_same_kernel_size, self.conv_strides, self.padding, self.data_format, name = 'logits')
        
    # return convolution2d layer
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, padding, data_format, name = 'conv'):
        return tf.layers.conv2d(inputs = input_layer, filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, kernel_initializer = self.initializer, name = name)

    # return convolution2d_transpose layer
    def _get_conv2d_transpose_layer(self, input_layer, num_filters, kernel_size, strides, padding, data_format, name = 'conv_tr'):
        return tf.layers.conv2d_transpose(inputs = input_layer, filters = num_filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, kernel_initializer = self.initializer, name = name)

    # return ELU activation function
    def _get_elu_activation(self, input_layer, name = 'elu'):
        return tf.nn.elu(input_layer, name = name)

    # return the dropout layer
    def _get_dropout_layer(self, input_layer, rate = 0.5, name = 'dropout'):
        return tf.layers.dropout(inputs = input_layer, rate = rate, training = self.training, name = name)

    # return batch normalization layer
    def _get_batch_norm_layer(self, input_layer, name = 'bn'):
        return tf.layers.batch_normalization(input_layer, axis = self.bn_axis, training = self.training, name = name)

    # perform maxpool2d in vgg16 encoder
    def _max_pool(self, input_layer, name):
        return tf.nn.max_pool(value = input_layer, ksize = self.pool_kernel, strides = self.pool_strides, padding = self.padding, data_format = self.vgg_data_format, name = name)

    # perform convolution2d in vgg16 encoder
    def _conv_layer(self, input_layer, name):
        with tf.variable_scope(name):
            conv_kernel = self._get_conv_filter(name)
            conv_biases = self._get_bias(name)

            conv = tf.nn.conv2d(input = input_layer, filter = conv_kernel, strides = self.vgg_strides, padding = self.padding, data_format = self.vgg_data_format, name = name)
            bias = tf.nn.bias_add(value = conv, bias = conv_biases, data_format = self.vgg_data_format, name = 'bias' + name[4:])
            relu = tf.nn.relu(bias, name = 'relu' + name[4:])

            return relu

    # get pre-trained vgg16 convolution2d filter
    def _get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name = 'kernel')

    # get pre-trained vgg16 biases
    def _get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name = 'biases')
