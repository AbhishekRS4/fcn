# @author : Abhishek R S

import os
import time
import numpy as np
import cv2
import tensorflow as tf

import fcn_model
from fcn_utils import read_config_file, init

IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(3, 1).T

param_config_file_name = os.path.join(os.getcwd(), 'fcn_config.json')

# apply softmax on logits
def get_softmax_layer(logits, axis = 1, name = 'softmax'):
    probs = tf.nn.softmax(logits, axis = axis, name = name)
    return probs

# parse function for tensorflow dataset api
def parse_fn(img_name):
    img_string = tf.read_file(img_name)
    img = tf.image.decode_png(img_string, channels = 3)
    img = tf.cast(img, dtype = tf.float32)
    img_r, img_g, img_b = tf.split(value = img, axis = 2, num_or_size_splits = 3)
    img = tf.concat(values = [img_b, img_g, img_r], axis = 2)
    img = img - IMAGENET_MEAN
    img = tf.transpose(img, perm = [2, 0, 1])

    return img

# return tf dataset
def get_tf_dataset(images_list, num_epochs = 1, batch_size = 1):
    dataset = tf.data.Dataset.from_tensor_slices((images_list))
    dataset = dataset.map(parse_fn, num_parallel_calls = 8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)

    return dataset

# run inference on test set
def infer():
    print('Reading the config file..................')
    config = read_config_file(param_config_file_name)
    model_to_use = config['model_to_use']
    model_directory = config['model_directory'][model_to_use - 1] + str(config['num_epochs'])
    labels_directory = 'labels_' + str(config['num_epochs'])
    init(os.path.join(model_directory, labels_directory))
    print('Reading the Config File Completed........')
    print('')

    print('Preparing test data.....................')
    test_images_path = config['test_images_path'][1]
    print(test_images_path)
    test_list = os.listdir(test_images_path)
    test_images_list = [os.path.join(test_images_path, x) for x in test_list]
    num_test_samples = len(test_images_list)
    test_dataset = get_tf_dataset(test_images_list, 1, 1) 
    iterator = test_dataset.make_one_shot_iterator()
    test_images = iterator.get_next()

    print('Loading the Network.....................')
 
    axis = -1 
    if config['data_format'] == 'channels_first':
        axis = 1

    training_pl = tf.placeholder(tf.bool)

    net_arch = fcn_model.FCN(config['vgg_path'], config['num_kernels'], config['conv_kernel_size'], config['conv_same_kernel_size'], config['conv_strides'], config['conv_tr_kernel_size'], config['conv_tr_strides'], training_pl, config['data_format'], config['num_classes'])
 
    net_arch.vgg_encoder(test_images)

    if config['model_to_use'] == 1:
        net_arch.fcn8()
    elif config['model_to_use'] == 2:
        net_arch.fcn16()
    else:
        net_arch.fcn32()

    network_logits = net_arch.logits
    probs_prediction = get_softmax_layer(logits = network_logits, axis = axis)
    labels_prediction = tf.argmax(probs_prediction, axis = axis)
    print('Loading the Network Completed...........')
    print('')

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #ss = tf.Session(config = tf.ConfigProto(device_count = {'GPU': 1}))
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), os.path.join(model_directory, config['model_file'] + '-' + str(config['num_epochs']))))

    print('Inference Started.......................')

    for img_file in test_images_list:
        ti = time.time()
        labels_predicted = ss.run(labels_prediction, feed_dict = {training_pl : not(config['training'])})
        ti = time.time() - ti
        print('Time Taken for Inference : ' +str(ti))
        print('')

        labels_predicted = np.transpose(labels_predicted, [1, 2, 0]).astype(np.uint8)
        cv2.imwrite(os.path.join(os.getcwd(), os.path.join(model_directory, os.path.join(labels_directory, 'label_' + img_file.split('/')[-1]))), labels_predicted)

    print('Inference Completed') 

    print('')
    ss.close()

def main():
    infer()

if __name__ == '__main__':
    main()
