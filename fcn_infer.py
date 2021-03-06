# @author : Abhishek R S

import sys
import argparse
import os
import time
import numpy as np
import cv2
import tensorflow as tf

import fcn_model
from fcn_utils import read_config_file, init

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 3)

param_config_file_name = os.path.join(os.getcwd(), 'fcn_config.json')

# get softmax layer
def get_softmax_layer(logits, axis=1, name='softmax'):
    probs = tf.nn.softmax(logits, axis=axis, name=name)
    return probs

# parse function for tensorflow dataset api
def parse_fn(img_name):
    img_string = tf.read_file(img_name)
    img = tf.image.decode_png(img_string, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img_r, img_g, img_b = tf.split(value=img, axis=2, num_or_size_splits=3)
    img = tf.concat(values=[img_b, img_g, img_r], axis=2)
    img = img - IMAGENET_MEAN
    img = tf.transpose(img, perm=[2, 0, 1])

    return img

# return tf dataset
def get_tf_dataset(images_list, num_epochs=1, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((images_list))
    dataset = dataset.map(parse_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)

    return dataset

# run inference on test set
def infer(FLAGS):
    print('Initializing..................')
    model_dir = FLAGS.model_dir + FLAGS.model_to_use + \
        '_' + str(FLAGS.num_epochs)
    labels_dir = 'labels_' + FLAGS.which_set + \
        '_' + str(FLAGS.which_checkpoint_model)
    init(os.path.join(model_dir, labels_dir))
    print('Initializing completed........')
    print('')

    print('Preparing inference meta data.....................')
    images_dir_infer = os.path.join(FLAGS.data_dir, FLAGS.which_set)
    list_infer = os.listdir(images_dir_infer)
    list_images_infer = [os.path.join(images_dir_infer, x) for x in list_infer]
    num_samples_infer = len(list_images_infer)
    print('Preparing inference meta data completed...........')
    print('')

    print('Loading the network.....................')
    test_dataset = get_tf_dataset(list_images_infer, 1, 1)
    iterator = test_dataset.make_one_shot_iterator()
    image_features_infer = iterator.get_next()

    axis = -1
    if FLAGS.data_format == 'channels_first':
        axis = 1

    training_pl = tf.placeholder(tf.bool)
    net_arch = fcn_model.FCN(FLAGS.pretrained_weights,
                             training_pl, FLAGS.data_format, FLAGS.num_classes)
    net_arch.vgg16_encoder(image_features_infer)

    if FLAGS.model_to_use == 'fcn8':
        net_arch.fcn8()
    elif FLAGS.model_to_use == 'fcn16':
        net_arch.fcn16()
    else:
        net_arch.fcn32()

    network_logits = net_arch.logits
    probs_prediction = get_softmax_layer(logits=network_logits, axis=axis)
    labels_prediction = tf.argmax(probs_prediction, axis=axis)
    print('Loading the Network Completed...........')
    print('')

    print('Running inference on following data : ' + images_dir_infer)

    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    tf.train.Saver().restore(ss, os.path.join(os.getcwd(), model_dir,
                                              FLAGS.model_file + '-' + str(FLAGS.which_checkpoint_model)))

    print('Inference Started.......................')

    for img_file in list_images_infer:
        ti = time.time()
        labels_predicted = ss.run(
            labels_prediction, feed_dict={training_pl: False})
        ti = time.time() - ti
        print('Time Taken for Inference : ' + str(ti))
        print('')

        labels_predicted = np.transpose(
            labels_predicted, [1, 2, 0]).astype(np.uint8)
        cv2.imwrite(os.path.join(os.getcwd(), model_dir, labels_dir,
                                 'label_' + img_file.split('/')[-1]), labels_predicted)

    print('Inference Completed')

    print('')
    ss.close()


def main():
    print('Reading the config file..................')
    config = read_config_file(param_config_file_name)
    print('Reading the config file completed........')
    print('')

    pretrained_weights = config['model']['pretrained_weights']
    data_format = config['model']['data_format']
    num_classes = config['model']['num_classes']

    model_dir = config['checkpoint']['model_dir']
    model_file = config['checkpoint']['model_file']
    model_to_use = config['checkpoint']['model_to_use']
    num_epochs = config['training']['num_epochs']

    data_dir = config['inference']['data_dir']
    which_checkpoint_model = config['inference']['which_checkpoint_model']
    which_set = config['inference']['which_set']

    parser = argparse.ArgumentParser()
    parser.add_argument('-pretrained_weights', default=pretrained_weights,
                        type=str, help='full file path for pretrained weights')
    parser.add_argument('-data_format', default=data_format, type=str,
                        choices=['channels_first', 'channels_last'], help='data format')
    parser.add_argument('-num_classes', default=num_classes, type=int,
                        help='number of classes to be considered for training')

    parser.add_argument('-model_dir', default=model_dir,
                        type=str, help='directory to load the model')
    parser.add_argument('-model_file', default=model_file,
                        type=str, help='file name to load the model')
    parser.add_argument('-model_to_use', default=model_to_use, type=str,
                        choices=['fcn8', 'fcn16', 'fcn32'], help='model to use')
    parser.add_argument('-num_epochs', default=num_epochs, type=int,
                        help='used for correctly fetching model directory')

    parser.add_argument('-data_dir', default=data_dir,
                        type=str, help='base data directory')
    parser.add_argument('-which_checkpoint_model', default=which_checkpoint_model,
                        type=str, help='checkpoint model to use to run inference')
    parser.add_argument('-which_set', default=which_set, type=str,
                        choices=['train', 'valid', 'test'], help='data to use to run inference')

    FLAGS, unparsed = parser.parse_known_args()

    infer(FLAGS)


if __name__ == '__main__':
    main()
