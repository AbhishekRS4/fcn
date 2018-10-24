# @author : Abhishek R S

import sys
import argparse
import math
import os
import time
import numpy as np
import tensorflow as tf
from fcn_utils import init, read_config_file, get_tf_dataset
import fcn_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
param_config_file_name = os.path.join(os.getcwd(), 'fcn_config.json')

# define metrics 
def compute_metrics(groundtruth, prediction, axis=1, num_classes=15):
    groundtruth = tf.squeeze(groundtruth)
    prediction_labels = tf.argmax(
        tf.nn.softmax(prediction, axis=axis), axis=axis)

    with tf.name_scope('valid_metrics'):
        acc_value, acc_op = tf.metrics.accuracy(groundtruth, prediction_labels)

    return acc_value, acc_op

# define cross entropy loss
def compute_loss(groundtruth, prediction, axis=1, name='mean_cross_entropy'):

    if axis == 1:
        prediction = tf.transpose(prediction, perm=[0, 2, 3, 1])
        groundtruth = tf.transpose(groundtruth, perm=[0, 2, 3, 1])

    groundtruth = tf.squeeze(groundtruth)

    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=groundtruth, logits=prediction), name=name)

    return mean_ce

# return the optimizer op which is used to minimize the loss function
def get_optimizer(initial_learning_rate, loss_function, global_step, epsilon=0.0001):
    decay_steps = 300
    end_learning_rate = 0.000001
    decay_rate = 0.97
    power = 0.95

    learning_rate = tf.train.polynomial_decay(
        initial_learning_rate, global_step, decay_steps, end_learning_rate, power=power)

    adam_optimizer_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate, epsilon=epsilon).minimize(loss_function)

    return adam_optimizer_op

# save the trained model
def save_model(session, model_dir, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), model_dir,
        model_file), global_step=(epoch + 1))

# start batch training of the network
def batch_train(FLAGS):
    print('Initializing.............................')
    model_dir = FLAGS.model_dir + FLAGS.model_to_use + \
        '_' + str(FLAGS.num_epochs)
    init(model_dir)
    print('Initializing completed...................')
    print('')

    print('Preparing training meta data.......................')
    list_train = os.listdir(FLAGS.images_dir_train)
    list_valid = os.listdir(FLAGS.images_dir_valid)

    list_train = list_train
    list_valid = list_valid

    list_images_train = [os.path.join(
        FLAGS.images_dir_train, x) for x in list_train]
    list_labels_train = [os.path.join(FLAGS.labels_dir_train, x.replace(
        'leftImg8bit', 'label')) for x in list_train]

    list_images_valid = [os.path.join(
        FLAGS.images_dir_valid, x) for x in list_valid]
    list_labels_valid = [os.path.join(FLAGS.labels_dir_valid, x.replace(
        'leftImg8bit', 'label')) for x in list_valid]

    num_samples_train = len(list_images_train)
    num_batches_train = int(
        math.ceil(num_samples_train / float(FLAGS.batch_size)))

    num_samples_valid = len(list_images_valid)
    num_batches_valid = int(
        math.ceil(num_samples_valid / float(FLAGS.batch_size)))

    print('Preparing training meta data completed.............')
    print('')

    print('Learning rate : ' + str(FLAGS.learning_rate))
    print('Number of epochs to train : ' + str(FLAGS.num_epochs))
    print('Batch size : ' + str(FLAGS.batch_size))
    print('Number of train samples : ' + str(num_samples_train))
    print('Number of train batches : ' + str(num_batches_train))
    print('Number of validation samples : ' + str(num_samples_valid))
    print('Number of validation batches : ' + str(num_batches_valid))
    print('')

    print('Building the network.....................')
    axis = -1
    if FLAGS.data_format == 'channels_first':
        axis = 1

    dataset_train = get_tf_dataset(
        list_images_train, list_labels_train, FLAGS.num_epochs, FLAGS.batch_size)
    dataset_valid = get_tf_dataset(
        list_images_valid, list_labels_valid, FLAGS.num_epochs, FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes)
    features, labels = iterator.get_next()

    init_op_train = iterator.make_initializer(dataset_train)
    init_op_valid = iterator.make_initializer(dataset_valid)

    is_training = tf.placeholder(tf.bool)
    net_arch = fcn_model.FCN(FLAGS.pretrained_weights,
                             is_training, FLAGS.data_format, FLAGS.num_classes)
    net_arch.vgg16_encoder(features)
    print('vgg-16 encoder built')

    if FLAGS.model_to_use == 'fcn8':
        net_arch.fcn8()
        print('fcn-8 decoder built')
    elif FLAGS.model_to_use == 'fcn16':
        net_arch.fcn16()
        print('fcn-16 decoder built')
    else:
        net_arch.fcn32()
        print('fcn-32 decoder built')

    logits = net_arch.logits

    train_var_list = [v for v in tf.trainable_variables()]

    loss_1 = compute_loss(labels, logits, axis=axis)
    loss_2 = FLAGS.weight_decay * \
        tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    loss = loss_1 + loss_2

    acc_value, acc_op = compute_metrics(
        labels, logits, axis=axis, num_classes=FLAGS.num_classes)

    global_step = tf.placeholder(tf.int32)
    optimizer_op = get_optimizer(FLAGS.learning_rate, loss, global_step)
    extra_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    print('Building the network completed...........')
    print('')

    print('Training the network.....................')
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())
    ss.run(tf.local_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()
    valid_acc_per_epoch = list()
    valid_mean_iou_per_epoch = list()

    for epoch in range(FLAGS.num_epochs):
        ti = time.time()
        temp_train_loss_per_epoch = 0
        temp_valid_loss_per_epoch = 0
        temp_valid_acc_per_epoch = 0
        temp_valid_mean_iou_per_epoch = 0

        ss.run(init_op_train)
        for batch_id in range(num_batches_train):
            _, _, loss_per_batch = ss.run([extra_update_op, optimizer_op, loss], feed_dict={
                                          is_training: True, global_step: epoch})
            temp_train_loss_per_epoch += loss_per_batch

        ss.run(init_op_valid)
        for batch_id in range(num_batches_valid):
            loss_per_batch, _ = ss.run(
                [loss_1, acc_op], feed_dict={is_training: False})
            temp_valid_loss_per_epoch += loss_per_batch

        acc_valid = ss.run(acc_value)
        temp_valid_acc_per_epoch = acc_valid

        stream_vars_valid = [
            v for v in tf.local_variables() if 'valid_metrics' in v.name]
        ss.run(tf.variables_initializer(stream_vars_valid))

        ti = time.time() - ti
        train_loss_per_epoch.append(temp_train_loss_per_epoch)
        valid_loss_per_epoch.append(temp_valid_loss_per_epoch)
        valid_acc_per_epoch.append(temp_valid_acc_per_epoch)

        print(
            'Epoch : {0:d} / {1:d}, time taken : {2:.2f} sec.'.format(epoch + 1, FLAGS.num_epochs, ti))
        print('training loss : {0:.4f}, validation loss : {1:.4f}, validation accuracy : {2:.4f}'.format(
            temp_train_loss_per_epoch / num_batches_train, temp_valid_loss_per_epoch / num_batches_valid, temp_valid_acc_per_epoch))
        print('')

        if (epoch + 1) % FLAGS.checkpoint_epoch == 0:
            save_model(ss, model_dir, FLAGS.model_file, epoch)

    print('Training the network completed...........')
    print('')

    print('Saving the model.........................')
    save_model(ss, model_dir, FLAGS.model_file, epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)

    train_loss_per_epoch = np.true_divide(
        train_loss_per_epoch, num_batches_train)
    valid_loss_per_epoch = np.true_divide(
        valid_loss_per_epoch, num_batches_valid)

    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), model_dir,
                         FLAGS.model_metrics), (losses_dict))

    print('Saving the model completed...............')
    print('')

    ss.close()

def main():
    print('Reading the config file..................')
    config = read_config_file(param_config_file_name)
    print('Reading the config file completed........')
    print('')

    images_dir_train = config['data']['images_dir_train']
    labels_dir_train = config['data']['labels_dir_train']
    images_dir_valid = config['data']['images_dir_valid']
    labels_dir_valid = config['data']['labels_dir_valid']

    pretrained_weights = config['model']['pretrained_weights']
    data_format = config['model']['data_format']
    num_classes = config['model']['num_classes']

    learning_rate = config['training']['learning_rate']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    weight_decay = config['training']['weight_decay']
    checkpoint_epoch = config['training']['checkpoint_epoch']

    model_dir = config['checkpoint']['model_dir']
    model_file = config['checkpoint']['model_file']
    model_metrics = config['checkpoint']['model_metrics']
    model_to_use = config['checkpoint']['model_to_use']

    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir_train', default=images_dir_train,
                        type=str, help='directory with training images')
    parser.add_argument('-labels_dir_train', default=labels_dir_train,
                        type=str, help='directory with training labels')
    parser.add_argument('-images_dir_valid', default=images_dir_valid,
                        type=str, help='directory with validation images')
    parser.add_argument('-labels_dir_valid', default=labels_dir_valid,
                        type=str, help='directory with validation labels')

    parser.add_argument('-pretrained_weights', default=pretrained_weights,
                        type=str, help='full file path for pretrained weights')
    parser.add_argument('-data_format', default=data_format, type=str,
                        choices=['channels_first', 'channels_last'], help='data format')
    parser.add_argument('-num_classes', default=num_classes, type=int,
                        help='number of classes to be considered for training')

    parser.add_argument('-learning_rate', default=learning_rate,
                        type=float, help='learning rate')
    parser.add_argument('-num_epochs', default=num_epochs,
                        type=int, help='number of epochs to train')
    parser.add_argument('-batch_size', default=batch_size,
                        type=int, help='number of samples in a batch')
    parser.add_argument('-weight_decay', default=weight_decay,
                        type=float, help='weight decay')
    parser.add_argument('-checkpoint_epoch', default=checkpoint_epoch,
                        type=int, help='checkpoint epoch to save every kth model')

    parser.add_argument('-model_dir', default=model_dir,
                        type=str, help='directory to save the model')
    parser.add_argument('-model_file', default=model_file,
                        type=str, help='file name to save the model')
    parser.add_argument('-model_metrics', default=model_metrics,
                        type=str, help='file name to save metrics')
    parser.add_argument('-model_to_use', default=model_to_use, type=str,
                        choices=['fcn8', 'fcn16', 'fcn32'], help='model to use')

    FLAGS, unparsed = parser.parse_known_args()

    batch_train(FLAGS)

if __name__ == '__main__':
    main()
