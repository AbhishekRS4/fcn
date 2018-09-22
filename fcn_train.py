# @author : Abhishek R S

import math
import os
import time
import numpy as np
import tensorflow as tf
from fcn_utils import init, read_config_file, get_tf_dataset
import fcn_model 

param_config_file_name = os.path.join(os.getcwd(), 'fcn_config.json')

# return cross entropy loss
def cross_entropy_loss(ground_truth, prediction, axis = 1, name = 'mean_cross_entropy'):

    if axis == 1:
        prediction = tf.transpose(prediction, perm = [0, 2, 3, 1])
        ground_truth = tf.transpose(ground_truth, perm = [0, 2, 3, 1])

    ground_truth = tf.squeeze(ground_truth)
    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = ground_truth, logits = prediction), name = name)
    return mean_ce

# return the optimizer which has to be used to minimize the loss function
def get_optimizer(learning_rate, loss_function, epsilon = 0.0001):
    adam_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = epsilon).minimize(loss_function)

    return adam_optimizer

# save the trained model
def save_model(session, model_directory, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), os.path.join(model_directory, model_file)), global_step = (epoch + 1))

# start batch training of the network
def batch_train():

    print('Reading the config file..................')
    config = read_config_file(param_config_file_name)
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    model_to_use = config['model_to_use']
    print('Reading the config file completed........')
    print('')

    print('Initializing.............................')
    model_directory = config['model_directory'][model_to_use - 1] + str(config['num_epochs'])
    init(model_directory)
    print('Initializing completed...................')
    print('')

    print('Preparing train data.......................')
    train_list = os.listdir(config['train_images_path'])
    valid_list = os.listdir(config['valid_images_path'])

    train_images_list = [os.path.join(config['train_images_path'], x) for x in train_list]
    train_labels_list = [os.path.join(config['train_labels_path'], x.replace('leftImg8bit', 'label')) for x in train_list]

    valid_images_list = [os.path.join(config['valid_images_path'], x) for x in valid_list]
    valid_labels_list = [os.path.join(config['valid_labels_path'], x.replace('leftImg8bit', 'label')) for x in valid_list]

    num_train_samples = len(train_images_list)
    num_train_batches = int(math.ceil(num_train_samples / float(batch_size)))

    num_valid_samples = len(valid_images_list)
    num_valid_batches = int(math.ceil(num_valid_samples / float(batch_size)))

    print('Preparing train data completed.............')
    print('')

    print('Building the network.....................')
    axis = -1
    if config['data_format'] == 'channels_first': 
        axis = 1 
   
    training_pl = tf.placeholder(tf.bool) 
    train_dataset = get_tf_dataset(train_images_list, train_labels_list, num_epochs, batch_size) 
    valid_dataset = get_tf_dataset(valid_images_list, valid_labels_list, num_epochs, batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes) 
    features, labels = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)

    net_arch = fcn_model.FCN(config['vgg_path'], config['num_kernels'], config['conv_kernel_size'], config['conv_same_kernel_size'], config['conv_strides'], config['conv_tr_kernel_size'], config['conv_tr_strides'], training_pl, config['data_format'], config['num_classes'])
    net_arch.vgg_encoder(features)

    if config['model_to_use'] == 1:
        net_arch.fcn8()
    elif config['model_to_use'] == 2:
        net_arch.fcn16()
    else:
        net_arch.fcn32() 
    
    logits = net_arch.logits
    
    weight_decay = 0.0005
    train_var_list = [v for v in tf.trainable_variables()]

    loss_1 = cross_entropy_loss(labels, logits, axis = axis)
    loss_2 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    loss = loss_1 + loss_2 

    learning_rate = config['learning_rate']
    print('Learning rate : ' + str(learning_rate))

    optimizer_op = get_optimizer(learning_rate, loss)
    extra_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
 
    print('Building the network completed...........')
    print('')
    
    print('Number of epochs to train : ' + str(num_epochs))
    print('Batch size : ' + str(batch_size))
    print('Number of train samples : ' + str(num_train_samples))
    print('Number of train batches : ' + str(num_train_batches))
    print('Number of validation samples : ' + str(num_valid_samples))
    print('Number of validation batches : ' + str(num_valid_batches))
    print('')

    print('Training the network.....................')
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #ss = tf.Session(config = tf.ConfigProto(device_count = {'GPU': 1}))
    ss = tf.Session()
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()

    for epoch in range(num_epochs):
        ti = time.time()
        temp_train_loss_per_epoch = 0
        temp_valid_loss_per_epoch = 0
        ss.run(train_init_op)

        for batch_id in range(num_train_batches):
            
            _, _, loss_per_batch = ss.run([extra_update_op, optimizer_op, loss], feed_dict = {training_pl : bool(config['training'])})
            temp_train_loss_per_epoch += loss_per_batch

        ss.run(valid_init_op)

        for batch_id in range(num_valid_batches):

            loss_per_batch = ss.run(loss_1, feed_dict = {training_pl : not(config['training'])})
            temp_valid_loss_per_epoch += loss_per_batch

        ti = time.time() - ti
        train_loss_per_epoch.append(temp_train_loss_per_epoch)
        valid_loss_per_epoch.append(temp_valid_loss_per_epoch)

        print('Epoch : ' + str(epoch + 1) + '/' + str(num_epochs) + ', time taken : ' + str(ti) + ' sec.')
        print('Avg. training loss : ' + str(temp_train_loss_per_epoch / num_train_batches) + ', Avg. validation loss : ' + str(temp_valid_loss_per_epoch / num_valid_batches))
        print('')

        if (epoch + 1) % config['checkpoint_epoch'] == 0:
            save_model(ss, model_directory, config['model_file'], epoch)

    print('Training the network completed...........')
    print('')
    
    print('Saving the model.........................')
    save_model(ss, model_directory, config['model_file'], epoch)
    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)

    train_loss_per_epoch = np.true_divide(train_loss_per_epoch, num_train_batches)
    valid_loss_per_epoch = np.true_divide(valid_loss_per_epoch, num_valid_batches)
   
    losses_dict = dict()
    losses_dict['train_loss'] = train_loss_per_epoch
    losses_dict['valid_loss'] = valid_loss_per_epoch

    files_dict = dict()
    files_dict['train_list'] = train_list
    files_dict['valid_list'] = valid_list

    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, config['model_metrics'])), (losses_dict))
    np.save(os.path.join(os.getcwd(), os.path.join(model_directory, 'train_valid_list.npy')), (files_dict))

    print('Saving the model completed...............')
    print('')

    ss.close()

def main():
    batch_train()

if __name__ == '__main__':
    main()
