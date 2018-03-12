"""multilabel_cnn.
"""
"""
Copyright 2018 Holly Grimm.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import os
import gc
import math
from itertools import chain
from libs import dataset_utils as dsutils
from libs import utils as utils
import IPython.display as ipyd
from sklearn.metrics import fbeta_score
plt.style.use('ggplot')

# Set hyperparameters
epoch_arr = [120, 10, 10]
n_epochs = sum(epoch_arr)
learn_rates = [0.001] * epoch_arr[0]
learn_rates.extend([0.0001] * epoch_arr[1])
learn_rates.extend([0.00001] * epoch_arr[2])
batch_size = 60
input_shape = [256, 256, 3]
resize_factor = 1 # how much input shape should be resized
validation_percent = 0.05 # percent of training data to use for validation
n_filters = [32, 32, 64, 64, 128, 128, 256, 256]
filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
fc_size = 512
keep_prob_train = .25
keep_prob_train_fc1 = .5
ckpt_name='multilabel_cnn.ckpt'

# Get paths
train_jpg_dir, test_jpg_dir, train_csv_file, test_csv_template_file = dsutils.get_data_files_paths()

# Read train CSV file
labels_df = pd.read_csv(train_csv_file)

# Create labels map
labels_sorted = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
labels_map = {l: i for i, l in enumerate(labels_sorted)}

# Create lists of filenames, string labels, and onehot labels for training data
image_paths = []
labels_str_list = []
labels_onehot_list = []
for file_name, labels in labels_df.values:
    image_paths.append('{}/{}.jpg'.format(train_jpg_dir, file_name))
    labels_str_list.append(labels)
    onehotlabels = dsutils.multilabeldense_to_one_hot(labels, labels_map)
    labels_onehot_list.append(onehotlabels)
    
# Read test CSV file
labels_test_df = pd.read_csv(test_csv_template_file)

# Create lists of filenames and onehot labels for test data
test_image_paths = []
test_onehot_list = []
for file_name, labels in labels_test_df.values:
    test_image_paths.append('{}/{}.jpg'.format(test_jpg_dir, file_name))
    test_onehot_list.append([0]* len(labels_map))

# Determine size of validation set
# FIXME: randomize the split of training and validation sets
index_split_train_val = round(len(image_paths) * (1-validation_percent))

# Resize shape
resize_shape = [int(input_shape[0] * resize_factor),
                int(input_shape[1] * resize_factor),
                input_shape[2]
               ]

def build_net(graph, training=True, validation=False):
    """Helper for creating a 2D convolution model.

    Parameters
    ----------
    graph : tf.Graph
        default graph to build model
    training : bool, optional
        if true, use training dataset
    validation : bool, optional
        if true, use validation dataset

    Returns
    -------
    batch : list
        list of images
    batch_labels : list
        list of labels for images
    batch_image_paths : list
        list of paths to image files
    init : tf.group
        initializer functions
    x :
        input image
    y :
        labels
    phase_train : tf.bool
        is training
    keep_prob : tf.float32
        keep probability for conv2d layers
    keep_prob_fc1 :  tf.float32
        keep probability for fully connected layer
    learning_rate : tf.float32
        learning rate
    h : 
        output of sigmoid
    loss : 
        loss
    optimizer : 
        optimizer
    saver : tf.train.Saver

    """

    with graph.as_default():    
        x = tf.placeholder(tf.float32, [None] + resize_shape, 'x')
        # TODO: use len(labels_map)
        y = tf.placeholder(tf.int32, [None, 17], 'y')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        keep_prob_fc1 = tf.placeholder(tf.float32, name='keep_prob_fc1')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Create Input Pipeline for Train, Validation and Test Sets
        if training:
            batch, batch_labels, batch_image_paths = dsutils.create_input_pipeline(
                image_paths=image_paths[:index_split_train_val],
                labels=labels_onehot_list[:index_split_train_val],
                batch_size=batch_size,
                n_epochs=n_epochs,
                shape=input_shape,
                crop_factor=resize_factor,
                training=training,
                randomize=True)
        elif validation:
            batch, batch_labels, batch_image_paths = dsutils.create_input_pipeline(
                image_paths=image_paths[index_split_train_val:],
                labels=labels_onehot_list[index_split_train_val:],
                batch_size=batch_size,
                # only one epoch for test output
                n_epochs=1,
                shape=input_shape,
                crop_factor=resize_factor,
                training=training)    
        else:
            batch, batch_labels, batch_image_paths = dsutils.create_input_pipeline(
                image_paths=test_image_paths,
                labels=test_onehot_list,
                batch_size=batch_size,
                # only one epoch for test output
                n_epochs=1,
                shape=input_shape,
                crop_factor=resize_factor,
                training=training)

        Ws = []
        
        current_input = x

        for layer_i, n_output in enumerate(n_filters):
            with tf.variable_scope('layer{}'.format(layer_i)):
                # 2D Convolutional Layer with batch normalization and relu
                h, W = utils.conv2d(x=current_input,
                                        n_output=n_output,
                                        k_h=filter_sizes[layer_i],
                                        k_w=filter_sizes[layer_i])
                h = tf.layers.batch_normalization(h, training=phase_train)
                h = tf.nn.relu(h, 'relu' + str(layer_i))

                # Apply Max Pooling Every 2nd Layer
                if layer_i % 2 == 0:
                    h = tf.nn.max_pool(value=h,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')

                # Apply Dropout Every 2nd Layer
                if layer_i % 2 == 0:
                    h = tf.nn.dropout(h, keep_prob)

                Ws.append(W)
                current_input = h

        h = utils.linear(current_input, fc_size, name='fc_t')[0]
        h = tf.layers.batch_normalization(h, training=phase_train)
        h = tf.nn.relu(h, name='fc_t/relu')
        h = tf.nn.dropout(h, keep_prob_fc1)

        logits = utils.linear(h, len(labels_map), name='fc_t2')[0]
        h = tf.nn.sigmoid(logits, 'fc_t2')

        # must be the same type as logits
        y_float = tf.cast(y, tf.float32)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=y_float)
        loss = tf.reduce_mean(cross_entropy)

        if training:
            # update moving_mean and moving_variance so it will be available at inference time
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        saver = tf.train.Saver()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        return batch, batch_labels, batch_image_paths, init, x, y, phase_train, keep_prob, keep_prob_fc1, learning_rate, h, loss, optimizer, saver

def get_fbeta_score(y, y_pred):
    """
    Compute F_beta score, the weighted harmonic mean of precision and recall

    Parameters
    ----------
    y : array
        y true
    y_pred : array 
        y predicted 

    Returns
    -------
    fbeta_score : float
    """
    threshold = .2
    return fbeta_score(y, np.array(y_pred) > threshold, beta=2, average='samples')


# Train
graphA = tf.Graph()
batch, batch_labels, batch_image_paths, init, x, y, phase_train, keep_prob, keep_prob_fc1, learning_rate, h, loss, optimizer, saver = build_net(graphA, training=True)

with tf.Session(graph=graphA) as sess:
    sess.run(init)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Restore checkpoint
    if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
        saver.restore(sess, "./" + ckpt_name)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    sample_score = 0.0
    # FIXME: n_files = len(image_paths) - index_split_train_val
    n_files = len(image_paths) - 1
    batches_per_epoch = math.ceil(n_files/batch_size)
    save_step=100
    
    try:
        while not coord.should_stop() and epoch_i < n_epochs:
            batch_i += 1
            train, train_labels, train_image_paths = sess.run(
                        [batch, batch_labels, batch_image_paths])
            train_xs = train / 255.0
            train_cost, y_pred, _ = sess.run([loss, h, optimizer], feed_dict={
                x: train_xs,
                y: train_labels,
                learning_rate: learn_rates[epoch_i],
                phase_train: True,
                keep_prob: keep_prob_train,
                keep_prob_fc1: keep_prob_train_fc1})

            batch_sample_score = get_fbeta_score(train_labels, y_pred)
#             print(batch_i, batch_sample_score)
            cost += train_cost
            sample_score += batch_sample_score
            # TODO: Run validation every epoch, Currently problematic with Tensorflow queues
            if batch_i % batches_per_epoch == 0:
                print('epoch:', epoch_i)
                print('average cost:', cost / batch_i)
                print('average sample score:', sample_score / batch_i)
                cost = 0
                sample_score = 0
                batch_i = 0
                epoch_i += 1
            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver.save(sess, "./" + ckpt_name)
#                            global_step=batch_i,
#                            write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()


    # Wait until all threads have finished.
    coord.join(threads)

def fbeta(true_label, prediction):
   return fbeta_score(true_label, prediction, beta=2, average='samples')

def get_optimal_threshhold(true_label, prediction, iterations = 100):
    best_threshhold = [0.2]*17    
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if  temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    return best_threshhold
                

# Create new graph and run validation
graphV = tf.Graph()
batch, batch_labels, batch_image_paths, init, x, y, phase_train, keep_prob, keep_prob_fc1, learning_rate, h, loss, optimizer, saver = build_net(graphV, validation=True, training=False)

val_predictions = []
val_true_labels = []
val_filenames = []

n_files = index_split_train_val

with tf.Session(graph=graphV) as sess:
    sess.run(init)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, "./" + ckpt_name)

    keep_prob_test = 1.0
    keep_prob_test_fc1 = 1.0 
    
    try:
        while not coord.should_stop():
            validation, validation_labels, validation_imagepaths = sess.run(
                [batch, batch_labels, batch_image_paths])
            validation_xs = validation / 255.0
            predictions = h.eval(feed_dict={x: validation_xs,
                    phase_train: False,
                    keep_prob: keep_prob_test,
                    keep_prob_fc1: keep_prob_test_fc1}, session=sess)
            val_true_labels.extend(validation_labels)
            val_predictions.extend(predictions)
            val_filenames.extend(test_imagepaths)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()
        
    # Wait until all threads have finished.
    coord.join(threads)
                
t1 = get_optimal_threshhold(np.array(val_true_labels), np.array(val_predictions))

# Clean up the session.
sess.close()

print(t1)
print(val_true_labels[0])

# Create new graph and generate predictions for the test data
graphB = tf.Graph()
batch, batch_labels, batch_image_paths, init, x, y, phase_train, keep_prob, keep_prob_fc1, learning_rate, h, loss, optimizer, saver = build_net(graphB, training=False)

testpredictions = []
testfilenames = []

n_files = len(test_image_paths)

with tf.Session(graph=graphB) as sess:
    sess.run(init)

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver.restore(sess, "./" + ckpt_name)

    keep_prob_test = 1.0
    keep_prob_test_fc1 = 1.0

    try:
        while not coord.should_stop():
            test, test_labels, test_imagepaths = sess.run(
                [batch, batch_labels, batch_image_paths])
            test_xs = test / 255.0
            predictions = h.eval(feed_dict={x: test_xs,
                    phase_train: False,
                    keep_prob: keep_prob_test,
                    keep_prob_fc1: keep_prob_test_fc1}, session=sess)

#             thresholds = [0.2] * len(labels_map)
            thresholds = t1
    
            predictions_labels = []
            for prediction in predictions:
                labels = [0]* len(labels_map)
                for i, value in enumerate(prediction):
                    if value > thresholds[i]:
                        labels[i] = 1
                textlabels = [labels_sorted[i] for i, value in enumerate(prediction) if value > thresholds[i]]
                predictions_labels.append(textlabels)

            testpredictions.extend(predictions_labels)
            testfilenames.extend(test_imagepaths)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()
        
    # Wait until all threads have finished.
    coord.join(threads)
    
    # Clean up the session.
    sess.close()