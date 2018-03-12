"""Utils for creating datasets.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

----------------
Modifications copyright (C) 2018 Holly Grimm
* modified create_input_pipeline function
* added get_data_files_paths function
* added multilabeldense_to_one_hot function
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from .utils import download_and_extract_zip, download_and_extract_tar
import random

def create_input_pipeline(image_paths, labels, batch_size, n_epochs, shape,
                          crop_factor=1.0, n_threads=4, training=True, randomize=False):
    """Creates a pipefile from a list of image files.
    Includes batch generator/central crop/resizing options.
    The resulting generator will dequeue the images batch_size at a time until
    it throws tf.errors.OutOfRangeError when there are no more images left in
    the queue.

    Parameters
    ----------
    files : list
        List of paths to image files.
    batch_size : int
        Number of image files to load at a time.
    n_epochs : int
        Number of epochs to run before raising tf.errors.OutOfRangeError
    shape : list
        [height, width, channels]
    crop_shape : list
        [height, width] to crop image to.
    n_threads : int, optional
        Number of threads to use for batch shuffling
    """

    # We first create a "producer" queue.  It creates a production line which
    # will queue up the file names and allow another queue to deque the file
    # names all using a tf queue runner.
    # Put simply, this is the entry point of the computational graph.
    # It will generate the list of file names.
    # We also specify it's capacity beforehand.

    # image_paths_tf = tf.convert_to_tensor(image_paths, dtype=string, name='image_paths')
    # labels_tf = tf.convert_to_tensor(labels, dtype=string, name='labels')

    if training:
        # Remove num_epochs to continue indefinitely
        input_queue = tf.train.slice_input_producer(
            [image_paths, labels], shuffle=training)
    else:
        input_queue = tf.train.slice_input_producer(
        [image_paths, labels], num_epochs=n_epochs, shuffle=training)
 
    # We pass the filenames to this object which can read the file's contents.
    # This will create another queue running which dequeues the previous queue.
    file_contents = tf.read_file(input_queue[0])

    # And then have to decode its contents as we know it is a jpeg image
    imgs = tf.image.decode_jpeg(
        file_contents,
        channels=3 if len(shape) > 2 and shape[2] == 3 else 0)

    # We have to explicitly define the shape of the tensor.
    # This is because the decode_jpeg operation is still a node in the graph
    # and doesn't yet know the shape of the image.  Future operations however
    # need explicit knowledge of the image's shape in order to be created.
    imgs.set_shape(shape)

    # Next we'll centrally crop the image to the size of 100x100.
    # This operation required explicit knowledge of the image's shape.

    rsz_shape = [int(shape[0] * crop_factor),
                     int(shape[1] * crop_factor)]
    imgs = tf.image.resize_images(imgs, rsz_shape)

    # TODO: Scale image by 1 +/- .150
    # tf.image.central_crop(imgs, central_fraction)
    # tf.image.resize_image_with_crop_or_pad(imgs, target_height, target_width)
    #uint8image = tf.random_crop(uint8image, (224, 224, 3))

    if randomize:
        imgs = tf.image.random_flip_left_right(imgs)
        imgs = tf.image.random_flip_up_down(imgs, seed=None)
        
        # TODO: Random Rotation
        # random_rot = random.randint(1,5)
        # imgs = tf.image.rot90(imgs)
        
        if (random.randint(1,3) == 1):
            imgs = tf.image.transpose_image(imgs)


    # Now we'll create a batch generator that will also shuffle our examples.
    # We tell it how many it should have in its buffer when it randomly
    # permutes the order.
    min_after_dequeue = len(image_paths) // 10

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    if training:
        # Randomize the order and output batches of batch_size.
        batch, batchlabels, batchfilenames = tf.train.shuffle_batch([imgs, input_queue[1], input_queue[0]],
                                    enqueue_many=False,
                                    batch_size=batch_size,
                                    capacity=capacity,
                                    min_after_dequeue=min_after_dequeue,
                                    num_threads=n_threads)
    else:
        batch, batchlabels, batchfilenames = tf.train.batch([imgs, input_queue[1], input_queue[0]],
                                    enqueue_many=False,
                                    batch_size=batch_size,
                                    capacity=capacity,
                                    num_threads=n_threads,
                                    allow_smaller_final_batch=True)

    # alternatively, we could use shuffle_batch_join to use multiple reader
    # instances, or set shuffle_batch's n_threads to higher than 1.

    return batch, batchlabels, batchfilenames

def get_data_files_paths():
    """
    Returns the input file folders path
    
    :return: list of strings
        The input file paths as list [train_jpg_dir, test_jpg_dir, train_csv_file, test_csv_template_file]
    """

    data_root_folder = os.path.abspath("input/")
    train_jpg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpg_dir = os.path.join(data_root_folder, 'test-jpg')
    train_csv_file = os.path.join(data_root_folder, 'train_tags.csv')
    test_csv_template_file = os.path.join(data_root_folder, 'test_tags_blank.csv')
    return [train_jpg_dir, test_jpg_dir, train_csv_file, test_csv_template_file]


def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors.

    Parameters
    ----------
    labels : array
        Input labels to convert to one-hot representation.
    n_classes : int, optional
        Number of possible one-hot.

    Returns
    -------
    one_hot : array
        One hot representation of input.
    """
    return np.eye(n_classes).astype(np.float32)[labels]

def multilabeldense_to_one_hot(tags, labels_map):
    """Convert class labels from scalars to one-hot vectors.

    Parameters
    ----------
    tags : array
        Input labels to convert to one-hot representation.
    labels_map :
        Number of possible one-hot.

    Returns
    -------
    one_hot : array
        One hot representation of input.
    """
    targets = np.zeros(len(labels_map), np.int)
    for t in tags.split(' '):
        targets[labels_map[t]] = 1
    return targets.tolist()


# TODO: use n_classes instead of n_labels
class DatasetSplit(object):
    """Utility class for batching data and handling multiple splits.

    Attributes
    ----------
    current_batch_idx : int
        Description
    images : np.ndarray
        Xs of the dataset.  Not necessarily images.
    labels : np.ndarray
        ys of the dataset.
    n_classes : int
        Number of possible labels
    num_examples : int
        Number of total observations
    """

    def __init__(self, images, labels):
        """Initialize a DatasetSplit object.

        Parameters
        ----------
        images : np.ndarray
            Xs/inputs
        labels : np.ndarray
            ys/outputs
        """
        self.images = np.array(images).astype(np.float32)
        if labels is not None:
            self.labels = np.array(labels).astype(np.int32)
            self.n_classes = len(np.unique(labels))
        else:
            self.labels = None
        self.num_examples = len(self.images)

    def next_batch(self, batch_size=100):
        """Batch generator with randomization.

        Parameters
        ----------
        batch_size : int, optional
            Size of each minibatch.

        Yields
        ------
        Xs, ys : np.ndarray, np.ndarray
            Next batch of inputs and labels (if no labels, then None).
        """
        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(self.images)))
        epoch_images = self.images[current_permutation, ...]
        if self.labels is not None:
            epoch_labels = self.labels[current_permutation, ...]

        # Then iterate over the epoch
        self.current_batch_idx = 0
        while self.current_batch_idx < len(self.images):
            end_idx = min(self.current_batch_idx + batch_size, len(self.images))
            this_batch = {
                'images':
                epoch_images[self.current_batch_idx:end_idx],
                'labels':
                epoch_labels[self.current_batch_idx:end_idx]
                if self.labels is not None else None
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels']


# TODO: use n_classes instead of n_labels
class Dataset(object):
    """Create a dataset from data and their labels.

    Allows easy use of train/valid/test splits; Batch generator.

    Attributes
    ----------
    all_idxs : list
        All indexes across all splits.
    all_inputs : list
        All inputs across all splits.
    all_labels : list
        All labels across all splits.
    n_classes : int
        Number of labels.
    split : list
        Percentage split of train, valid, test sets.
    test_idxs : list
        Indexes of the test split.
    train_idxs : list
        Indexes of the train split.
    valid_idxs : list
        Indexes of the valid split.
    """

    def __init__(self, Xs, ys=None, split=[1.0, 0.0, 0.0], one_hot=False, n_classes=1):
        """Initialize a Dataset object.

        Parameters
        ----------
        Xs : np.ndarray
            Images/inputs to a network
        ys : np.ndarray
            Labels/outputs to a network
        split : list, optional
            Percentage of train, valid, and test sets.
        one_hot : bool, optional
            Whether or not to use one-hot encoding of labels (ys).
        n_classes : int, optional
            Number of classes represented in ys (used for one hot embedding).
        """
        self.all_idxs = []
        self.all_labels = []
        self.all_inputs = []
        self.train_idxs = []
        self.valid_idxs = []
        self.test_idxs = []
        self.n_classes = n_classes
        self.split = split

        # Now mix all the labels that are currently stored as blocks
        self.all_inputs = Xs
        n_idxs = len(self.all_inputs)
        idxs = range(n_idxs)
        rand_idxs = np.random.permutation(idxs)
        self.all_inputs = self.all_inputs[rand_idxs, ...]
        if ys is not None:
            self.all_labels = ys if not one_hot else dense_to_one_hot(ys, n_classes=n_classes)
            self.all_labels = self.all_labels[rand_idxs, ...]
        else:
            self.all_labels = None

        # Get splits
        self.train_idxs = idxs[:round(split[0] * n_idxs)]
        self.valid_idxs = idxs[len(self.train_idxs):
                               len(self.train_idxs) + round(split[1] * n_idxs)]
        self.test_idxs = idxs[(len(self.valid_idxs) + len(self.train_idxs)):
                              (len(self.valid_idxs) + len(self.train_idxs)
                               ) + round(split[2] * n_idxs)]

    @property
    def X(self):
        """Inputs/Xs/Images.

        Returns
        -------
        all_inputs : np.ndarray
            Original Inputs/Xs.
        """
        return self.all_inputs

    @property
    def Y(self):
        """Outputs/ys/Labels.

        Returns
        -------
        all_labels : np.ndarray
            Original Outputs/ys.
        """
        return self.all_labels

    @property
    def train(self):
        """Train split.

        Returns
        -------
        split : DatasetSplit
            Split of the train dataset.
        """
        if len(self.train_idxs):
            inputs = self.all_inputs[self.train_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.train_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def valid(self):
        """Validation split.

        Returns
        -------
        split : DatasetSplit
            Split of the validation dataset.
        """
        if len(self.valid_idxs):
            inputs = self.all_inputs[self.valid_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.valid_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    @property
    def test(self):
        """Test split.

        Returns
        -------
        split : DatasetSplit
            Split of the test dataset.
        """
        if len(self.test_idxs):
            inputs = self.all_inputs[self.test_idxs, ...]
            if self.all_labels is not None:
                labels = self.all_labels[self.test_idxs, ...]
            else:
                labels = None
        else:
            inputs, labels = [], []
        return DatasetSplit(inputs, labels)

    def mean(self):
        """Mean of the inputs/Xs.

        Returns
        -------
        mean : np.ndarray
            Calculates mean across 0th (batch) dimension.
        """
        return np.mean(self.all_inputs, axis=0)

    def std(self):
        """Standard deviation of the inputs/Xs.

        Returns
        -------
        std : np.ndarray
            Calculates std across 0th (batch) dimension.
        """
        return np.std(self.all_inputs, axis=0)
