#!/usr/bin/env python

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import util


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784], name="image_inputs")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="actual_class")
    """
    Inputs

    x : 28px by 28px images converted into a [Batch Size * 784] matrix
    y: [Batch Size * 10] matrix of one hot encodings representing the actual class of the image
       (ie. [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ] where the index of 1 is the class)

    """
    with tf.name_scope("conv1"):
        # We arbitrarily choose to use 32 filters (Output Depth) to learn 32 features for each 5x5 patch
        # TODO: Figure out how each filter learns to look for a different feature.
        # TODO: How to choose a number of filters to use. What if... too many/too few?
        W_conv1 = util.weight([5, 5, 1, 32], name="Weight_conv1")
        b_conv1 = util.bias([32], name="bias_conv1")
