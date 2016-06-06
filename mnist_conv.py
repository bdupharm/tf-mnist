#!/usr/bin/env python

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.Graph().as_default():
    # Inputs
    x = tf.placeholder(tf.float32, shape=[None, 784], name="image_inputs")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="actual_class")
    """
    Placeholder creates a container for an input image using tensorflow's graph.
    We allow the first dimension to be None, since this will eventually
    represent out mini-batches, or how many images we feed into a network
    at a time during training/validation/testing

    x : 28px by 28px images converted into a [(Batch Size * 28^2) x 1] column vector
    y : [Batch Size * 10] matrix of one-hot encodings representing the actual class of the image
       (ie. [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ] where the index of 1 is the class)
    """
    with tf.name_scope("conv1"):
        # We arbitrarily choose to use 32 filters (Output Depth) to learn 32 features for each 5x5 patch
        # TODO: Figure out how each filter learns to look for a different feature.
        # TODO: How to choose a number of filters to use. What if... too many/too few?
        # This paper gives some insight: https://arxiv.org/pdf/1312.1847v2.pdf
        # http://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-parameters

        # It seems that deeper models (more layers) are suggested for when you have
        # a parameter budget. It is more difficult to train an accurate shallow net
        # with multiple kernals than it was to train a deeper net with fewer kernals.
        # It seems that for our case sincd we have a smaller dataset,
        # it would be advisable to add more layers and extract more features
        # then decide if there is a signifigant enough increase in accuracy vs
        # trade-off with computation
        W_conv1 = utils.weight([5, 5, 1, 32], name="Weight_conv1")
        b_conv1 = utils.bias([32], name="bias_conv1")
