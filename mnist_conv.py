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
        # TODO: What if spatial size is not a square matrix? Yeezus tell me
        # This paper gives some insight: https://arxiv.org/pdf/1312.1847v2.pdf
        # http://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-parameters

        # It seems that deeper models (more layers) are suggested for when you have
        # a parameter budget. It is more difficult to train an accurate shallow net
        # with multiple kernals than it was to train a deeper net with fewer kernals.
        # It seems that for our case sincd we have a smaller dataset,
        # it would be advisable to add more layers and extract more features
        # then decide if there is a signifigant enough increase in accuracy vs
        # trade-off with computation
        W_conv1 = utils.weight([5, 5, 1, 32], name="weight_conv1")
        b_conv1 = utils.bias([32], name="bias_conv1")

        """
        In order to calculate the spatial output of the convolutional layer, the following
        equation can be used `[W - F + 2P]/S + 1` where:

        W : Spatial size of input (ie. input tensor of 28*28*1 has a spatial size of 28)
        F : Spatial size of kernel/filter/weights (ie. filter size of 5*5*1 has a spatial size of 5)
        P : Zero-padding (Can ensure that the output spatial size == input spatial size)
        S : Stride

        ie. in our example the values would be:
         W = 28
         F = 5
         P = 0
         S = 1

        The resulting spatial size of our output is 24*24*32.
        Each one of these 576 neurons (each one has a depth column of 32).
        Additionally, each one of these neurons is connected to a 5*5*1 field of input.


        """

        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(utils.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = utils.max_pool_2x2(h_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = utils.weight([5, 5, 32, 64])
        b_conv2 = utils.bias([64])
        h_conv2 = tf.nn.relu(utils.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = utils.max_pool_2x2(h_conv2)

    with tf.name_scope("fc"):
        W_fc1 = utils.weight([7 * 7 * 64, 1024])
        b_fc1 = utils.bias([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = utils.weight([1024, 10])
    b_fc2 = utils.bias([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

