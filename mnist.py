#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    """
    function in the form of:
        f(x_i, W, b) = Wx_i + b
        which is a linear mapping of image pixels to class scores.
        W and b are the parameters of this function which change after each iteration

        1) W * x_i => [ 0.2, 0.5, 0.6, 0.3, 1.2, .5, .2, .9, .2, .6]  # does not sum to 1
           return a K element array representing the probabilities that an image belongs to each class K

        2) + b => Adds biases to each of the classes

        3) softmax() => [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # sums to 1
           returns a K element array w/ normalized probabilities that an image belongs to each class K

    Variables (Learning Parameters)
    x_i : an image with all its pixels flattened out into a [D] vector
    b : "bias" vector of size [K]
    W : "weight" matrix of size [D * K] (transpose of x)

    """
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    """
    Represents the cross-entropy between the p distribution and the estimated distribution q

    Defined as:
        H(p,q) = - summation{p(x)*log(q(x))}

    This represents a a second order equation with a defined minima so gradient descent
    converges to only 1 minima.

    """
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    """
    cross_entropy example

    assume inaccurate output:
    output = [0.3, 0.2, 0.5] expected = [0, 1, 0]
    cross entropy would be -0.2

    assume accurate output:
    output = [0.3, 0.5, 0.2] expected = [0, 1, 0]
    cross entropy would be -0.5

    Notice that the accurate output has a more negative value and therefore favored since
    the loss function aims to minimize the cross entropy

    """
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            import ipdb;ipdb.set_trace()

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
