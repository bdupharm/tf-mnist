#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)
    '''
    function in the form of:
        f(x_i, W, b) = Wx_i + b
        which is a linear mapping of image pixels to class scores. W and b are the 
        parameters of this function which change after each iteration
    
    Variables 
    x_i : an image with all its pixels flattened out into a [D x 1] column vector
    b : "bias" vector of size [K x 1]
    W : "weight" matrix of size [K x D]
    '''

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    '''
    Represents the cross-entropy between the p distribution and the estimated distribution q

    Defined as:
        H(p,q) = - summation{p(x)*log(q(x))}

    This represents a a second order equation with a defined minima so gradient descent 
    converges to only 1 minima. 
    '''
    
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    '''
    cross_entropy example

    assume inaccurate output:
    output = [0.3, 0.2, 0.5] expected = [0, 1, 0]
    cross entropy would be -0.2

    assume accurate output:
    output = [0.3, 0.5, 0.2] expected = [0, 1, 0]
    cross entropy would be -0.5

    Notice that the accurate output has a more negative value and therefore favored since
    the loss function aims to minimize the cross entropy
    '''
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    with sess.as_default():
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
