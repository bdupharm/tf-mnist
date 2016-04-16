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
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope("summaries"):
        weight_summary = tf.scalar_summary("weight/mean", tf.reduce_mean(W))
        bias_summary = tf.scalar_summary("bias/mean", tf.reduce_mean(b))
        softmax_summary = tf.scalar_summary("softmax/mean", tf.reduce_mean(y))
        loss_summary = tf.scalar_summary("loss/mean", tf.reduce_mean(cross_entropy))

    summary_operation = tf.merge_all_summaries()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter("./summaries/", sess.graph)

    with sess.as_default():

        for step in range(1000):

            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if step % 10 == 0:
                summaries = sess.run(summary_operation, feed_dict={x: batch[0], y_: batch[1]})
                summary_writer.add_summary(summaries, step)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
