#!/usr/bin/env python
"""
An example of implementing multinomial logistic (softmax) regression with a single layer of
perceptrons using Tensorflow

Ouput: Confidence prediction (as an array) of which class an observation in the class belongs to 
"""

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True) 
"""
mnist is a DataSet object containing the following:
    - 55000 images and labels for primary training
    - 5000 images and labesl for iterative validation of training accuracy
    - 10000 images and labels for final testing of trained accuracy
"""
print("Number of images/labels for model:")
print("Primary training: " + str(mnist.train.num_examples))
print("Iterative validation: " + str(mnist.test.num_examples))
print("Final testing: " + str(mnist.validation.num_examples))
print("")

"""
Images are stored as a n-dim array [ n_observations x n_features]
Labels are stored as [n_observations x n_labels]
    where each observation is a one-hot vector
"""
print("Dimensions of the Image and Label tensors: ")
print("Images: " + str(mnist.train.images.shape),"Labels: " + str(mnist.train.labels.shape))

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
    with tf.name_scope("hidden1"):
        W = tf.Variable(tf.zeros([784,10]), name="weights")
        b = tf.Variable(tf.zeros([10]), name="biases")

    # Sigmoid unit
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
    x_i : an image with all its pixels flattened out into a [D x 1] vector
    b : "bias" vector of size [K x 1]
    W : "weight" matrix of size [D * K] (transpose of x)
    """
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1],
                                   name="xentropy")
    """
    Represents the cross-entropy between the true (p) distribution and the estimated (q) distribution

    Defined as:
        H(p,q) = - summation{p(x)*log(q(x))}

    As q converges to p, the product of p*log(q) will increase and therefore, H(p,q) will become
    more negative

    The cross-entropy function p*log(q) represents a a second order equation with a defined minima so gradient descent
    converges to only 1 minima.
    """

    # Loss function    
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    """
    cross_entropy example

    assume inaccurate output:
    output = [0.3, 0.2, 0.5]
    expected = [0, 1, 0]
    cross entropy would be -0.2

    assume accurate output:
    output = [0.3, 0.5, 0.2]
    expected = [0, 1, 0]
    cross entropy would be -0.5

    Notice that the accurate output has a more negative value and therefore favored since
    the loss function aims to minimize the cross entropy
    """

    # SUMMARIES
    tf.scalar_summary(loss.op.name, loss)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        summary_writer = tf.train.SummaryWriter("/tmp/tf-summaries/", sess.graph)

        start = time.time()
        for step in range(1000):
            image_inputs, actual_classes = mnist.train.next_batch(50)
            _, loss_value = sess.run([train_op, loss], feed_dict={x: image_inputs, y_: actual_classes})

            summary_str = sess.run(summary_op, feed_dict={x: image_inputs, y_: actual_classes})
            summary_writer.add_summary(summary_str, step)

            if step % 100 == 0:
                duration = time.time() - start
                print "Step {}: loss = {:.2f} ({:.3f} sec)".format(step, loss_value,
                                                                 duration)


        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        """
        Arguements of the maxima (argmax) refers to the point/s of the 
        domain of a function where the function is maximized
        
        In this context, argmax returns the index of the greatest value in the array
        """

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Calling `Tensor.eval()` == `tf.get_default_session().run(Tensor)`
        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
