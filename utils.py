#
# Utility Functions for Building a Tensorflow Graph
#

import tensorflow as tf


def weight(shape, name="filter"):
    """Initialize a weight tensor of :param shape: with...
     - mean of 0.0
     - standard deviation of 0.1.

    :param shape: Shape of the return tensor
    `[Width, Height, Input Depth (Color Channels), Output Depth]`

    **NOTE**: The `Output Depth` is the # of of filters or kernels we would like to use.
    Each one of these filters will learn a different feature for its `W * H` input patch.
    People either call this the`Depth Column` or `Fibre`.

    Why? Initializing weights w/ a small amount of noise
    can help prevent 0 gradients.

    """
    # mean is 0.0 by default, which means that that tensor is centered on the true value.
    # Imagine a signal that wiggles randomly through time across some limited range,
    # the center of this range is equal to the true value of the tensor
    # If the mean was non-zero, it would create a bias or an offset to the true value
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias(shape, name="bias"):
    """Initialize a bias tensor of :param shape: with a
    slightly positive initial bias of 0.1.

    :param shape: Shape of the return tensor
    `[Output Depth]`

    Why the positive bias? Apparently, if you are using a ReLU activation fx
    then doing so can help avoid "dead neurons".
    See: http://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name="conv"):
    """Takes 4-D input tensor x and applies a sliding window filter W to compute
    a 2-D convolution.

    The dimensions of the input are:
    `[Batch Size, Height, Width, Depth (Color Channels)]`

    Uses a stride of `[1, 1, 1, 1]` where most cases of `strides =  [1, stride, stride, 1]`
     - `stride[0] = stride[3] = 1` (vertical stride) and
     - `stride[1] = stride[2] = stride` (horizontal stride)

     :param x: input tensor
     :param W: filter or kernel (weights) that we will slide over x

    Convolutions are much easier to grasp from a visual perspective so see:
    http://cs231n.github.io/convolutional-networks/
    http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

    Why?
     - Using fully connected layers for high dimensional inputs such as images
       is impractical and, frankly, unnecessary. We don't care about learning
       the probabilities for each and every pixel but general features instead.
     - Draws comparisons to rods/cones in the retina
     - Downsizing reduces amount of parameters/computation in the network
     - Control overfitting

    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)


def max_pool_2x2(x, name="max_pool"):
    """Applies a 2x2 max pooling layer to input tensor x.

    MaxPooling is a form of non-linear down-sampling which reduces the volume spatially
    by eliminating non-maximal values (effectively "pooling" the signifigant values):
    (ie. downsizes a 224x224x64 input tensor to a 112x112x64 ouput tensor)
    Note that pooling preserves the depth of the input.

    Generic downsampling
    ex [1, 3, 5, 7, 9]
    Downsample array by 2 would result in [1, 5, 7]

    Why?
     - Downsampling reduces the dimensionality of parameters/computation in the network
     - Control overfitting
        - Downsampling followed by convolution results in interpolation
          which inherently reduces overfitting

    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME", name=name)
