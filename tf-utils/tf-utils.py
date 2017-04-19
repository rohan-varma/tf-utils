import tensorflow as tf
import numpy as np
SUPPORTED_ACTS = ['tanh', 'relu', 'sigmoid'] # the activations I usually use
SUPPORTED_POOL = ['max', 'avg']
SUPPORTED_PAD = ['SAME', 'VALID']

def conv2d(x, W):
    """Performs a 2d convolution operation.
    Params:
    x: input tensor
    W: input kernel matrix 
    """
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """Performs a max pooling operation.
    Params:
    x: input tensor
    """
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    """Performs an average pooling operation. 
    Params:
    x: input tensor
    """
    return tf.nn.avg_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def conv_layer(scope, x, filter_dim, pool = 'max', use_lrn = True, reshape_dim_list = None):
    """returns a convolutional layer that applies a convolution operation using a kernel of dimension filter_dim.
    The weights are convolved over the input, after which a bias is applied and the result is sent to a ReLu. 
    Params:
    scope: the variable scope of the layer (ie, conv-layer-1)
    x: the input tensor
    filter_dim: a list of the filter dimensions
    pool: either max or avg
    use_lrn: True if you want local response noramlization applied after the pooling operation.
    reshape_dim_list: If you want, a list of the dimensions to reshape the result to (ie, for prep into feeding to FC layer)
    returns: output tensor after applying all ops to input x
    """
    with tf.variable_scope(scope):
        W_conv1 = tf.get_variable('wconv1', filter_dim, initializer = tf.truncated_normal_initializer(stddev = 0.02))
        b_conv1 = tf.get_variable('wconv2', filter_dim[-1], initializer = tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        assert( pool in SUPPORTED_POOL or pool is None, "Error: unsupported pooling operation")
        if pool == 'max':
            h_pool1 = max_pool_2x2(h_conv1)
        else if pool == 'avg':
            h_pool1 = avg_pool_2x2(h_conv1)
        else:
            h_pool1 = h_conv1
        if use_lrn:
            h_pool1 = tf.nn.local_response_normalization(h_pool1)
        if reshape_dim_list:
            h_pool1 = tf.reshape(h_pool1, reshape_dim_list)
        return h_pool1
    
def fc_layer(scope, x, weight_shape, activation = 'relu', dropout_prob = None):
    """Creates a fully connected layer with support for different activations and dropouts.
    Params:
    scope: the scope for the variables declared
    x: input tensor (ie, original data or output of a previous layer)
    activation: relu, tanh, or sigmoid
    dropout_prob: None if not using dropout, else a number between 0 and 1 inclusive.
    """
    with tf.variable_scope(scope):
        W_fc1 = tf.get_variable('wfc', weight_shape, initializer = tf.truncated_normal_initializer(stddev = 0.02))
        b_fc1 = tf.get_variable('bfc', weight_shape[-1], initializer = tf.constant_initializer(0))
        h_fc1 = tf.matmul(x, W_fc1) + b_fc1
        assert(activation in SUPPORTED_ACTS or activation is None, "Error: unsupported activation request.")
        if activation == 'relu': 
            h_fc1 = tf.nn.relu(h_fc1)
        elif activation == 'tanh':
            h_fc1 = tf.nn.tanh(h_fc1)
        elif activation == 'sigmoid':
            h_fc1 = tf.nn.sigmoid(h_fc1)
        if dropout_prob:
            h_fc1 = tf.nn.dropout(h_fc1, keep_prob = 1 - dropout_prob)
        return h_fc1

def deconv_layer(scope, x, output_shape, kernel_dim, strides = [1,2,2,1], padding = 'SAME', activation = 'relu'):
    """Performs a deconvolution operation for generative models. 
    Params:
    scope: the scope for the variables declared
    x: input tensor
    output_shape: list of the output shape desired from this deconv operation
    kernel_dim: the dimension of the (de) conv kernel to use
    strides: the strides across the input that the kernel should make while (de) convolving
    padding: SAME or VALID
    activation: relu, tanh, or sigmoid. ReLu is advised, unless it's the output layer
     for which tanh should be used.
    """
        assert(activation in SUPPORTED_ACTS or activation is None, "Error: unsupported activation request")
        assert(padding in SUPPORTED_PAD, "Error: unsupported padding request")
        with tf.variable_scope(scope):
            W_conv1 = tf.get_variable('wconv1', [kernel_dim, kernel_dim, output_shape[-1], int(x.get_shape()[-1])], 
                                      initializer = tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('bconv1', [output_shape[-1]], initializer = tf.constant_initializer(.1))
            h_conv1 = tf.nn.conv2d_transpose(x, W_conv1, output_shape = output_shape, strides = strides, 
                                             padding = padding) + b_conv1
            h_conv1 = tf.contrib.layers.batch_norm(inputs = h_conv1, center=True, scale=True, 
                                                   is_training=True, scope = 'batch-norm')
            if activation == 'relu':
                h_conv1 = tf.nn.relu(h_conv1)
            elif activation == 'sigmoid':
                h_conv1 = tf.nn.relu(h_conv1)
            elif actication == 'tanh':
                h_conv1 = tf.nn.tanh(h_conv1)
            return h_conv1

