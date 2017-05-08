import tensorflow as tf
import numpy as np

def weight_variable(shape):
    """Initializes weights randomly from a normal distribution
    Params: shape: list of dimensionality of tensor
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Initializes the bias term randomly from a normal distribution.
    Params: shape: list of dimensionality for the bias term.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(scope, x, weight_shape, activation = 'relu', keep_prob = 1.0):
    with tf.variable_scope(scope):
        W_fc = weight_variable(weight_shape)
        b_shape = [weight_shape[-1]]
        b_fc = bias_variable(b_shape)
        h_fc = tf.matmul(x, W_fc) + b_fc
        if activation == 'relu': h_fc = tf.nn.relu(h_fc)
        if activation == 'softmax': h_fc = tf.nn.softmax(h_fc)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob=keep_prob) if keep_prob != 1.0 else h_fc
        return h_fc_drop
