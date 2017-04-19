from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from tf_utils.utils import *
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("done loading mnist")

# define placeholders for our training variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
dropout_prob = tf.placeholder(tf.float32)

h_fc1_drop = fc_layer('fc-layer-1', x, [784, 100], activation = 'relu', dropout_prob = dropout_prob)
h_fc2_drop = fc_layer('fc-layer-2', h_fc1_drop, [100, 30], activation = 'relu', dropout_prob = dropout_prob)
y_pred = fc_layer('fc-layer-3', h_fc2_drop, [30, 10], activation = None)
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_pred)
opt = tf.train.MomentumOptimizer(1e-4, 0.5, name='Momentum', use_nesterov=True).minimize(cross_entropy_loss)
cp = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(cp, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
	print("training!")
	sess.run(init)
	for i in range(10000):
		batch = mnist.train.next_batch(100)
		sess.run(opt, feed_dict = {x: batch[0], y_: batch[1], dropout_prob: 0.5})
		if i % 100 == 0:
			print("epoch: {}".format(i + 1))
			print(acc.eval(feed_dict = {x: batch[0], y_: batch[1], dropout_prob:0.5}))
	print("done training!")
	test_acc = acc.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, dropout_prob: 0.0})
	print("test acc: ".format(test_acc))
sess.close()