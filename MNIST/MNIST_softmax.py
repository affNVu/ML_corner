# Load libraries
import os
import numpy as numpy
import tensorflow as tf

# Declare working directory
cwd = os.getcwd()

# Load dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(cwd+"/MNIST_data/", one_hot=True)
# Create placeholder
#x = tf.placeholder(tf.float32, shape=[None, 784])	# Input
x = tf.placeholder(tf.float32, [None, 784])
# Initialise Variables
W = tf.Variable(tf.zeros([784,10]))	# Edge weight
b = tf.Variable(tf.zeros([10]))	# bias

# Create the model
y = tf.nn.softmax(tf.matmul(x, W)) + b

# Define cost
#y_ = tf.placeholder(tf.float32, [None,10])
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Start
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test train
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




