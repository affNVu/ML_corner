# Load libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran

# Declare working directory
cwd = os.getcwd()

# Load dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(cwd+"/MNIST_data/", one_hot=True)

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    X_train = mnist.train.images[:num,:]
    print ('X_train Examples Loaded = ' + str(X_train.shape))
    Y_train = mnist.train.labels[:num,:]
    print ('Y_train Examples Loaded = ' + str(Y_train.shape))
    print('')
    return X_train, Y_train

def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test

def display_digit(num):
    print(Y_train[num])
    label = Y_train[num].argmax(axis=0)
    image = X_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = X_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, X_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

def create_placeholder(n_x, n_y):
	"""
	Arguments:
	n_x -- scalar, size of a flatten image vector
	n_y -- scalar, number of classes

	Returns:
	X -- placeholder for the data of input, of shape [None, n_x]
	Y -- placeholder for the data of output, of shape [None, n_y]
	"""
	X = tf.placeholder(tf.float32, [None, n_x], name="X")
	Y = tf.placeholder(tf.float32, [None, n_y], name="Y")
	return X, Y


def initialize_parameters():
	"""
	Initialise parameters i.e.
	W: [n_x, 10]
	b: [10, 1]
	"""
	# Initialise Variables
	W = tf.Variable(tf.zeros([784,10]))	# Edge weight
	b = tf.Variable(tf.zeros([10]))	# bias
	parameters = {"W": W, "b": b}
	return parameters

def forward_propagation(X, parameters):
	"""
	Forward propagation

	Arguments:
	X -- input data placeholder
	parameters -- python dictionary for the parameters
	"""
	W = parameters['W']
	b = parameters['b']
	# Create the model
	Z = tf.add(tf.matmul(X, W), b)
	return Z


def compute_cost(Z, Y):
	"""
	Compute the cost of the model

	Arguments:
	Z -- output of forward propagation i.e. the final output layer
	Y -- labels

	Returns: Tensor of the cost function
	"""
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z))
	return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
	num_epochs = 1719, minibatch_size = 32, print_cost = True):
	"""
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = , number of training examples = )
    Y_train -- test set, of shape (output size = , number of training examples = )
    X_test -- training set, of shape (input size = , number of training examples = )
    Y_test -- test set, of shape (output size = , number of test examples = )
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
	(m, n_x) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	# Create the placeholder
	X, Y = create_placeholder(n_x, n_y)

	# Initialise the parameters
	parameters = initialize_parameters()

	# Forward propagation
	Z = forward_propagation(X, parameters)

	# Calculate cost
	cost = compute_cost(Z, Y)

	# Back propagation
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initi bathces
	X_train_batches, Y_train_batches = tf.train.slice_input_producer(
                                    [X_train, Y_train],
                                    shuffle=True)
	X_train_minibatch, Y_train_minibatch = tf.train.batch(
		[X_train_batches, Y_train_batches],
		batch_size=minibatch_size,
		allow_smaller_final_batch=True)

	# Init all the variables
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		# Run the initi
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		for epoch in range(num_epochs):
			x_batch = X_train_minibatch.eval()
			y_batch = Y_train_minibatch.eval()
			# RUN
			_ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: x_batch, 
				Y: y_batch})    
			# Print the cost every epoch
			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
			print(sess.run(parameters['b']))
			if epoch == 5:
				return 0

		# End the threads
		coord.request_stop()
		coord.join(threads)

		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# lets save the parameters in a variable
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the correct predictions
		print("Z shape:\t" + str(Z.shape))
		print("Y shape:\t" + str(Y.shape))
		correct_prediction = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

		return parameters

X_train, Y_train = TRAIN_SIZE(55000)
X_test, Y_test = TEST_SIZE(1024)
#display_digit(ran.randint(0, X_train.shape[0]))
#display_mult_flat(0,400)

parameters = model(X_train, Y_train, X_test, Y_test)
#print(parameters)

# Test one random image
from random import randint
num = randint(0, 1000)
img = mnist.test.images[num]
image = img.reshape([1,784])
Z = forward_propagation(image, parameters)
#plt.imshow(img, cmap=plt.get_cmap('gray_r'))
#plt.show()

index = tf.Session().run(tf.argmax(tf.nn.softmax(Z),1))
image_to_show = mnist.test.images[num].reshape([28,28])
plt.imshow(image_to_show, cmap=plt.get_cmap('gray_r'))
plt.show()
print("Algorithm predicts: y = " + str(index))

# See the weight matrix
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = parameters['W'][:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
plt.show()
