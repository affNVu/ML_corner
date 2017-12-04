# Load libraries
import math
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran

# Declare working directory
cwd = os.getcwd()

# Load dataset
from random import randint
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets(cwd+"/MNIST_data/", one_hot=True)


# -------- I/O Helper functions -----------

# - Read and parse input from Kaggle
def load_data_set(dir_path):
	file_path = cwd + dir_path  # Folder contain the Kaggle dataset
	with open(file_path) as f:
		# - Read line
		content = f.readlines()

		# - Prepare X and Y ndarray
		m = len(content)-1
		X = np.zeros([m, 784])
		Y = np.zeros([m, 10])

		for i in range(m):
			img_vals = content[i+1].split(',')	# Skip the first LINE which is the label

			# One hot encoding for the label of data
			label = np.zeros([10])
			number = int(img_vals[0])
			label[number] += 1

			# Put in correct place
			Y[i] = label
			X[i] = [int(s) for s in img_vals[1:785]] 
		return X, Y

# - Given the whole training set, we split the set into train/test sets
# - The size of the train set is: 40000
# - The size of the test set is: 2000
def split_training_set(X, Y):
	m = X.shape[0]
	train_size = 40000
	# Shuffle
	permutation = list(np.random.permutation(m))
	i = permutation[0]
	shuffled_X = X[permutation, :]
	shuffled_Y = Y[permutation, :]

	# Cut the whole set into train/test set
	X_train = shuffled_X[0:train_size,:]
	Y_train = shuffled_Y[0:train_size,:]
	X_test = shuffled_X[train_size:m, :]
	Y_test = shuffled_Y[train_size:m, :]

	# Return result
	return X_train, Y_train, X_test, Y_test

def load_training_set(dir_path = '/Kaggle/train.csv'):
	X_whole, Y_whole = load_data_set(dir_path)
	""" Training """
	print("Whole training set is loaded.")
	print("-----------------------------")
	print("Training shape:" + str(X_whole.shape)) # Training size from Kaggle is [42000, 784] i.e. each image is size 28*28
	print("Splitting training set.")
	(X_train, Y_train, X_test, Y_test) = split_training_set(X_whole, Y_whole)
	print("-----------------------------")
	print("X_train shape:" + str(X_train.shape)) 
	print("Y_train shape:" + str(Y_train.shape)) 
	print("X_test shape:" + str(X_test.shape)) 
	print("Y_test shape:" + str(Y_test.shape)) 
	return X_train, Y_train, X_test, Y_test

# --------- SOME FUNCTION FOR LOAD/Display Pictures -------------------------

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    X_train = mnist.train.images[:num,:]
    Y_train = mnist.train.labels[:num,:]
    print('')
    return X_train, Y_train

def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test

def display_digit(num):
    print(Y_train[num])
    label = Y_train[num].argmax(axis=0)
    image = X_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# --------------- Some Helper Functions 

def random_mini_batches(X, Y, mini_batch_size = 64): 
	"""
	Creates a list of random minibatches from (X, Y)

	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- labels vector
	mini_batch_size -- size of the mini-batches, integer

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""
	m = X.shape[0]
	mini_batches = []

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation, :]
	shuffled_Y = Y[permutation, :]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[mini_batch_size*k:mini_batch_size*(k+1), :]
		mini_batch_Y = shuffled_Y[mini_batch_size*k:mini_batch_size*(k+1), :]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[mini_batch_size*num_complete_minibatches:m, :]
		mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_minibatches:m, :]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
    
	return mini_batches

# - Create Model
# - The model is as follows
# - CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

# - Create placeholders
def create_placeholder(n_H0, n_W0, n_C0, n_y):
	"""
	Create tensorflow placeholders

	Arguments:
	n_H0 -- scallar, height of an input image
	n_W0 -- scalar, width of an input image
	n_C0 -- scalar, number of channels of the input (RGB)
	n_y -- scalar, number of classes

	Returns:
	X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0]
	Y -- placeholder fot the input labels, of shape [None, n_y]
	"""
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name="X")
	Y = tf.placeholder(tf.float32, [None, n_y], name="Y")
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	return X, Y, keep_prob

def initialize_parameters():
	"""
	Initialise weight parameters to build a neural network
	The shapes are:
		W1 : [4, 4, 3, 8]
		W2 : [2, 2, 8, 16]
	Returns:
	parameters -- a dictionary of tensors containing W1, W2
	"""
	W1 = tf.get_variable("W1", [5, 5, 1, 32], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
	parameters = {"W1": W1, "W2": W2}
	return parameters

def forward_prop(X, parameters, keep_prob):
	"""
	Implements the forward prop:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	
	Arguments: 
	X -- input dataset place holder of shape [input size, number of training data]
	parameters -  W1, W2
	"""
	W1 = parameters['W1']
	W2 = parameters['W2']
	# Conv2d: stride of 1, padding SAME
	Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
	# RELU
	A1 = tf.nn.relu(Z1)
	# MAXPOOL
	P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	# CONV2D
	Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
	# RELU
	A2 = tf.nn.relu(Z2)
	# MAXPOOL
	P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# FLATTEN
	P2 = tf.contrib.layers.flatten(P2)
	# DENSE
	D = tf.contrib.layers.fully_connected(P2, num_outputs = 1024, activation_fn=None)
	# RELU
	DA = tf.nn.relu(D)
	# Optional Drop out regularisation
	dropout = tf.nn.dropout(DA, keep_prob=keep_prob)
	# FULLY-CONNECTED without non-linear activation function (not call softmax).
	Z3 = tf.contrib.layers.fully_connected(dropout, num_outputs = 10, activation_fn=None)
	return Z3

def compute_cost(Z3, Y):
	"""
	Computes the cost

	Arguments:
	Z3 -- output of forward prop of shape [10, num of training]
	Y -- labels

	Returns:
	cost 
	"""
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
	return cost

def model(X_train, Y_train, X_test, Y_test, 
	prob = 0.3, learning_rate = 0.001, minibatch_size = 512, num_epochs = 50, 
	print_cost = True):
	"""
	Implements a three-layer ConvNet in Tensorflow:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

	Arguments:
	X_train -- training set, of shape (None, 64, 64, 3)
	Y_train -- test set, of shape (None, n_y = 6)
	X_test -- training set, of shape (None, 64, 64, 3)
	Y_test -- test set, of shape (None, n_y = 6)
	learning_rate -- learning rate of the optimization
	num_epochs -- number of epochs of the optimization loop
	minibatch_size -- size of a minibatch
	print_cost -- True to print the cost every 100 epochs

	Returns:
	train_accuracy -- real number, accuracy on the train set (X_train)
	test_accuracy -- real number, testing accuracy on the test set (X_test)
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""
	(m, n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []
	# Create placeholder
	X, Y, keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)
	# Initialize parameters
	parameters = initialize_parameters()
	# Forward prop
	Z3 = forward_prop(X, parameters, keep_prob)
	# Compute cost
	cost = compute_cost(Z3, Y)
	# Run backprop
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Saver
	saver = tf.train.Saver()

	# Init global
	init = tf.global_variables_initializer()

	#Start the session
	with tf.Session() as sess:

		sess.run(init)
		for epoch in range(num_epochs):
			epoch_cost = 0.
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:

				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch	
				# Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: prob})
				epoch_cost += minibatch_cost / num_minibatches
                
			# Print the cost every epoch
			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(epoch_cost)

		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		#plt.show()

		# Save the parameters
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the correct predictions
		predict_op = tf.argmax(Z3, 1)
		tf.add_to_collection('predict_op', predict_op)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print(accuracy)
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		# Save the model to a preset (change later)
		saver.save(sess, str(cwd+'/cvn-mnist-model/cvn-mnist'))
		return train_accuracy, test_accuracy, parameters

# ------------- Predict ---------------------
def load_test_set(dir_path = '/Kaggle/test.csv'):
	file_path = cwd + dir_path  # Folder contain the Kaggle dataset
	with open(file_path) as f:
		# - Read line
		content = f.readlines()

		# - Prepare X and Y ndarray
		m = len(content)-1
		X = np.zeros([m, 784])

		for i in range(m):
			img_vals = content[i+1].split(',')	# Skip the first LINE which is the label
			X[i] = [int(s) for s in img_vals[:]] 

		print("Test set loaded\nTest set has shape:" + str(X.shape))
		return X

# - Predict result:
# - Arguments: input X of shape [None, 28, 28, 1]
def predict(X_test):
	# - Restore the model and run prediction
	with tf.Graph().as_default() as g:
		with tf.Session() as sess:
			new_saver = tf.train.import_meta_graph(str(cwd+'/cvn-mnist-model/cvn-mnist.meta'))
			new_saver.restore(sess, tf.train.latest_checkpoint(str(cwd+'/cvn-mnist-model/')))
			# - Recover the ops and variables
			predict_op = tf.get_collection("predict_op")[0]
			X = g.get_tensor_by_name("X:0")
			keep_prob = g.get_tensor_by_name("keep_prob:0")

			# - Run the model
			Y = sess.run(predict_op, feed_dict={X: X_test, keep_prob: 1})
			print(Y[0:10])
			return Y



# - Test one random image
def test_random_image():
	# - Test one random image
	num = randint(0, 1000)
	img = mnist.test.images[num]
	image = img.reshape([1,28,28,1])

	# - Show the image
	image_to_show = mnist.test.images[num].reshape([28,28])
	plt.imshow(image_to_show, cmap=plt.get_cmap('gray_r'))
	plt.show()

	# - Load the model and run prediction
	with tf.Graph().as_default() as g:
		with tf.Session() as sess:
			new_saver = tf.train.import_meta_graph(str(cwd+'/cvn-mnist-model/cvn-mnist.meta'))
			new_saver.restore(sess, tf.train.latest_checkpoint(str(cwd+'/cvn-mnist-model/')))
			predict_op = tf.get_collection("predict_op")[0]
			X = g.get_tensor_by_name("X:0")
			keep_prob = g.get_tensor_by_name("keep_prob:0")
			print(sess.run(predict_op, feed_dict={X: image, keep_prob: 1}))


if __name__ == "__main__":
	X_train, Y_train, X_test, Y_test = load_training_set()
	print(str(Y_train))
	quit()
	# - Prelim
	# - Reshape data to shapes for convnet: X[m, 28, 28, 1] and Y[m, 10]
	num_train_data = X_train.shape[0]
	num_test_data = X_test.shape[0]
  	X_train = X_train.reshape([num_train_data,28,28,1])
	Y_train = Y_train.reshape([num_train_data,10])
	X_test = X_test.reshape([num_test_data,28,28,1])
	Y_test = Y_test.reshape([num_test_data,10])
	# - Train model
	# - Default hyper params for the models are as follows:
	# - prob = 0.4, learning_rate = 0.0009, minibatch_size = 512, num_epochs = 20
	_, _, parameters = model(X_train, Y_train, X_test, Y_test) # The model is saved after this step

	"""
	X = load_test_set()
	m = X.shape[0]
	X = X.reshape([m, 28, 28, 1])
	Y = predict(X)
	with open(cwd+"/Kaggle/prediction.txt", 'w') as f:
		f.write("ImageId,Label\n")
		for i in range(m):
			string = str(i+1) + "," + str(Y[i]) + '\n'
			f.write(string)
		f.close()
	"""

	