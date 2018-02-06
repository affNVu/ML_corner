# Load libraries
import math
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran
import json
# Declare working directory
cwd = os.getcwd()

# Load dataset
from random import randint


# ------------------------------------------- I/O Functions ---------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# - Read raw JSON file 
# - Returns:
# 	X:	The training set
# - Note:
#	X is of shape [band_1, band_2, inc_angle, id, is_iceberg] where band_1, band_2 is a flatten image of shape 75*75 = 5625
#	Thus X is currently of shape [5625, 5265, 1, 1, 1] 
def read_json(dir_path = '/dataset/train.json', train = True):
    file_path = cwd + dir_path  # Folder contain the Kaggle dataset

    # Read the input JSON file
    X = []
    with open(file_path) as data_file:    
        data = json.load(data_file)
        # Parse data
        #for i in range(len(data)):
        for i in range(len(data)):
            obj = data[i]

            # Each obj is a json with: band_1, band_2, inc_angle, id, is_iceberg
            band_1 = [float(s) for s in obj["band_1"]]
            band_2 = [float(s) for s in obj["band_2"]]
            inc_angle = obj["inc_angle"]
            id = obj["id"]
            is_iceberg = obj["is_iceberg"] if train else -1

            # Put into shape 
            x = [band_1, band_2, inc_angle, id, is_iceberg]
            X.append(x)
    return X


# - Preprocess input
# - Arguments:
#	X: a list of length m, each has: [[5625], [5265], 1, 1, 1] ] 
#	ignore_angle: True if we choose to ignore the ignore_angle
# - Returns:
#	X: reshape training set of shape [m, 75,75,2]
#	Y: labels of shape [m, 1]
def preprocess_input(X_unprocessed, ignore_angle = True, train = True):
	# init
	m = len(X_unprocessed)
	X = np.zeros([m, 75, 75, 2])
	id = []
	Y = np.zeros([m, 2])
	# Ignore the angle
	if ignore_angle:
		# Reshape X into [75, 75, 2]
		for i in range(m):
			x_raw = X_unprocessed[i]
			band_1 = np.array(x_raw[0]).reshape([75,75])
			band_2 = np.array(x_raw[1]).reshape([75,75])
			x = np.stack((band_1,band_2),  axis=-1)
			X[i] = x
			id.append(x_raw[3])
			# One hot encoding for label
			if train:
				label = np.zeros([2])
				number = x_raw[4]
				label[number] += 1
				Y[i] = label
			# assert the size
			#print(np.array_equal(band_1, x[:,:,0]) )
			#print(x.shape)
	return X, Y, id


# - Display one random image from the dataset
# - Arguments:
#	X: input data, of shape (m, [75,75,2])
#	the first layer is HH, the second layer is HV, see the documnetation for more details
def display_random_image(X, Y):
	# - Select one random image
	num = randint(0, X.shape[0]-1)
	img = X[10, :]
	isIce = "True" if Y[num] else "False"
	# - Extract the bands
	band_1 = img[:, :, 0]
	band_2 = img[:, :, 1]
	print(band_1)
	print (" ------------- BAND 2 -------------- ")
	print(band_2)
	# - Show the image
	f, axarr = plt.subplots(1, 2, sharey=True)
	axarr[0].imshow(band_1, cmap=plt.get_cmap('gray_r'))
	axarr[0].set_title("band_1")
	axarr[1].imshow(band_2, cmap=plt.get_cmap('gray_r'))
	axarr[1].set_title("band_2")

	plt.suptitle('Is this an iceberg: ' + isIce)
	plt.show()



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


# ----------------------------------------------- HELPER FUNCTIONS --------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

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

# - Given the whole training set, we split the set into train/test sets
"""
	Arguments:
		+X: training set
		+Y: label set
		+frac: fraction of training size i.e. X_size = frac/|X|

	Return:
		+X_train, Y_train: training set
		+X_test, Y_test: test set
"""
def split_training_set(X, Y, frac = 0.95):
	m = X.shape[0]
	train_size = int(m * frac)
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

	return X_train, Y_train, X_test, Y_test

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# --------------------------------------------- CONVNET -------------------------------------------------------
# ---------------CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED ----------
# -------------------------------------------------------------------------------------------------------------


# - Create the place holders for tensorflow
""" - Arguments:
	(n_H0, n_W0, n_C0): shape of the input where
		+ n_H0 is the height of the input image, in this case 75
		+ n_W0 is the width the input image, in this case 75
		+ n_C0 is the number of channels of the input, in this case 2
	n_y: label
	- Returns:
		+ X -- placeholder for the data input of shape [None, n_H0, n_W0, n_C0]
		+ Y -- placeholder for the input labels, of shape [None, n_y]
"""
def create_placeholder(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
	Y = tf.placeholder(tf.float32, [None, n_y], name = "Y")
	keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
	return X, Y, keep_prob

# - Initialise weight matrix for the convnet for the neural network
"""
	- The shapes for the weight matrix are:
		+ W1: [5, 5, 2, 32]
		+ W2: [5, 5, 32, 64]
	- Return:
		+ parameters: a dicitonary of tensors containing W1 - W2
"""
def initialize_parameters():
	W1 = tf.get_variable("W1", [5, 5, 2, 32],
		initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [5, 5, 32, 64],
		initializer = tf.contrib.layers.xavier_initializer())
	parameters = {"W1": W1, "W2": W2}
	return parameters

# - Forward propagation:
"""
	Recal that the model architecture is
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
	- Arguments:
		+ X: input
		+ parameters: dictionary of weight matrixes
		+ keep_prob: drop out probability 
"""
def forward_prop(X, parameters, keep_prob):
	W1 = parameters['W1']
	W2 = parameters['W2']

	#Conv2d: stride of 1, padding SAME
	Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = "SAME")
	#RELU
	A1 = tf.nn.relu(Z1)
	# MAXPOOL: stride 2 thus reduce the size in half
	P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")
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
	Z3 = tf.contrib.layers.fully_connected(dropout, num_outputs = 2, activation_fn=None)
	return Z3

# - Compute the cost
"""
	- Arguments:
		+ Z3: the output of forward prop of shape [m,2]
		+ Y: labels of shape [m,1]
"""
def compute_cost(Z3, Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
	return cost


# - Define the model
"""
	Arguments:
		X_train -- training set, of shape (None, 75, 75, 2)
		Y_train -- test set, of shape (None, n_y = 2)
		X_test -- training set, of shape (None, 75, 75, 2)
		Y_test -- test set, of shape (None, n_y = 2)
		learning_rate -- learning rate of the optimization
		num_epochs -- number of epochs of the optimization loop
		minibatch_size -- size of a minibatch
		print_cost -- True to print the cost every 100 epochs

	Returns:
		train_accuracy -- real number, accuracy on the train set (X_train)
		test_accuracy -- real number, testing accuracy on the test set (X_test)
		parameters -- parameters learnt by the model. They can then be used to predict.
"""
def model(X_train, Y_train, X_test, Y_test, 
	prob = 0.6, learning_rate = 0.0001, minibatch_size = 32, num_epochs = 10, 
	print_cost = True):
	(m, n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []
	# Create placeholder
	X, Y, keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)
	# Initialise parameters
	parameters = initialize_parameters()
	# Forward prop
	Z3 = forward_prop(X, parameters, keep_prob)
	# Compute cost
	cost = compute_cost(Z3, Y)
	# Run backprop
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	# Save the model
	saver = tf.train.Saver()
	# Init global
	init = tf.global_variables_initializer()

	# Start the sessions
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			epoch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
			for minibatch in minibatches:
					# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch
				# Run
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
		plt.show()

		# Save the parameters
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		# Calculate the probability
		output_prob = tf.nn.softmax(Z3)
		tf.add_to_collection('output_prob', output_prob)

		# Calculate the prediction
		predict_op = tf.argmax(Z3, 1)
		tf.add_to_collection('predict_op', predict_op)
		correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1})
		test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1})
		print("Train Accuracy:", train_accuracy)
		print("Test Accuracy:", test_accuracy)
		# Save the model to a preset (change later)
		saver.save(sess, str(cwd+'/cvn-model/cvn-iceberg'))
		return train_accuracy, test_accuracy, parameters


# --------------------------------------------- PREDICT -------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def load_test_set(dir_path = '/dataset/train.json'):
	raw_input = read_json(dir_path, train = False)
	X, _, id = preprocess_input(raw_input, ignore_angle = True, train = False)
	return X, id

# - Predict result:
# - Arguments: input X of shape [None, 28, 28, 1]
def predict(X_test):
	# - Restore the model and run prediction
	with tf.Graph().as_default() as g:
		with tf.Session() as sess:
			new_saver = tf.train.import_meta_graph(str(cwd+'/cvn-model/cvn-iceberg.meta'))
			new_saver.restore(sess, tf.train.latest_checkpoint(str(cwd+'/cvn-model/')))
			# - Recover the ops and variables
			prob_op = tf.get_collection("output_prob")
			#predict_op = tf.get_collection("predict_op")
			X = g.get_tensor_by_name("X:0")
			keep_prob = g.get_tensor_by_name("keep_prob:0")

			# - Run the model
			Y = sess.run(prob_op, feed_dict={X: X_test, keep_prob: 1})
			return Y
# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	is_training = True
	if is_training:
		X_raw = read_json()
		X, Y, _ = preprocess_input(X_raw)
		print("Dataset Loaded:")
		print("- X has shape: " + str(X.shape))
		print("- Y has shape: " + str(Y.shape))
		(X_train, Y_train, X_test, Y_test) = split_training_set(X, Y)
		print("Now splitting dataset ... ")
		print("Done! X_train, Y_train has size: " + str(X_train.shape) + ',' + str(Y_train.shape))
		_, _, parameters = model(X_train, Y_train, X_test, Y_test, 
			prob = 1, learning_rate = 0.0001, minibatch_size = 1, num_epochs = 5) # The model is saved after this step
	else:
		X_test, id = load_test_set()
		m = X_test.shape[0]
		Y = predict(X_test)
		print(Y[0])
		with open(cwd+"/dataset/prediction.txt", 'w') as f:
			f.write("id,is_iceberg\n")
			for i in range(m):
				# Get the ID
				x_id = id[i]
				string = x_id + "," + str(Y[0][i][1]) + '\n'
				f.write(string)
			f.close()


