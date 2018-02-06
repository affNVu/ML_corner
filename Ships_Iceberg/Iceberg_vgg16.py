# Load libraries
import math
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran
import json
# Declare working directory
from nst_utils import *
from Iceberg_convnet import *

cwd = os.getcwd()


"""
	-- IMAGE AUGMENTATION to fit the VGG16 model
	Some image augmentation, however not in the sense (or it is, perhaps) of proper 'augmentation'.
	The following functions just try to fit the input picture which is of size [75,75,2]
	to the input of VGG16 which requires picture of size [224,224,3].

	So, the following functions do:
	1. add zeros padding and insert the input picture at an offset
	2. add an additional channel 

"""

def add_extra_padding_to_picture(X, H, W):
	"""
	Add extra padding to the picture to make it the desire size
	Arguments:
		X -- the input picture of shape [n_h, n_w, n_c]
		H -- desired height
		W -- desired width
	Returns:
		X_ new picture with added paddings
	"""
	m, n_H, n_W, n_C  = X.shape
	X_ = np.zeros([m, H, W, n_C])
	X_[:, :n_H, :n_W, :] = X
	return X_

def add_extra_layer_to_one_picture(X, additional = "0"):
	"""
	Add an additional channel to the input, since the input here has only two layers ie. HH and HV
	Arguments:
		- X: one picture of shape [n_H, n_W, n_C]
		- additional: 
			"0" -- the extra layer is all 0,
			"HH" -- the extra layer is HH,
			"HV" -- the extra layer is HV,
			"mean" -- the extra layer is mean i.e. (HH+HV)/2
	Returns:
		- Augmented X
	"""
	n_H, n_W, _ = X.shape
	HH = X[:, :, 0]
	HV = X[:, :, 1]
	extra_layer = np.zeros([n_H, n_W])
	if additional == "HH":
		extra_layer = HH
	elif additional == "HV":
		extra_layer = HV
	elif additional == "mean":
		extra_layer = (HH+HV)/2
	return np.dstack((X, extra_layer))

def augment_picture_with_layer(X):
	"""
	Augment layer with additional layer
	Arguments:
		X -- input of shape [m, n_H, n_W, n_C]
	Returns:
		Agumented X of shape [m, n_H, n_W, n_C + 1]
	"""
	m, n_H, n_W, n_C = X.shape
	X_aug = np.zeros([m, n_H, n_W, (n_C + 1)])
	for i in range(1):
		x = X[i]
		X_aug[i] = add_extra_layer_to_one_picture(x)
	return X_aug


# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	# First load the test set
	# Remember that the default input of VGG16 is pictures with 3 channels i.e. RGB
	# We might have to augment the picture here
	X_raw = read_json()
	X, Y, _ = preprocess_input(X_raw)
	X = X[0:1,:]
	Y = Y[0:1,:]
	X_test = augment_picture_with_layer(X)
	X_test = add_extra_padding_to_picture(X_test, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
	num_epochs = 2

	# Load the VGG16 model.
	model = load_vgg_model(cwd + "/pretrained-model/imagenet-vgg-verydeep-19.mat")
	# Assign the content image to be the input of the VGG model.  
	# Check nst_utils.py and see that the input layer of the VGG is named 'input'

	# Access the last layer
	last_layer = model['fc8']
	# Define cost and optimisation ops
	Z = tf.contrib.layers.fully_connected(last_layer, num_outputs = 2, activation_fn=None)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(model['input'].assign(X_test))
		for epoch in range(num_epochs):
			sess.run(train_step)
			epoch_cost = sess.run(cost)
			print("Cost: " + str(epoch_cost))
	



