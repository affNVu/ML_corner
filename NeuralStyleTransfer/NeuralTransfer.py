import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf


def compute_content_cost(a_C, a_G):
	"""
	Computes the content cost

	Arguments:
	a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
	a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

	Returns: 
	J_content -- scalar that you compute using equation 1 above.
	"""
	# Retrieve dimensions from a_G 
	m, n_H, n_W, n_C = a_G.get_shape().as_list()

	# Reshape a_C and a_G
	a_C_unrolled = tf.reshape(a_C, shape=[n_H*n_W, n_C])
	a_G_unrolled = tf.reshape(a_G, shape=[n_H*n_W, n_C])
	# compute the cost with tensorflow
	normalised_factor = 1./(4*n_H*n_W*n_C)
	J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) * normalised_factor
	return J_content

def gram_matrix(A):
	"""
	Argument:
	A -- matrix of shape (n_C, n_H*n_W)

	Returns:
	GA -- Gram matrix of A, of shape (n_C, n_C)
	"""
	GA = tf.matmul(A, tf.transpose(A))
	return GA


def compute_layer_style_cost(a_S, a_G):
	"""
	Arguments:
	a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
	a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

	Returns: 
	J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
	"""

	# Retrieve dimension of the input image
	m, n_H, n_W, n_C = a_G.get_shape().as_list()

	# Reshape the images to have them of shape (n_C, n_H*n_W) by unroll then reshape
	a_S = tf.transpose(tf.reshape(a_S,(n_H*n_W,n_C)))
	a_G = tf.transpose(tf.reshape(a_G,(n_H*n_W,n_C)))

	# Computing gram_matrices for both images S and G
	GS = gram_matrix(a_S)
	GG = gram_matrix(a_G)

	# Compute
	J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) * (1./(4*n_C*n_C*n_H*n_H*n_W*n_W))

	return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
	"""
	Computes the overall style cost from several chosen layers

	Arguments:
	model -- our tensorflow model
	STYLE_LAYERS -- A python list containing:
	                    - the names of the layers we would like to extract style from
	                    - a coefficient for each of them

	Returns: 
	J_style -- tensor representing a scalar value, style cost defined above by equation (2)
	"""

	# initialize the overall style cost
	J_style = 0.

	for layer_name, coeff in STYLE_LAYERS:

	    # Select the output tensor of the currently selected layer
	    out = model[layer_name]

	    # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
	    a_S = sess.run(out)

	    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
	    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
	    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
	    a_G = out
	    
	    # Compute style_cost for the current layer
	    J_style_layer = compute_layer_style_cost(a_S, a_G)

	    # Add coeff * J_style_layer of this layer to overall style cost
	    J_style += coeff * J_style_layer

	return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 100):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """ 
    J = alpha*J_content + beta*J_style
    return J

def model_nn(sess, input_image, num_iterations = 3000):
    
	# Initialize global variables (you need to run the session on the initializer)
	sess.run(tf.global_variables_initializer())

	# Run the noisy input image (initial generated image) through the model. Use assign().
	sess.run(model['input'].assign(input_image))

	for i in range(num_iterations):
		# Run the session on the train_step to minimize the total cost
		sess.run(train_step)

		# Compute the generated image by running the session on the current model['input']
		generated_image = sess.run(model['input'])

		# Print every 20 iteration.
		if i%20 == 0:
			Jt, Jc, Js = sess.run([J, J_content, J_style])
			print("Iteration " + str(i) + " :")
			print("total cost = " + str(Jt))
			print("content cost = " + str(Jc))
			print("style cost = " + str(Js))

			# save current generated image in the "/output" directory
			save_image(cwd + "/output/" + str(i) + ".png", generated_image)

	# save last generated image
	save_image(cwd + '/output/generated_image.jpg', generated_image)

	return generated_image

# ------------------------------------------------------------
if __name__ == "__main__":
	# Get current working directory
	cwd = os.getcwd()
	# Start interactive session
	sess = tf.InteractiveSession()

	# Load, reshape and normalise content image
	content_image = resize_image("images/content.jpg")
	content_image = reshape_and_normalize_image(content_image)


	# Load, reshape and normalize style image
	style_image = resize_image("images/style3.jpg")
	style_image = reshape_and_normalize_image(style_image)

	# Now, we initialize the "generated" image as a noisy image created from the content_image. 
	# By initializing the pixels of the generated image to be mostly noise but still slightly correlated with the content image, 
	# this will help the content of the "generated" image more rapidly match the content of the "content" image. 
	generated_image = generate_noise_image(content_image)

	# Load the VGG16 model.
	model = load_vgg_model(cwd + "/pretrained-model/imagenet-vgg-verydeep-19.mat")


	# To get the program to compute the content cost, we will now assign `a_C` and `a_G` to be the appropriate hidden layer activations.
	# We will use layer `conv4_2` to compute the content cost. The code below does the following:
	# 
	# 1. Assign the content image to be the input to the VGG model.
	# 2. Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
	# 3. Set a_G to be the tensor giving the hidden layer activation for the same layer. 
	# 4. Compute the content cost using a_C and a_G.

	# Assign the content image to be the input of the VGG model.  
	# Check nst_utils.py and see that the input layer of the VGG is named 'input'
	sess.run(model['input'].assign(content_image))

	# Select the output tensor of layer conv5_1
	out = model['conv4_2']

	# Set a_C to be the hidden layer activation from the layer we have selected
	a_C = sess.run(out)
	# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
	# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
	# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
	a_G = out

	# Compute the content cost
	J_content = compute_content_cost(a_C, a_G)


	# **Note**: At this point, a_G is a tensor and hasn't been evaluated. 
	# It will be evaluated and updated at each iteration when we run the Tensorflow graph in model_nn() below.

	# Assign the input of the model to be the "style" image 
	sess.run(model['input'].assign(style_image))

	# Compute the style cost
	# The architect of the VGG is
	"""
	graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
	"""
	STYLE_LAYERS = [
		('conv1_1', 0.5),
		#('conv1_2', 0.1),
		('conv2_1', 0.3),
		#('conv2_2', 0.1),
		('conv3_1', 0.15),
		#('conv3_2', 0.1),
		#('conv3_3', 0.2),
		#('conv3_4', 0.2),
		('conv4_1', 0.1),
		#('conv4_2', 0.1),
		('conv5_1', 0.05)]
	J_style = compute_style_cost(model, STYLE_LAYERS)

	# Calculate cost
	J = total_cost(J_content, J_style)

	# Use Adam optimizer
	optimizer = tf.train.AdamOptimizer(1.5)

	# defi
	train_step = optimizer.minimize(J)

	# Run the model and save the image
	model_nn(sess, generated_image)