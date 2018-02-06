# ML_corner
My ML corner -- various experiments using: e.g. tensorflow

Experiments are sorted in chronological (to remind myself) from start - current, as follows:
1. [HelloWorld -- Iris](https://github.com/affNVu/ML_corner/tree/master/HelloWorld_Iris), this was done following a tutorial from [Machine learning mastery](https://machinelearningmastery.com/hello-world-of-applied-machine-learning/). Algorithms used including LogisticRegression, KNeighborsClassifier, etc.
2. [MNIST](https://github.com/affNVu/ML_corner/tree/master/MNIST), using LogisticRegression with softmax; a simple few hidden layers neural network; a convolutional neural network. The codes used here are adapated from Andrew Ng's Deep Learning course from coursera. The convnet model achieves an accuracy of 98.842% on Kaggle.
3. [Neural Style Transfer exp](https://github.com/affNVu/ML_corner/tree/master/NeuralStyleTransfer). Having a good fun playing with Neural Style Transfer using the VGG16 network. The materials for this exp is taken from the course: Deep Learning by Andrew Ng on Coursera. 
Parameters used to generate the images can be found in the log file. 

The following is an example of 2000 iterations, with the style layers - weights: <br />
/* <br />
    ('conv1_1', 0.5), <br />
		('conv2_1', 0.3), <br />
		('conv3_1', 0.15), <br />
		('conv4_1', 0.1), <br />
		('conv5_1', 0.05) <br />
*/  <br />

![alt text](https://i.imgur.com/wvcAVPC.png "Style image") <a href="url"><img src="https://i.imgur.com/QW9pQOk.jpg?1 " height="400" width="300" ></a>


