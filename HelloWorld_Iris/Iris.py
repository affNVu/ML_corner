# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import os
cwd = os.getcwd()

# Load dataset
filePath = cwd + "/Iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(filePath, names=names)

# shape
def data_visualisation(dataset):
	print(dataset.shape)
	print(dataset.head(20))
	print(dataset.describe())
	print(dataset.groupby('class').size())
	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	plt.show()
	# histograms
	dataset.hist()
	plt.show()
	# scatter plot matrix
	scatter_matrix(dataset)
	plt.show()

def create_training_set(dataset):
	array = dataset.values
	X = array[:, 0:4]
	Y = array[:,4]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	return X_train, X_validation, Y_train, Y_validation

def test_algorithms(X_train, X_validation, Y_train, Y_validation, seed, scoring):
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

def kNN(X_train, X_validation, Y_train, Y_validation):
	# Make predictions on validation dataset
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))

X_train, X_validation, Y_train, Y_validation = create_training_set(dataset)
kNN(X_train, X_validation, Y_train, Y_validation)
#test_algorithms(X_train, X_validation, Y_train, Y_validation, seed = 7, scoring = 'accuracy')