#Artificial Neural Network
#Theano - CPU and GPU - numerical processing library
#GPU is very important
#tensorflow also CPU or GPU

##############################DATA PREPROCESSING##############################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# before moving on... do we have any categorical variables? yes.
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])

#this creates dummy variables, ie making the countries all their own variable
#instead of 0 1 2 in country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #dont fall in dummy variable trap - gets rid of one country column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##############################MAKING THE ANN##############################

# Importing the Keras librarries and packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
# calling the model/nerual network "classifier"
classifier = Sequential()

# Adding the input layer and the first hidden layer
# by adding the hidden layer it automatically gives us how many inputs we have
# this is just making the hidden layer
# units is dimensionality of output space
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="glorot_uniform"))

# Add second hidden layer, don't need to say the inputs anymore
classifier.add(Dense(activation="relu", units=6, kernel_initializer="glorot_uniform"))

# Adding output layer
# if you had more than one dependent variable, like if you onehotencoded it
# you would have units as t - 1, where t is number of variables
# and you would ave the activation function as softmax - softmax is just
# sigmoid for many variables
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))

# Compiling the ANN, which means applying stochastic gradient descent to it
# using logarithmic loss as loss function, if you have a binary output then 
# algorim is called binary_crossentropy; categorical is categorical_crossentropy
# metrics is expecting a list of metrics
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)




##############################MAKE PREDICTIONS AND EVALUATING MODEL##############################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #id y_pred is larger than 0.5 it returns true, if not it returns false

'''
new prediction below
double array ensures you're inputting a horizontal line from a 2D array
prediction has to be thje same scale, so we use the same sc object that was fitted to the training set
this makes it the same scale.
'''
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# this worked, if you do 1504+213 / 2000 you get 86%ish accuracy, which is what you
# get in the ANN as well, so we did well without even tuning or anything

##############################EVALUATING, IMPROVING, TUNING THE ANN##############################

# Evaluating the ANN
# K-Fold Cross Validation belongs to sci-kit learn

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score #kfold cross validation function

# function to build classifier
def build_classifier():
    