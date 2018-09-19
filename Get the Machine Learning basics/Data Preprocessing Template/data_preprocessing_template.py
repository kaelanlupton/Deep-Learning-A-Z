# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
#each column in the dataset has an index, 0 onwards 
X = dataset.iloc[:, :-1].values #the first array
y = dataset.iloc[:, -1].values

# Splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#already fitted to training set, dont need to fit again

'''

#COPY ALL OF THE ABOVE FOR START OF THINGS

# don't need to find feature scaling to Y because only 0 and 1, but i think if
# it can take a huge range of values then feature scaling will be necessary


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #if axis = 0 we are taking means of columns, if = 1 we are taking means of rows
imputer = imputer.fit(X[:, 1:3]) #dontreallyneed the imputer =
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)

'''
