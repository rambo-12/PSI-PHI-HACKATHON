# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:01:37 2019

@author: ritvik
"""

# Artificial Neural Network

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Climate.csv')
X = dataset.iloc[:, 3:23].values
y = dataset.iloc[:, 23].values
y = np.reshape(y, (3749,1))

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()
y[:,0] = labelencoder_y.fit_transform(y[:,0])
ohe = OneHotEncoder(categorical_features = [0])
y= ohe.fit_transform(y).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.reshape(X_train, (2811,1,20))
X_test = np.reshape(X_test, (938,1,20))

# Neural Network Architecture

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

classifier = Sequential()
classifier.add(LSTM(units = 4, activation = 'tanh', input_shape = (None,20) ))
classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 250, validation_data = (X_test , y_test))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

pred = ohe.inverse_transform(y_pred)
yy = ohe.inverse_transform(y_test)

    
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yy, pred)
print('\n')
print('CONFUSION MATRIX')
print(cm)

from sklearn.metrics import classification_report
cr = classification_report(yy,pred)
print(cr)
