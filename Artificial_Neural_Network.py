# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:14:18 2020

@author: Nooreldean Koteb
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1 ].values
y = dataset.iloc[:, -1].values


#Encoding geographical categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Country', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

#Fixing dummy variable trap
x = x[:, 1:]

#Encoding gender catagorical data
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

#Splitting dataset between training set and test set
from sklearn.model_selection import train_test_split
#random_state not required
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )


#feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)





#Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#Used to improve ANN by reducing overfitting
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer

#Old keras API
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#2020 Edition
classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

#Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


#Making the predictions and evaluating the model

# #Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#Predicting a single new observation
"""Predicting if the customer with the following info will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc_x.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# #Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#---------------------------------------------------#
# #Evaluating the ANN
# #Evaluation imports
#
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
#
# def build_classifier():
#     #Initializing the ANN
#     classifier = Sequential()
#     #Adding the input layer and the first hidden layer
#     classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
#     # Adding the second hidden layer
#     classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
#     #Adding the output layer
#     classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     #Compiling the ANN
#     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
#   
#     return classifier
#
# classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#
# #K-Fold cross validation
# accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10, n_jobs = -1)
# mean = accuracies.mean()
# variance = accuracies.std()
#---------------------------------------------------#
# #Improving the ANN
# #Dropout regularization to reduce overfitting if needed
# #Randomly disables nodes to avoid too much dependency on one node
# #This should be placed under the added nodes above
#
# from keras.layers import Dropout
# classifier.add(Dropout(p = 0.1))
#---------------------------------------------------#
# #Tuning the ANN 
# #in Artificial_Neural_Network.py
# #Parameter tuning using grid search
# #Making the ANN & Evaluating the ANN
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# #Evaluation imports
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV

# def build_classifier(optimize, nodes):
#     #Initializing the ANN
#     classifier = Sequential()
#     #Adding the input layer and the first hidden layer
#     classifier.add(Dense(nodes, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
#     # Adding the second hidden layer
#     classifier.add(Dense(nodes, kernel_initializer = 'uniform', activation = 'relu'))
#     #Adding the output layer
#     classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     #Compiling the ANN
#     classifier.compile(optimizer = optimize,  loss = 'binary_crossentropy', metrics = ['accuracy'] )
#  
#     return classifier
#
# classifier = KerasClassifier(build_fn = build_classifier)

# #parameters to be tested
# parameters = {'batch_size': [25, 32], 
#               'epochs': [100, 500],
#               'optimize': ['adam', 'rmsprop'],
#               'nodes': [3, 6, 12]}
#
# grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
#
# grid_search = grid_Search.fit(x_train, y_train)
#
# #The best parameters and accuracy
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
#---------------------------------------------------#
