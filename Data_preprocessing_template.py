# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:01:48 2020

@author: Nooreldean Koteb
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# x = onehotencoder.fit_transform(x).toarray()
#2020 Edition
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Country", OneHotEncoder(categories = 'auto'), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

#Encoding y catagorical data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Splitting dataset between training set and test set
from sklearn.model_selection import train_test_split
#random_state not required
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
#Could also scale countries by removing [:, 3:5]
x_train[:,3:5] = sc_x.fit_transform(x_train[:,3:5])
x_test[:,3:5] = sc_x.transform(x_test[:,3:5])