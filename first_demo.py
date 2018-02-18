# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:58:52 2018

@author: Ingvar Bjarki Einarsson
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

import numpy as np
import matplotlib.pyplot as plt

from optimization_algorithms import gradientDescentLasso




def computer_1(X, y, theta, weight_decay):
    learning_rate = 0.0005
    m = len(y)
    numIterations = 10000
    print(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations))
    return(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations))
    
    # X is the data attributes and y are the targets
    #clf = LassoCV()
    #model = SelectFromModel(clf, threshold = 0.025).fit(X, y)
 
    #n_features = model.transform(X)
    
    #print(n_features)
    #print(n_features.shape[1])
    #TODO: finna ut hvernig eg fae binnary vector af rettum features
    
    

    #test = SelectKBest(score_func=chi2, k=4)
    #fit = test.fit(X, y)
    #features = fit.transform(X)
    # summarize selected features
    #print(features[0:5,:])
    



def computer_2(X, y):
    # X is the data attributes and y are the targets
    clf = LassoCV()
    model = SelectFromModel(clf, threshold = 0.25).fit(X, y)
    n_features = model.transform(X).shape[1]
    
        
def validation():
    pass



#=============================== Data processing ===============================
    
is_model = 'iris' #'digits'

if is_model == 'digits':    
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    X = [] 
    y = [] 
    # now we only want to classify 1 and 2
    for image_and_label in images_and_labels:
        image, label = image_and_label
        if label == 1 or label == 2:
            
            # put the image on the correct from for the clf
            # that is we turn the matrix to list
            attributes_flat = []
            for attribute in image:
                # add bais term in the front
                #attribute = attribute.insert(0, 1)
                attributes_flat.extend(attribute)
                
            X.append(attributes_flat)
            y.append(label)
            
elif is_model == 'iris':
    iris = datasets.load_iris()
    X = []
    for attribute in iris.data:
        # we add bias term in the front
        attribute = np.insert(attribute, 0, 1)
        X.append(attribute)
    X = np.asanyarray(X)
    y = iris.target    

    
    
# now we want to split the data set into 2 different data set
# to simmulate different users   
    

# We check if our data set has even or odd numbers, and decide how to split the data
    
n = len(y)
if n % 2 == 0:
    n_half = int(n / 2)
    # we split the data for the two computers
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
    
    computer_1(X_train1, y_train1)
    computer_2(X_train2, y_train2)
else:
    # if odd, we make the first dataset have 1 more record than the second
    n_half = int((n + 1) / 2)
    # we split to train and test before we do feature selection
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
    
    computer_1(X_train1, y_train1)
    #computer_2(X_train2, y_train2)


################# The heart of the program ############################################

#TODO: for lykkja sem kallar oft a follin og sameinar featurinn haegt og rolega..
theta = [0 for i in range(len(X[0]))]
num_rounds = 10
weight_decay = 2 # for the lasso regression
for i in range(num_rounds):

    #computer_1_gradient = computer_1(X_train1, y_train1, theta, weight_decay )
    #computer_2_gradient = computer_2(X_train2, y_train2, theta, weight_decay )
    #global_gradient = [x + y for x, y in zip(computer_1_gradient, computer_2_gradient)]
    #theta = global_gradient
    pass


# check what features to use







