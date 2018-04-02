# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:07:13 2018

@author: Ingvar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from optimization_algorithms import  gradientDescent, gradientDescent_converge

def computer_1(X, y, theta):
    # X is the data attributes and y are the targets, theta is the gradient
    learning_rate = 0.0005
    m = len(y) 
    numIterations = 10
    return(gradientDescent_converge(X, y, theta, learning_rate, m, numIterations))
    
def computer_2(X, y, theta):
    # X is the data attributes and y are the targets, theta is the gradient
    learning_rate = 0.0005
    m = len(y) 
    numIterations = 10
    return(gradientDescent_converge(X, y, theta, learning_rate, m, numIterations))




digits = datasets.load_digits()
n_samples = len(digits.images)
X_without_bias = digits.images.reshape((n_samples, -1))
y = digits.target
 
# now we only want to do binary classification of two numbers
# so we take only number 0 and 2 ---- 9 and 4 are probably most similar
index_of_zeros =  np.flatnonzero( y == 4 ) #returns the indexes
index_of_tows = np.flatnonzero( y == 9 )
 
# merge the two together and  sort them
new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
new_indexes = np.sort(new_indexes)
y = y[new_indexes]
X_without_bias = X_without_bias[new_indexes]
# since we are classifying with the sign - we translate the y vector  to -1 to 1
y[y == 4] = -1
y[y == 9] = 1
# we add bias term in front -- done for the gradient decent
records, attributes = np.shape(X_without_bias)
X = np.ones((records, attributes + 1))
X[:,1:] = X_without_bias
 
# we split to train and test before we do feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 26)

# split the data differently according to if it is odd or even
if len(X_train) % 2 == 0:
    n = int(len(y) / 2)
    # we split the data for the two computers
    X_train1, y_train1 = X_train[:n], y_train[:n]
    X_train2, y_train2 = X_train[n:], y_train[n:]
    print('test1')
    
else:
    # if odd, we make the first data set have 1 more record than the second
    # split the data for the two computers
    n = int((len(y) + 1) / 2)
    X_train1, y_train1 = X_train[:n], y_train[:n]
    X_train2, y_train2 = X_train[n:], y_train[n:]
    
# in each itreation we use different amount of data to see how the model improvese with increased data
num_splits = 1    
total_amount_of_data = [int(n/num_splits) for i in range(num_splits)] #not lin space i numpy..
total_amount_of_data_intervals = np.cumsum(total_amount_of_data)

# now we perform the distributed lasso regression
num_rounds = 101
weight_decay = 0.001 # for the lasso regression - they tried values from 0.0001-0.1
Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
accuracies = []
accuracies_single = []
for n in total_amount_of_data_intervals:  
    theta = np.array([0.0 for i in range(len(X[0]))])
    print('Iteration n:',n)
        
    for i in range(num_rounds):
        computer_1_gradient = computer_1(X_train1[:n], y_train1[:n], theta )
        computer_2_gradient = computer_2(X_train2[:n], y_train2[:n], theta)
        total_gradients = computer_1_gradient + computer_2_gradient
        
        theta  =  Gamma(theta - 0.0001 *total_gradients)
        
        #if i % 10 == 0:
            #print('**********{}***********'.format(i))
            #total_correct = 0
            #for i in range(len(y_test)):
                #prediction = np.sign(np.dot(theta, X_test[i]))
                #if prediction == y_test[i]:
                    #total_correct += 1

            #print('\ntotal correct: ', total_correct)

            
    # Evaluate the model -- check for error rate
    total_correct = 0
    for i in range(len(y_test)):
        prediction = np.sign(np.dot(theta, X_test[i]))
        #print(prediction, y_test[i])
        if prediction == y_test[i]:
            total_correct += 1
    
    print('\ntotal correct distributed lasso: ', total_correct)


############## If only one computer did the analysis on there own data ############################
    theta = np.array([0.0 for i in range(len(X[0]))])
    for i in range(num_rounds):       
        theta = computer_1(X_train1[:n], y_train1[:n], theta )
        theta  =  Gamma(theta - 0.0001 *theta)
    
    # Evaluate the model -- check for error rate
    total_correct = 0
    for i in range(len(y_test)):
        prediction = np.sign(np.dot(theta, X_test[i]))
        if prediction == y_test[i]:
            total_correct += 1
    
    print('\ntotal correct single computer: ', total_correct)


############ If all data was at a centeralized location ###########################################

    all_predictors = np.concatenate((X_train1[:n], X_train2[:n]))
    all_responses = np.concatenate((y_train1[:n], y_train2[:n]))
    theta = np.array([0.0 for i in range(len(X[0]))])
    for i in range(num_rounds):
        theta = computer_1(all_predictors, all_responses, theta)
        theta  =  Gamma(theta - 0.0001 *theta)

    # Evaluate the model -- check for error rate
    total_correct = 0
    for i in range(len(y_test)):
        prediction = np.sign(np.dot(theta, X_test[i]))
        if prediction == y_test[i]:
            total_correct += 1
    
    print('\ntotal correct - central location: ', total_correct)
