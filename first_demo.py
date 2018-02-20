# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:58:52 2018

@author: Ingvar Bjarki Einarsson
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


import numpy as np
import matplotlib.pyplot as plt
import time
#from optimization_algorithms import gradientDescentLasso, gradientDescent

is_Linux = True
if is_Linux:
    from optimization_algorithms_c import gradientDescent
else:
    from optimization_algorithms import  gradientDescent

def computer_1(X, y, theta, weight_decay):
     # X is the data attributes and y are the targets
    #print('computer 1 is running..')
    learning_rate = 0.0005
    m = len(y)
    numIterations = 10000
    #print(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations, weight_decay))
    #return(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations, weight_decay))
    return(gradientDescent(X, y, theta, learning_rate, m, numIterations))
    

def computer_2(X, y, theta, weight_deca):
    # X is the data attributes and y are the targets
    #print('computer 2 is running..')
    learning_rate = 0.0005
    m = len(y)
    numIterations = 10000
    #print(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations, weight_decay))
    #return(gradientDescentLasso(X, y, theta, learning_rate, m, numIterations, weight_decay))
    return(gradientDescent(X, y, theta, learning_rate, m, numIterations))
    
        
def evaluate(X_train, y_train, X_test, y_test, neural_net, knn):
    # neural_net is a boolen value, if user wants to get the results from a nural network
    # knn is a boolen value, if users want to get the reulsts from a nural network
    # if both values are false an information about the function will be stated
    
    if neural_net:
        mlp = MLPClassifier(hidden_layer_sizes=(30,30))
        mlp.fit(X_train, y_train)
        error_rate =1 -  mlp.score(X_test, y_test)
        return(error_rate)        
        
    if knn:
        error_rates = []
        for i in range(0,100,10):
            neigh = KNeighborsClassifier(n_neighbors=10)
            neigh.fit(X_train, y_train)
            error_rates.append(1-neigh.score(X_test, y_test))
        return(min(error_rates))
        
    if not neural_net and not knn:
        return("at least one argument must be true, to get neural_net or/and knn predictions")
    pass


start_time = time.time()
#=============================== Data processing ===========================================================
#===========================================================================================================
is_model = 'digits' #'iris' 
if is_model == 'digits':    
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X_without_bias = digits.images.reshape((n_samples, -1))
    # we add bias term in front -- done for the gradient decent
    _records, _attributes = np.shape(X_without_bias)
    X = np.ones((_records, _attributes + 1))
    X[:,1:] = X_without_bias
    y = digits.target
    
elif is_model == 'iris':
    iris = datasets.load_iris()
    X = []
    for attribute in iris.data:
        # we add bias term in the front -- done fo the gradient decent
        attribute = np.insert(attribute, 0, 1)
        X.append(attribute)
    X = np.asanyarray(X)
    y = iris.target    



      
########### now we want to split the data set into 2 different data set to simmulate ##########
########### 2 different users - but use different amount of data each time           ##########
    

n = len(y)
if n % 2 == 0:
    n_half = int(n / 2)
    # we split the data for the two computers
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
    

else:
    # if odd, we make the first data set have 1 more record than the second
    n_half = int((n + 1) / 2)
    # we split to train and test before we do feature selection
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
    


################# The heart of the program ############################################



num_splits = 1    
total_amount_of_data = [int(len(y)/num_splits) for i in range(num_splits)]
total_amount_of_data_intervals = np.cumsum(total_amount_of_data)
score_feature_selection = []
score_without_feature_selection = []
# We check if our data set has even or odd numbers, and decide how to split the data

for n in total_amount_of_data_intervals:  
    theta = [0.0 for i in range(len(X[0]))]
    num_rounds = 100
    weight_decay = 0.001 # for the lasso regression - they tried values from 0.0001-0.1
    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    print('Iteration n:',n)
    t2 = time.time()
    for i in range(num_rounds):
        #laga nparray
        computer_1_gradient = computer_1(X_train1[:n], y_train1[:n], np.array(theta), weight_decay )
        computer_2_gradient = computer_2(X_train2[:n], y_train2[:n], np.array(theta), weight_decay )
        global_gradient = [Gamma(0.5 * (x + y)) for x, y in zip(computer_1_gradient, computer_2_gradient)]
        theta = global_gradient
        #print('theta', theta)
    print('gradient time',time.time() - t2)
    
    
    # check what features to use
    threshold = 0.0001
    features = [1 if abs(feature) > threshold else 0 for feature in theta ]
    
    # now we remove the feature from the data set.. before we train the ML model
    features_to_remove = [i for i, feature in enumerate(features) if feature == 0]
    X_train1_with_feature_selection = X_train1
    X_test1_with_feature_selection = X_test1
    num_removed = 1
    print('features to remove', features_to_remove)
    for i, feature_to_remove in enumerate(features_to_remove):
        if i > 0:
            # sincce now the matrix is not as bigg as in the first run
            feature_to_remove -= num_removed
            num_removed += 1
        X_train1_with_feature_selection = np.delete(X_train1_with_feature_selection, feature_to_remove, axis = 1)
        X_test1_with_feature_selection = np.delete(X_test1_with_feature_selection, feature_to_remove, axis = 1)
        #print('removed feature', feature_to_remove)
        
    
    
    ############################ Evaluate the result by running classifiers from computer 1 ###########################
    ############################ on the data with and without the feature selection         ###########################
    
    score_feature_selection.append(evaluate(X_train1_with_feature_selection, y_train1, X_test1_with_feature_selection, y_test1, True, False))
    score_without_feature_selection.append(evaluate(X_train1, y_train1, X_test1, y_test1, True, False))
    
    #print(evaluate(X_train1_with_feature_selection, y_train1, X_test1_with_feature_selection, y_test1, False, True))
    #print(evaluate(X_train1, y_train1, X_test1, y_test1, False, True))



############################# Plot the results ####################################
# probably need to do 3 lines per plot - db lasso, just lassso, without feature selection
#line_up, = plt.plot(total_amount_of_data_intervals, score_feature_selection, '--o', color = 'red', alpha = 0.6, label = 'With feature selection')
#line_down, = plt.plot(total_amount_of_data_intervals, score_without_feature_selection, '--o', color = 'blue', alpha = 0.6, label = 'Without feature selection')

if not is_Linux:
    pass
    #plt.legend(handles=[line_up, line_down])
    #plt.xlabel('N')
    #plt.ylabel('Error rate')
    #plt.savefig('result.png')
    #plt.show()


print('time: ', time.time() - start_time)



