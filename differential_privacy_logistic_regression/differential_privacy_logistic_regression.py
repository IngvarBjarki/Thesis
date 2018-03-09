# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:11:07 2018

@author: Ingvar
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



binnary = True
debugg = False
regularization_constant = 5
# load the data
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target


if binnary:
    
    # now we only want to do binary classification of two numbers
    # so we take only number  -- 9 and 4 are probably most similar
    index_of_zeros =  np.flatnonzero( y == 4 ) #returns the indexes
    index_of_tows = np.flatnonzero( y == 9 )
     
    # merge the two together and  sort them
    new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
    new_indexes = np.sort(new_indexes)
    y = y[new_indexes]
    X = X[new_indexes]
    # since we are classifying with the sign - we translate the y vector  to -1 to 1
    y[y == 4] = -1
    y[y == 9] = 1
    
 
all_accuracys = defaultdict(list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 42)
    
# in each itreation we use different amount of data to see how the model improvese with increased data
num_splits = 15    
total_amount_of_data = [int(len(y_train)/num_splits) for i in range(num_splits)] #not lin space i numpy..
total_amount_of_data_intervals = np.cumsum(total_amount_of_data)

for n in total_amount_of_data_intervals:
    
    
    clf = LogisticRegression(penalty="l2", C=regularization_constant)
    clf.fit(X_train[:n], y_train[:n])
    if debugg:
        print(clf.score(X_test, y_test))
        print(len(X_train))
    weights = clf.coef_[0]
    
    
    scikit_proba = clf.predict_proba(X_test)
    scikit_predict = clf.predict(X_test)
    num_correct_predictions = 0
    for i in range(len(y_test)):
        arg = np.dot(weights, X_test[i])
        prediction_probability = sigmoid(arg)
        if debugg:
            print('my prediction', prediction_probability )
            print('scikit prediction proba', scikit_proba[i])
            print('scikit prediction', scikit_predict[i])
            print('truth', y_test[i])
        
        # check which class to predict
        if prediction_probability > (1 - prediction_probability):
            predicition = 1
        else:
            predicition = -1
        
        truth = y_test[i]
        if predicition == truth:
            num_correct_predictions += 1
    
    
        # add the score
    all_accuracys['Without DP'].append(1 - clf.score(X_test, y_test))
    #accur = num_correct_predictions / len(y_test)
    #print('accur', accur)
        
        
    ############# add differential privacy #########################
    
    #np.random.laplace()
    
    

    sensitivity = 2 / (len(y_train) * regularization_constant)
    epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 3]
    for epsilon in epsilons:
        accur = 0
        for i in range(1000):
            weights_perturb = np.array([weight + np.random.laplace(0, sensitivity / epsilon)  for weight in weights])
            
            num_correct_predictions = 0
            for i in range(len(y_test)):
                arg = np.dot(weights_perturb, X_test[i])
                prediction_probability = sigmoid(arg)
                
                # check which class to predict
                if prediction_probability > (1 - prediction_probability):
                    predicition = 1
                else:
                    predicition = -1
                
                truth = y_test[i]
                if predicition == truth:
                    num_correct_predictions += 1
        
        
        accur += num_correct_predictions / len(y_test)
        #print('accur perturb', accur/500)
        all_accuracys['DP noise ' + str(epsilon)].append(1 - accur)
        
############ Plot the results #################################
for i in all_accuracys:
    print(i)
    plt.plot(total_amount_of_data_intervals, all_accuracys[i], label = i, alpha = 0.75)
plt.legend()
plt.show()
