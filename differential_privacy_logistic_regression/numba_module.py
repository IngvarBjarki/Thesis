import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from multiprocessing import Pool
from numba import jit
from functions import sigmoid



###### This module can be imported into the differential_privacy_logistic_regresion_multi module      ############
###### To get the heart of the program compiled with numba, however, I expirence only small amount    ############
###### of increase in speed, so in light of simplisity a chose to use the main func inside the module ############
@jit
def main_numba(all_args):


    debugg = False

    X, y, total_amount_of_data_in_interval, test_size, dimensionality = all_args

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = True)

    regularization_constant = 0.2
    num_rounds_to_avg = 1000
    
    
    all_accuracies = defaultdict(list)
    avg_noise_for_each_n = defaultdict(list)
    # we use tuple because lambda functions are not pickable- thus dont work with multiprocessing -uses que
    all_weights = defaultdict(tuple)
    for n in total_amount_of_data_in_interval:
        # C is the inverse of the regularization strength so we turn it to get the corr value
        clf = LogisticRegression(penalty="l2", C =  1 / regularization_constant)
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
        all_accuracies['Without DP'].append(1 - clf.score(X_test, y_test))
        all_weights[str(n)] = (weights)
        
            
        ############# add differential privacy #########################
        
        sensitivity = 2 / (len(y_train) *  regularization_constant)
        epsilons = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 3)
        for epsilon in epsilons:
            accur = 0
            total_noise = 0
            for i in range(num_rounds_to_avg):
                noise = []
                for i in range(dimensionality):
                    noise.append(np.random.laplace(0, sensitivity / epsilon))
                #noise = np.array([np.random.laplace(0, sensitivity / epsilon) for i in range(dimensionality)])
                noise = np.asarray(noise)

                weights_perturb = weights + noise
                
                ######## total noise.. ########
                # evaluate the model
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
            
                
                total_noise += sum(abs(noise)) # vantar liklega abs gildinn
                accur += num_correct_predictions / len(y_test)

            # first index has the lowest n and then it increases
            all_accuracies['$\epsilon$ = ' + str(epsilon)].append(1 - accur / num_rounds_to_avg)
            avg_noise_for_each_n['noise eps = ' + str(epsilon)].append(total_noise / num_rounds_to_avg)
            #all_noise_and_weights['weights eps = ' + str(epsilon)].append()
    return (all_accuracies, avg_noise_for_each_n, all_weights)