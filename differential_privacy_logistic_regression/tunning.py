# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:55:32 2018

@author: Ingvar
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from collections import Counter, defaultdict
import json
import time
import os
if __name__ == '__main__':
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    num1 = 4
    num2 = 9
    y = []
    X = []
    with open("../../data/mnist_train.csv") as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(num) for num in line[1:]]
                y.append(label)
                X.append(features)
    
    
    y = np.asarray(y)
    X = np.asarray(X)
    
    
    y[y == num1] = -1
    y[y == num2] = 1
    
    
    number_of_training_samples = len(y) 
    dimensionality = len(X[0])
    # in each itreation we use different amount of data to see how the model improvese with increased data
    num_splits = 30    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] #not lin space i numpy..
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    print('le go')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = True)
    results = defaultdict(lambda: defaultdict(list))
    for i in range(45):
        # shuffle the data for randomness...
        X, y = shuffle(X, y)
        for n in total_amount_of_data_in_interval:
            
            param_gird = {'C':np.linspace(0,4000,201)[1:]}
            log_regress = LogisticRegression()
            clf = GridSearchCV(estimator = log_regress, param_grid = param_gird, cv = 5, n_jobs = -1)
            clf.fit(X[:n], y[:n])
            best_weight_decay = clf.best_estimator_.C
            results[str(n)]['best_weight_decays'].append(best_weight_decay)
            results[str(n)]['score'].append(clf.best_score_)

    
    
    
    for key in results:
        weight_decay = results[key]['best_weight_decays']
        score = np.mean(results[key]['score'])
        print('The best weight decay for n = {} is: {} with value of {}'.format(key, weight_decay, score))
        
    with open('logistic_results.json', 'w') as f:
        json.dump(results, f)
        
    print('results stored in logistic_results.json')
    
    time.sleep(5000)
    
    #shut down
    os.system('shutdown -s')
    
        
    '''
    all_weight_decays = []
    for key, value in weight_decays.items():
        print('The best weight decay for n = {} is: {} with value of {}'.format(key, value['best'], value['score']))
        all_weight_decays.append(value['best'])
    
    most_comon_weight_decays = Counter(all_weight_decays)
    value, times = most_comon_weight_decays.most_common(1)[0]
    print('The most comon or one of the most common weight decay is {} and happens {} out of {} times'.format(value, times, num_splits))
        
    '''