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

if __name__ == '__main__':
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    '''
     # load the data
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    
    # now we only want to do binary classification of two numbers
    # so we take only number  -- 9 and 4 are probably most similar
    
    index_of_zeros =  np.flatnonzero( y == num1 ) #returns the indexes
    index_of_tows = np.flatnonzero( y == num2 )
     
    # merge the two together and  sort them
    new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
    new_indexes = np.sort(new_indexes)
    y = y[new_indexes]
    X = X[new_indexes]
    # since we are classifying with the sign - we translate the y vector  to -1 to 1
    '''
    
    num1 = 4
    num2 = 9
    y = []
    X = []
    with open("../data/mnist/mnist_train.csv") as l:
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
    num_splits = 2    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] #not lin space i numpy..
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    print('le go')
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = True)
    results = defaultdict(lambda: defaultdict(list))
    for i in range(3):
        # shuffle the data for randomness...
        X, y = shuffle(X, y)
        for n in total_amount_of_data_in_interval:
            
            param_gird = {'C':np.linspace(0,6,241)[1:2]}
            log_regress = LogisticRegression()
            clf = GridSearchCV(estimator = log_regress, param_grid = param_gird, cv = 5)
            clf.fit(X[:n], y[:n])
            best_weight_decay = clf.best_estimator_.C
            results[n]['best_weight_decays'].append(best_weight_decay)
            results[n]['score'].append(clf.best_score_)

    
    
    
    for key in results:
        weight_decay = results[key]['best_weight_decays']
        score = np.mean(results[key]['score'])
        print('The best weight decay for n = {} is: {} with value of {}'.format(key, weight_decay, score))
    '''
    all_weight_decays = []
    for key, value in weight_decays.items():
        print('The best weight decay for n = {} is: {} with value of {}'.format(key, value['best'], value['score']))
        all_weight_decays.append(value['best'])
    
    most_comon_weight_decays = Counter(all_weight_decays)
    value, times = most_comon_weight_decays.most_common(1)[0]
    print('The most comon or one of the most common weight decay is {} and happens {} out of {} times'.format(value, times, num_splits))
        
    '''