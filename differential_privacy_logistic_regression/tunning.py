# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:55:32 2018

@author: Ingvar
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from collections import Counter



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


 # load the data
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
dimensionality = len(X[0])
# now we only want to do binary classification of two numbers
# so we take only number  -- 9 and 4 are probably most similar
num1 = 4
num2 = 9
index_of_zeros =  np.flatnonzero( y == num1 ) #returns the indexes
index_of_tows = np.flatnonzero( y == num2 )
 
# merge the two together and  sort them
new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
new_indexes = np.sort(new_indexes)
y = y[new_indexes]
X = X[new_indexes]
# since we are classifying with the sign - we translate the y vector  to -1 to 1
y[y == num1] = -1
y[y == num2] = 1

test_size = 0.1
number_of_training_samples = int((1 - test_size) * len(X)) # the train test split function also rounds down

# in each itreation we use different amount of data to see how the model improvese with increased data
num_splits = 15    
total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] #not lin space i numpy..
total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = True)

weight_decays = {}
for n in total_amount_of_data_in_interval:
    
    param_gird = {'C':np.linspace(0,6,241)[1:]}
    log_regress = LogisticRegression()
    clf = GridSearchCV(estimator = log_regress, param_grid = param_gird, cv = 10)
    clf.fit(X, y)
    best_weight_decay = clf.best_estimator_.C
    weight_decays[n] = {'best': best_weight_decay, 'score': clf.best_score_}
    
all_weight_decays = []
for key, value in weight_decays.items():
    print('The best weight decay for n = {} is: {} with value of {}'.format(key, value['best'], value['score']))
    all_weight_decays.append(value['best'])
    
most_comon_weight_decays = Counter(all_weight_decays)
value, times = most_comon_weight_decays.most_common(1)
print('The most comon or one of the most common weight decay is {} and happens {} out of {} times'.format(value, times, num_splits)
    
