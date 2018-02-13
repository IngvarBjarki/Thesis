# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:58:52 2018

@author: Ingvar Bjarki Einarsson
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def computer_1(X, y):
    # X is the data attributes and y are the targets
    clf = LassoCV()
    model = SelectFromModel(clf, threshold = 0.025).fit(X, y)
 
    n_features = model.transform(X)
    
    print(n_features)
    print(n_features.shape[1])
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

def plot_data():
    pass


da_model = 'iris' #'digits'

if da_model == 'digits':    
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
                attributes_flat.extend(attribute)
                
            X.append(attributes_flat)
            y.append(label)
elif da_model == 'iris':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target    

    
    
# now we want to split the data set into 2 different data set
# to simmulate different users    
n = len(y)
# if else statement for spliting between the computers
if n % 2 == 0:
    n_half = int(n / 2)
    # we split to train and test before we do feature selection
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


#TODO: for lykkja sem kallar oft a follin og sameinar featurinn haegt og rolega..
    








