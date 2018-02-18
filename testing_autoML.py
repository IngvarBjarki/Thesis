# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:26:38 2018

@author: Ingvar Bjarki Einarsson

"""


# We try out the autoML library -- use Iris dataset


from sklearn import datasets
import autosklearn.classification
import sklearn.model_selection


iris = datasets.load_iris()
X = iris.data[:,:2] # take the first two features
y = iris.target


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, random_state = 1)

print('running autoML...')
autoML = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=1200, per_run_time_limit=300)
autoML.fit(X_train, y_train)
y_hat = autoML.predict(X_test)



print('-------------------------------------------------------------------')
print('Accuracy score', sklearn.metrics.accuracy_score(y_test, y_hat))
print('-------------------------------------------------------------------')
print('show models', autoML.show_models())
print('------------------------------------------------------------------')
print('get_models_with_weights', autoML.get_models_with_weights())
print('------------------------------------------------------------------')
print('get_params', autoML.get_params())
print('==============================================================================')
print('nr2', autosklearn.metrics.accuracy(y_test, y_hat))
print('==============================================================================')
print('show_models',autoML.show_models())
