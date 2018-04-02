# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:37:50 2018

@author: helga
"""

import numpy as np

#import random
#import matplotlib.pyplot as plt

# with help from my friends on stack overflow -- ans 2
# https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy



def gradientDescentLasso(x, y, theta, learning_rate, n, numIterations, weight_decay, tol = 10**(-4)):
    #print('gradient decent Lasso!')
    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    xTrans = x.transpose()
    cost = 9999 # due to minimization
    previous_cost = 0 # due to minimization
    i = 0
    print('\n=============== gradientDecentLasso =========================')
    while(i < numIterations and abs(cost - previous_cost) > tol):
        guess = np.dot(x, theta)
        loss = guess - y
        previous_cost = cost
        # avg cost per example (the 2 in 2*n doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * n)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / n
        # update
        theta = Gamma(theta - learning_rate * gradient)
        #print('gradientDecentLasso i = {}'.format(i))
        i += 1
    #print('cost:', cost)
    #print('previous_cost:', previous_cost)
    print('gradientDecentLasso i = {}'.format(i))
    return theta


def get_gradient(x, y, theta, n, previous_cost = 9999.9, tol = 10**(-4)):
    xTrans = x.transpose()   
    guess = np.dot(x, theta)
    loss = guess - y
    cost = np.sum(loss ** 2) / (2 * n)
    gradient = np.dot(xTrans, loss) / n
    #print('**cost:', cost)
    if abs(cost - previous_cost) <tol:
        return(True, gradient, cost)
    else:
        return(False, gradient, cost)


