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

def gradientDescent(x, y, theta, learning_rate, m, numIterations):
    xTrans = x.transpose()
    
    for i in range(0, numIterations):
        guess = np.dot(x, theta)
        loss = guess - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        
        # update
        theta = theta - learning_rate * gradient
    return theta



def gradientDescentLasso(x, y, theta, learning_rate, m, numIterations, weight_decay):
    
    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    xTrans = x.transpose()
    for i in range(0, numIterations):
        guess = np.dot(x, theta)
        loss = guess - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        #cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        
        # update
        theta = Gamma(theta - learning_rate * gradient)
    return theta



def gradientDescent_converge(x, y, theta, learning_rate, m, numIterations):
    xTrans = x.transpose()
    cost = 9999 # due to minimization
    i = 0
    while(i < numIterations and cost > 0.025):
        guess = np.dot(x, theta)
        loss = guess - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        
        # update
        theta = theta - learning_rate * gradient
        i += 1
        #print('cost:',cost)
    return theta


def get_gradient(x, y, theta, m):
    xTrans = x.transpose()   
    guess = np.dot(x, theta)
    loss = guess - y
    gradient = np.dot(xTrans, loss) / m
    return gradient



def gradientDescentLasso(x, y, theta, learning_rate, m, numIterations, weight_decay):
    
    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    xTrans = x.transpose()
    for i in range(0, numIterations):
        guess = np.dot(x, theta)
        loss = guess - y
        
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
            
        # update
        theta = Gamma(theta - learning_rate * gradient)
    return theta


'''
def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

plt.plot(y, 'o')
plt.plot(range(100), [point*theta[1] + theta[0] for point in range(100)])
plt.show()


gradientDescentLasso(x, y, theta, alpha, m, numIterations)
print(theta)
plt.plot(y, 'o')
plt.plot(range(100), [point*theta[1] + theta[0] for point in range(100)])
plt.show()
'''
