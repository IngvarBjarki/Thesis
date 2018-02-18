# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:53:24 2018

@author: Ingvar
"""

import matplotlib.pyplot as plt
import numpy as np




def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta







def gradient_decent_error(X, y, weights, b):
    # this function is to calculate errors
    # this function helps to find a stoppping point for the gradient
    least_square_error = 0
    
    for j, attributes in enumerate(X):
        line_value = 0
        for i, A_i in enumerate(attributes):
        
            # let's calculate the line
            #print("A_i error",A_i)
            #print("weight", weight)
            line_value += A_i*weights[i] 

        least_square_error += (y[j] - (line_value + b))**2
    return(least_square_error / float(len(X)))




def gradient_decent(data, y, step_size, max_itrations, b = 0):
    #this gradient decent is for lasso regression
    #data: all the data points
    #x: init gauess
    #step_size: the stepsize of the gradient
    #max_iteration: if we dont achive converges we loop this many times
    
    iteration = 0
    loss = 10 # smoothing higher than the threshold
    gradient = [0 for i in range(len(data[0]))] # init gradient as 0
    
    while(iteration < max_itrations and loss > 0.01):
        for j, attributes in enumerate(data):
            previous_gradient = gradient
            #print(attributes)
            #print(type(attributes))
            for i, A_i in enumerate(attributes):        
            
                gradient[i] = previous_gradient[i] - step_size *(A_i*previous_gradient[i] - y[j])
        
   
        #b = b - step_size * (sum(previous_gradient) - y[j])
        weights = gradient
        loss = gradient_decent_error(data, y, weights, b)
        print('loss', loss)
        print("iteration num", iteration)
        print("gradient", gradient)
        print('b', b)
        
        iteration += 1
        
    
    return(gradient, b)
    
    
    
    
import random

y = []
x = []
multiplier = 3
for i in range(100):
    if i % 4 == 0:
        multiplier += 2
    y.append(multiplier * random.normalvariate(2,2))
    x.append([multiplier])
    
j = [x]

print([i for i in range(len(j[0][0]))])

#print([x])
grad, b = gradient_decent(x, y, 0.01, 200)

line = [val[0]*grad[0] + b for val in x]
    
plt.plot(x, y, 'o')
plt.plot(x,line,color = 'red')
plt.show()



















#################################################

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

b,c = gradient_decent(x, y, 0.01, 200)

print(b,c)










############################## /essi var i hinu############################



def gradient_decent_error(X, y, weights, b):
    # this function is to calculate errors
    # this function helps to find a stoppping point for the gradient
    total_error = 0
    for i, A_i in enumerate(X):
        
        # let's calculate the line
        line_value = [A_i*weight for weight in X]
        line_value_sum = sum(line_value)
        
        total_error += (y[i] - (line_value_sum + b))**2
    return(total_error / float(len(X)))


def gradient_decent(data, x, step_size, max_itrations, m, b = 0):
    #this gradient decent is for lasso regression
    #data: all the data points
    #x: init gauess
    #step_size: the stepsize of the gradient
    #max_iteration: if we dont achive converges we loop this many times
    
    iteration = 0
    loss = 10 # smoothing higher than the threshold
    gradient = [0 for i in len(data[0])] # init gradient as 0
    
    while(iteration < max_itrations or loss > 0.01):
        for attributes in range(len(data)):
            previous_gradient = gradient
            for i, A_i in enumerate(attributes):        
                # calculate the inner function of the gradient function by first
                # multiplying A_i to all the wieghts and then sum them upp to get the predicted val
                # A_i*x -- in the paper
                line_value = [A_i*weight for weight in previous_gradient]
                line_value_sum = sum(line_value)
                gradient[i] = gradient[i] - step_size *(A_i*(line_value_sum - y))
        
        # MIG VANTAR LIKA DRASLID I JOFNU 2 I BLADINU..
        # claculate b =... 
        b = b - step_size * (sum(previous_gradient) - y)
        weights = gradient
        loss = gradient_decent_error(data, weights, b)
        iteration += 1
    
    return(gradient)
        


