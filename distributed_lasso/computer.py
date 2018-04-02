# -*- coding: utf-8 -*-
from optimization_algorithms import  get_gradient, gradientDescentLasso

class Computer():
    def __init__(self, m):
        # m is the number of data points
        self.cost = 999999999 # due to minimzation
        self.m = m
    
    def get_gradients(self, X, y, theta):
        # X is the predictor variables, y is the response and theta are the weights
        # we check if we have run the method before by checking if
        # the cost has changed, if so we can pass in the cost to calculate
        # the cost difference between the runs, this helps us find out if we have converged
        if self.cost == 999999999:
            is_converged, gradient, cost = get_gradient(X, y, theta, self.m)
        else:
            is_converged, gradient, cost = get_gradient(X, y, theta, self.m, previous_cost = self.cost)
        self.cost = cost
        return(is_converged, gradient)
    
    def lasso_gradiants(self, X, y, theta, learning_rate, num_rounds, weight_decay):
        return gradientDescentLasso(X, y, theta,
                             learning_rate, self.m, num_rounds, weight_decay)
    def set_cost(self, cost):
        self.cost = cost
    
    def set_m(self, m):
        self.m = m