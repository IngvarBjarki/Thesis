# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:25:44 2018

@author: helga
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.linspace(-6,6,1000)
y = [sigmoid(x) for x in X]


plt.plot(X,y, color = 'black')
plt.savefig('sigmoid.eps', format = 'eps')
plt.show()
