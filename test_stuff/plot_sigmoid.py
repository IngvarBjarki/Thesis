# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:25:44 2018

@author: helga
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.linspace(-6,6,1000)
y = [sigmoid(x) for x in X]

sns.set_style('darkgrid')
pal = sns.hls_palette(8, l=.3, s=.8)
sns.set_palette(pal)
plt.plot(X,y)
plt.savefig('sigmoid.eps', format = 'eps')
plt.show()
