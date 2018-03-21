# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:41:34 2018

@author: helga
"""

import numpy as np
y = []
X = []
with open("../data/mnist/mnist_train.csv") as l:
    for i , line in enumerate(l):
        line = line.split(",")
        y.append(int(line[0]))
        X.append([float(i) for i in line[1:]])
        print(line)
        print("\n")
        print(len(line))
        print("\n")
        if i == 5:
            break
        
y = np.asarray(y)
X = np.asarray(X)