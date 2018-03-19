# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:56:08 2018

@author: helga
"""

from numba import jit, autojit
import numba
from numpy import arange
from multiprocessing import Pool
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import numpy as np
# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
g = lambda x: x**2
@autojit(locals=dict(reslen=np.float64))
def sum2d(arr):
    arr, b = arr
    clf = LogisticRegression()
    all_weights = defaultdict(tuple)
    epsilons = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 3)
    for epsilon in epsilons:
        print('je')
    #c = np.array((1.0 for i in range(10)), dtype = np.float64)
    c = np.zeros(16)
    k = []
    for i in range(10):
        k.append(1)
    #k = (1 for i in range(10))
    j = np.asarray(k)
    M, N = arr.shape
    result = 0.0
    bla = defaultdict(list)
    bla['yo'].append(1)
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return (result,2)


if __name__ == '__main__':
    a = arange(9).reshape(3,3)
    args = [(a,4)] * 10
    p = Pool(4)
    res = p.map(sum2d, args)
    p.close()
    p.join()
    print(res)