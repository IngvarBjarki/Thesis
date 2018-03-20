# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:41:41 2018

@author: helga
"""

from mnist import MNIST
path = r"C:\Users\helga\OneDrive\Documents\Thesis\github\Thesis\python-mnist\data"
mndata = MNIST(path)
#path = r"C:\Users\helga\OneDrive\Documents\Thesis\github\Thesis\differential_privacy_logistic_regression\mnist\python-mnist"
#mndata = MNIST(path)
images, labels = mndata.load_training()