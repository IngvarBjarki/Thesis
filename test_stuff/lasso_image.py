# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:25:44 2018

@author: helga
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


X = np.linspace(-1,1,10)
X = [1, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1]
y = [abs(1 - x ) for x in X]

sns.set_style('darkgrid')
pal = sns.hls_palette(8, l=.3, s=.8)
sns.set_palette(pal)
plt.plot(range(len(y)),y)
#efig('lassin.eps', format = 'eps')
plt.show()
