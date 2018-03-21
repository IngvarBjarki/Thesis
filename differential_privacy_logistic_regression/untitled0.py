# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:08:59 2018

@author: helga
"""
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
sns.set_style('whitegrid')
#sns.set_palette(sns.color_palette("Reds_d", 9))
num = 9

sns.set_palette(sns.color_palette('Set1', num))

y = range(num)
palette = itertools.cycle(sns.color_palette())
for i in range(num):
    plt.plot(range(num), y, label = i)
    y = [i + 1 for i in y]
plt.legend()
plt.show()
