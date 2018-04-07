# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:36:54 2018

@author: s161294
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style = 'darkgrid')


with open("parameters_to_plot_fixed.json") as f:
    average_results = json.load(f)
    
    
for res in average_results:
    if not res == 'total_amount_of_data_intervals':
        plt.plot(average_results['total_amount_of_data_intervals'], average_results[res], '*--', alpha = 0.85, label=res)
plt.ylabel('Error rate')
plt.xlabel('Amount of data [N]')
plt.title('The average results from ' + str(48) + ' runs')
plt.legend()
plt.show()