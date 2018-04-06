# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:03:57 2018
@author: Ingvar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from multiprocessing import Pool
#from numba_module import main_numba
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from functools import partial



#@jit
def main(all_args):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    print("starting....")
    debugg = False

    X_train, y_train, X_test, y_test, total_amount_of_data_in_interval, dimensionality, epsilons  = all_args
    
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = True)

    # shuffle the data for randomnes in the smaller values of n
    X_train, y_train = shuffle(X_train, y_train)
    regularization_constant =  5 # this was obtained from the tunning program
    
    
    all_accuracies = defaultdict(list)
    avg_noise_for_each_n = defaultdict(list)
    var_noise_for_each_n = defaultdict(list)
    all_noise = defaultdict(partial(defaultdict, list)) # defaultdict inside defaultdict
    # we use tuple because lambda functions are not pickable- thus dont work with multiprocessing -uses que
    all_weights = defaultdict(tuple)
    for n in total_amount_of_data_in_interval:
          
        clf = LogisticRegression(penalty="l2", C=1 / regularization_constant)
        clf.fit(X_train[:n], y_train[:n])
        if debugg:
            print(clf.score(X_test, y_test))
            print(len(X_train))
        weights = clf.coef_[0]

        
        scikit_proba = clf.predict_proba(X_test)
        scikit_predict = clf.predict(X_test)
        num_correct_predictions = 0
        for i in range(len(y_test)):
            arg = np.dot(weights, X_test[i])
            prediction_probability = sigmoid(arg)
            if debugg:
                print('my prediction', prediction_probability )
                print('scikit prediction proba', scikit_proba[i])
                print('scikit prediction', scikit_predict[i])
                print('truth', y_test[i])
            
            # check which class to predict
            if prediction_probability > (1 - prediction_probability):
                predicition = 1
            else:
                predicition = -1
            
            truth = y_test[i]
            if predicition == truth:
                num_correct_predictions += 1
        
        
        ## add the score
        # we put the 9999 to make the wighout dp last when sorted with the epsilons
        all_accuracies[(9999, 'Without DP')].append(1 - clf.score(X_test, y_test)) 
        # tak the absolute value of the weights and then store it for later analysis
        all_weights[str(n)] = (abs(weights))
        
            
        ############# add differential privacy #########################
        
        
        sensitivity = 2 / (n * regularization_constant)
        for epsilon in epsilons:
     
            noise = np.array([np.random.laplace(0, sensitivity / epsilon) for i in range(dimensionality)])
            weights_perturb = weights + noise
            
            # evaluate the model
            num_correct_predictions = 0
            for i in range(len(y_test)):
                arg = np.dot(weights_perturb, X_test[i])
                prediction_probability = sigmoid(arg)
                
                # check which class to predict
                if prediction_probability > (1 - prediction_probability):
                    predicition = 1
                else:
                    predicition = -1
                
                truth = y_test[i]
                if predicition == truth:
                    num_correct_predictions += 1
        
                
                #total_noise = sum(abs(noise)) # vantar liklega abs gildinn
                accur = num_correct_predictions / len(y_test)

            # first index has the lowest n and then it increases
            all_accuracies[(epsilon, '$\epsilon$ = ' + str(epsilon))].append(1 - accur)
            avg_noise_for_each_n[epsilon].append(np.mean(abs(noise)))#total_noise / num_rounds_to_avg)
            var_noise_for_each_n[epsilon].append(np.var(noise))
            
            all_noise[n][epsilon] = noise.tolist()
        all_noise[n][99999999999999999999] = weights.tolist() # add the weights at the end for plottting
# =============================================================================
#                 if n not in all_noise or epsilon not in all_noise[n] :
#                     all_noise[n][epsilon] = [noise]
#                 else:
#                     all_noise[n][epsilon].append(noise)
# =============================================================================
            #all_noise_and_weights['weights eps = ' + str(epsilon)].append()
    print("leaving!!!")
    return (all_accuracies, avg_noise_for_each_n, all_weights, var_noise_for_each_n, all_noise)
            
if __name__ == '__main__':
    print("hallo")
    
    # load the data and select the binary classificatio problem
    num1 = 4
    num2 = 9
    
    y_train = []
    X_train = []
    with open('../mnist_train.csv') as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_train.append(label)
                X_train.append(features) 
    

    y_train = np.asarray(y_train)
    X_train = np.asarray(X_train)
    
    X_train = normalize(X_train)
    y_train[y_train == num1] = -1
    y_train[y_train == num2] = 1
    


    y_test = []
    X_test = []        
    with open('../mnist_test.csv') as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_test.append(label)
                X_test.append(features) 
    
    
    y_test = np.asarray(y_test)
    X_test = np.asarray(X_test)
    
    X_test = normalize(X_test)
    y_test[y_test == num1] = -1
    y_test[y_test == num2] = 1
    
    print('Data has ben loaded..')
    
    
    # The epsilons we are going to try to differential privacy
    epsilons = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.002, 0.01, 0.1, 3, 10]
    dimensionality = len(X_train[0])
    number_of_training_samples = len(X_train)
    
    # Select The colorschemme for the plots  
    sns.set_style('whitegrid')
    # the color palette dose not have enough  colors so we add colors that go well with it
    sns.set_palette(sns.color_palette("Set1", n_colors = 9) + sns.color_palette("Set2", n_colors = 3)[0:3:2] + [(1.0, 191/255, 0.0)])
    
    # in each itreation we use different amount of data to see how the model improvese with increased data
    num_splits = 30    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] 
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    
    # Lets do multi-threading to speed things up!
    t1 = time.time()
    num_instances = 10
    p = Pool(10)
    args = [(X_train, y_train, X_test, y_test, total_amount_of_data_in_interval, dimensionality, epsilons )] * num_instances
    results_and_weights_perturb = p.map(main, args)
    p.close()
    p.join()

    print('Time taken for multiprocessing: {}'.format(time.time() - t1))

    # get three list out of a list with tuples of three
    results, noise, all_weights, variance_noise, all_noises = zip(*results_and_weights_perturb)
    

    ################# START off by analyzig the prediction error #######################

    # use lambda to be able to have np array inside defaultdict
    average_results = defaultdict(lambda:np.array([0.0 for i in range(num_splits)]))
    for result in results:
        #print(result)
        for item in result:
            average_results[item] += np.array(result[item])
    
       
    
    fig = plt.figure()
    ax = plt.subplot(111)

    
    for result in sorted(average_results):
        average_results[result] /= num_instances
        # result of 1 is the string represantation of the result
        ax.plot(total_amount_of_data_in_interval, average_results[result], '-*', label = result[1]) 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Shrink current axis by 25%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    #plt.legend(loc='upper center', bbox_to_anchor=(1.22, 1.01), fancybox=True, shadow=True)
    
    plt.ylabel('Error rate')
    plt.xlabel('Amount of training data [N] ')
    plt.title('Differentialy private Logistic Regression')
    plt.show()

    
    
    #%%
    ###################### write statistics to excel for generating table in latex ##################
    ###################### Lets make two tables, one for mean and one fore variance ############
    ######################    Also make box plot of the means and the variances     #########
    
   
# =============================================================================
#     
# =============================================================================
    
    
    # combine all the thread values...
    very_all_noise = defaultdict(lambda: defaultdict(list))
    for i, noise in enumerate(all_noises):
        for n in noise:
            item = noise[n]
            for eps in item:
                very_all_noise[n][eps] = very_all_noise[n][eps] + all_noises[i][n][eps]


    # plot them..
    x_labels = ['$\epsilon = {}$'.format(eps) for eps in epsilons]
    x_labels.append('weights')
    for n in very_all_noise:
        item = very_all_noise[n]
        if n == 393 or n == 11004:
            print('BOOOMMM!!!')
            to_plot = []
            to_plot1 = []
            for eps in item:
                to_plot.append(very_all_noise[n][eps])
                to_plot1.append([abs(value) for value in very_all_noise[n][eps]])
            
            plt.title('for n = {}'.format(n))
            ax = sns.boxplot(data=to_plot)
            plt.yscale('log')
            plt.xticks(range(len(to_plot)), x_labels, rotation=45)
            plt.show()
            
            plt.title('for n = {}'.format(n))
            ax = sns.boxplot(data=to_plot)
            plt.xticks(range(len(to_plot)), x_labels, rotation=45)
            plt.show()
            
            plt.title('The strength.. n = {} log axis'.format(n))
            ax = sns.barplot(data=to_plot1, estimator = sum)
            plt.yscale('log')
            plt.xticks(range(len(to_plot)), x_labels, rotation=45)
            plt.show()















# intialize list of lists which will form a pandas data structure to plot things
#     pandas_values_mean, pandas_values_var = [], []
#     lenght_of_all_columns_for_pandas = len(epsilons) + 2 # 2 because of n and the weights
#     for i in range(len(total_amount_of_data_in_interval)):
#         pandas_values_mean.append([])
#         pandas_values_var.append([])
# 
#     # add the number of data points as a first attribute
#     for i, n in enumerate(total_amount_of_data_in_interval):
#         pandas_values_mean[i].append(n)
#         pandas_values_var[i].append(n)
#     
# 
#    
#     ####### Lets analyse the Noise ##########
#    
#     # find all the mean values which correspond to the same n and the same epsilon
#     averaged_noise = defaultdict(dict)
#     for item in noise:
#         i = 0
#         for key in sorted(item):
#             values = item[key]
#             # SKODA SORTED HERNA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             for n, value in enumerate(values):
#                 # we use two keys so we can sort after the i key for plotting.. and writing to excel
#                 # however, to ensure that we are doing corectly we use the other key as well do see that 
#                 # everything matches and for readability..
#                 new_key = 'n = ' + str(total_amount_of_data_in_interval[n]) + ' ' + str(key)
#                 sorting_key = i
#                 
#                 if sorting_key not in averaged_noise:
#                     averaged_noise[sorting_key]['value'] = [value]
#                     averaged_noise[sorting_key]['instance'] = new_key
#                 else:    
#                     averaged_noise[sorting_key]['value'].append(value)
#                 
#                 i += 1
#     
#     
#     
#     # find all the variance values which correspond to the same n and the same epsilon
#     var_noise = defaultdict(dict)
#     for item in variance_noise:
#         i = 0
#         for key in sorted(item):
#             values = item[key]
#             
#             for n, value in enumerate(values):
#                 # we use two keys so we can sort after the i key for plotting.. and writing to excel
#                 # however, to ensure that we are doing corectly we use the other key as well do see that 
#                 # everything matches and for readability..
#                 new_key = 'n = ' + str(total_amount_of_data_in_interval[n]) + ' ' + str(key)
#                 sorting_key = i
#                 
#                 if sorting_key not in var_noise:
#                     var_noise[sorting_key]['value'] = [value]
#                     var_noise[sorting_key]['instance'] = new_key
#                 else:    
#                     var_noise[sorting_key]['value'].append(value)
#                 
#                 i += 1
#     
#     
#     
#     
#     
#     
#     # mogulega splitta a kommu
#     # Lets try to do boxplot of the variances of the noise for n = 393
#     everything = []
#     for key in averaged_noise:
#         if averaged_noise[key]['instance'][:7] == 'n = 393':
#             everything.append(averaged_noise[key]['value'])
#             
#     
#     plt.boxplot(everything)
#     plt.yscale('log')
#     plt.show()
#     
















# =============================================================================
#     
#     
#     # calculate the statistics for the noise/epsilons
#     j, n_index = 0, 0
#     # we loop thorugh all epsilons for each n
#     for key in sorted(averaged_noise):
#         print(averaged_noise[key]['instance'])
#         pandas_values_mean[n_index].append(np.mean(averaged_noise[key]['value']))#averaged_noise[key]['value']))
#         pandas_values_var[n_index].append(np.var(var_noise[key]['value']))#averaged_noise[key]['value']))
#         if j == len(epsilons) - 1:
#             j = 0
#             n_index += 1
#         else:
#             j += 1
#     
#     
# 
#      ### analyse the weights ###
#     # Lets average all the instances of the weights, we know that each list inside all_weights
#     # is ordered by n, so that they start at the smallest and then increase towards the biggest
#     averaged_weights = defaultdict(lambda:np.array([0.0 for i in range(dimensionality)]))
#     for weights in all_weights:
#         for key in weights:
#             averaged_weights[key] += weights[key]
#             
#     for key in averaged_weights:
#         averaged_weights[key] /= num_instances
#             
#     # Collect the data for each n in list so we can build pandas data frame
#     i = 0
#     for key in sorted(averaged_weights):
#         value = averaged_weights[key]
#         pandas_values_mean[i].append(np.mean(value))
#         pandas_values_var[i].append(np.var(value))
#         i += 1
# 
# 
# 
#     # FOR PANDAS -- theses are the column names MUNNA AD LAGA
#     names = ['N'] + ['$\epsilon = {}$'.format(epsilon) for epsilon in epsilons] + ['Weights']
#     
#     df_means = pd.DataFrame(pandas_values_mean, columns = names)
#     df_vars = pd.DataFrame(pandas_values_var, columns = names)
# 
#     # now we change the order in the 
#     
#     # make box plot of the means and the variances
#     ax = sns.boxplot(data=df_means[names[1:]], palette = 'Set1') # exclude the N's
#     plt.xticks(rotation=45)
#     plt.yscale('log')
#     plt.title('Mean log axis')
#     plt.show()
#     
#     # make box plot of the means and the variances
#     ax = sns.boxplot(data=df_means[names[1:]], palette = 'Set1') # exclude the N's
#     plt.xticks(rotation=45)
#     plt.title('Mean')
#     plt.show()
#     
#     
#     ax = sns.boxplot(data=df_vars[names[1:]], palette = 'Set1')
#     plt.yscale('log')
#     plt.xticks(rotation=45)#ax.set_xticklabels(rotation=30) # ef failar plt.xticks(rotation=45)
#     plt.title('Variance log')
#     plt.show()
#     
#     ax = sns.boxplot(data=df_vars[names[1:]], palette = 'Set1')
#     plt.xticks(rotation=45)#ax.set_xticklabels(rotation=30) # ef failar plt.xticks(rotation=45)
#     plt.title('Variance')
#     plt.show()
#     
#     
#     # make a bar plot of the sum of the means and variances
#     #!!! gaeti gert rauda linu efst med maxinu svi tad sjaist vel hvad tetta er langt fra
#     # TILLA AD SUMMA ALL NOTA  estimator=sum tarf ad fa abs sum nei spurning ad hafa mean....
#     ax = sns.barplot(data=df_means[names[1:]].abs(), palette = 'Set1',  ci = None) # exclude the N's
#     plt.yscale('log')
#     plt.xticks(rotation=45)#ax.set_xticklabels(rotation=30)
#     plt.title('Mean of the Means')
#     plt.show()
#     
#     ax = sns.barplot(data=df_vars[names[1:]].abs(), palette = 'Set1',  ci = None)
#     plt.yscale('log')
#     plt.xticks(rotation=45)#ax.set_xticklabels(rotation=30) # ef failar plt.xticks(rotation=45)
#     plt.title('Mean of the variance')
#     plt.show()
#     
#     # save the two dataframes as a table in excel
#     writer = pd.ExcelWriter('output.xlsx')
#     df_means.to_excel(writer, 'Sheet1')
#     df_vars.to_excel(writer,  'Sheet2')
#     writer.save()
# =============================================================================
