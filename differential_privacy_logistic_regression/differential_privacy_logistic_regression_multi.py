# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:03:57 2018

@author: Ingvar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xlwt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from multiprocessing import Pool

sns.set_style('whitegrid')
sns.set_palette(sns.color_palette("Reds_d", 9))

def main(all_args):
    
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    debugg = False

    X, y, total_amount_of_data_in_interval, test_size, dimensionality = all_args

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    regularization_constant = 5
    num_rounds_to_avg = 1000
    
    
    all_accuracies = defaultdict(list)
    avg_noise_for_each_n = defaultdict(list)
    # we use tuple because lambda functions are not pickable- thus dont work with multiprocessing -uses que
    all_weights = defaultdict(tuple)
    for n in total_amount_of_data_in_interval:
          
        clf = LogisticRegression(penalty="l2", C=regularization_constant)
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
        
        
        # add the score
        all_accuracies['Without DP'].append(1 - clf.score(X_test, y_test))
        all_weights[str(n)] = (weights)
        #accur = num_correct_predictions / len(y_test)
       
            
        ############# add differential privacy #########################
        
        sensitivity = 2 / (len(y_train) * regularization_constant)
        epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 3]
        for epsilon in epsilons:
            accur = 0
            total_noise = 0
            for i in range(num_rounds_to_avg):
                noise = np.array([np.random.laplace(0, sensitivity / epsilon) for i in range(dimensionality)])
                weights_perturb = weights + noise
                
                ######## total noise.. ########
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
            
                
                accur += num_correct_predictions / len(y_test)
                total_noise += sum(noise)

            # first index has the lowest n and then it increases
            all_accuracies['$\epsilon$ = ' + str(epsilon)].append(1 - accur / num_rounds_to_avg)
            avg_noise_for_each_n['noise eps = ' + str(epsilon)].append(total_noise / num_rounds_to_avg)
            #all_noise_and_weights['weights eps = ' + str(epsilon)].append()
    print('HALLLOOOO!!!!!!!!')
    print(all_weights)
    print('/n')
    return (all_accuracies, avg_noise_for_each_n, all_weights)
            
if __name__ == '__main__':
    
    # load the data
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target
    dimensionality = len(X[0])
    # now we only want to do binary classification of two numbers
    # so we take only number  -- 9 and 4 are probably most similar
    num1 = 4
    num2 = 9
    index_of_zeros =  np.flatnonzero( y == num1 ) #returns the indexes
    index_of_tows = np.flatnonzero( y == num2 )
     
    # merge the two together and  sort them
    new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
    new_indexes = np.sort(new_indexes)
    y = y[new_indexes]
    X = X[new_indexes]
    # since we are classifying with the sign - we translate the y vector  to -1 to 1
    y[y == num1] = -1
    y[y == num2] = 1
    
    test_size = 0.1
    number_of_training_samples = int((1 - test_size) * len(X)) # the train test split function also rounds down

    # in each itreation we use different amount of data to see how the model improvese with increased data
    num_splits = 15    
    total_amount_of_data = [int(number_of_training_samples / num_splits) for i in range(num_splits)] #not lin space i numpy..
    total_amount_of_data_in_interval = np.cumsum(total_amount_of_data)
    
    # Lets do multi-threading to speed things up!
    num_instances = 7
    p = Pool(4)
    # BREYTA HER!!!
    args = [(X, y, total_amount_of_data_in_interval, test_size, dimensionality)] * num_instances
    results_and_weights_perturb = p.map(main, args)

    # get three list out of a list with tuples of three
    results, noise, all_weights = zip(*results_and_weights_perturb)
    
    ################# START off by analyzig the prediction error #######################

    # use lambda to be able to have np array inside defaultdict
    average_results = defaultdict(lambda:np.array([0.0 for i in range(num_splits)]))
    for result in results:
        for item in result:
            average_results[item] += np.array(result[item])
    
    fig = plt.figure()
    ax = plt.subplot(111)

    
    for result in sorted(average_results):
        average_results[result] /= num_instances
        ax.plot(total_amount_of_data_in_interval, average_results[result], label = result, alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Shrink current axis by 25%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    
    #plt.legend(loc='upper center', bbox_to_anchor=(1.22, 1.01), fancybox=True, shadow=True)
    
    plt.ylabel('Error rate', fontsize = 14)
    plt.xlabel('Amount of training data [N] ',  fontsize = 14)
    plt.title('Differentialy private Logistic Regression', fontsize = 16)
    plt.show()

    
    print('\n\n')
    print(noise[0]['noise eps = 0.0005'])
    
    #%%
    ###################### write statistics to excel for generating table in latex ##################
    ###################### Lets make two tables, one for mean and one fore variance ############
    ######################    Also make box plot of the means and the variances     #########
    
    epsilons = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 3]
    pandas_values_mean = []
    pandas_values_var = []
    
    
    pandas_values_mean.append(list(total_amount_of_data_in_interval))
    pandas_values_var.append(list(total_amount_of_data_in_interval))
    
    print('Lets analys the weights..! \n')
    print('\n')
    

    # Lets average all the instances of the weights, we know that each list inside all_weights
    # is ordered by n, so that they start at the smallest and then increase towards the biggest
    averaged_weights = defaultdict(lambda:np.array([0.0 for i in range(dimensionality)]))
    for weights in all_weights:
        for key in weights:
            averaged_weights[key] += weights[key]
            
    for key in averaged_weights:
        averaged_weights[key] /= num_instances
            
    print('ALRIGHTY!!!!!')
    # Collect the data in list so we can build pandas data frame
    weights_means = []
    weights_var = []
    for key in sorted(averaged_weights):
        #print(key, np.mean(value), np.var(value))
        #sheet.write(1 + i, 1, np.mean(value))
        #sheet.write(1 + i, 16, np.var(value))
        value = averaged_weights[key]
        weights_means.append(np.mean(value))
        weights_var.append(np.var(value))

    pandas_values_mean.append(weights_means)
    pandas_values_var.append(weights_var)

    
    print('\nLets analys the noise!\n')
    
    # https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    import re
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split('(\d+)', text) ]

    averaged_noise = defaultdict(list)
    for item in noise:
        for key, values in item.items():
            for n, value in enumerate(values):   
                new_key = 'n = ' + str(total_amount_of_data_in_interval[n]) + ' ' + key
                averaged_noise[new_key].append(value)
    
    
    # transpose the pandas lists so they are in the correct order such that
    # each list is N, weights
    pandas_values_mean = list(map(list, zip(*pandas_values_mean)))
    pandas_values_var = list(map(list, zip(*pandas_values_var)))

    # calculate the statistics for the noise/epsilons
    j, n_index = 0, 0
    # we loop thorugh all epsilons for each n
    for key in sorted(averaged_noise, key=natural_keys):
        print(key, j)
        pandas_values_mean[n_index].append(np.mean(averaged_noise[key]))
        pandas_values_var[n_index].append(np.var(averaged_noise[key]))
        if j == len(epsilons) - 1:
            print('heos')
            j = 0
            n_index += 1
        else:
            j += 1
    print('DONE')
    
    # FOR PANDAS -- theses are the column names
    names = ['N', 'Weights'] + ['$\epsilon = {}$'.format(epsilon) for epsilon in epsilons]
    

    
    df_means = pd.DataFrame(pandas_values_mean, columns = names)
    df_vars = pd.DataFrame(pandas_values_var, columns = names)
    
    # make box plot of the means and the variances
    ax = sns.boxplot(data=df_means[names[1:]], palette = 'Set3') # exclude the N's
    ax.set_xticklabels(rotation=30)
    ax = sns.boxplot(data=df_vars[names[1:]], palette = 'Set3')
    ax.set_xticklabels(rotation=30) # ef failar plt.xticks(rotation=45)
    
    # make a bar plot of the sum of the means and variances
    #!!! gaeti gert rauda linu efst med maxinu svi tad sjaist vel hvad tetta er langt fra
    ax = sns.barplot(data=df_means[names[1:]], palette = 'Set3') # exclude the N's
    ax.set_xticklabels(rotation=30)
    ax = sns.barplot(data=df_vars[names[1:]], palette = 'Set3')
    ax.set_xticklabels(rotation=30) # ef failar plt.xticks(rotation=45)
    
    
    # save the two dataframes as a table in excel
    writer = pd.ExcelWriter('output.xlsx')
    df_means.to_excel(writer, 'Sheet1')
    df_vars.to_excel(writer,  'Sheet2')
    writer.save()
    
    
    '''
    i, j = 1, 0 # i for row j for col
    for key in sorted(averaged_noise, key=natural_keys):
        print(i, j)
        sheet.write(i, 2 + j, np.mean(averaged_noise[key]))
        sheet.write(i, 17 + j, np.var(averaged_noise[key]))
        if j == len(epsilons) - 1:
            i += 1
            j = 0
            
        else:
            j+= 1
        '''
 
    
  
    
    # Lets do box plots of the weights and the noise of all the 
    
    