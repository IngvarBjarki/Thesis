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

np.seterr(all='ignore')


def main(all_args):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    print("starting....")
    debugg = False

    X_train, y_train, X_test, y_test, total_amount_of_data_in_interval, dimensionality, epsilons  = all_args
    
    
    # shuffle the data for randomnes in the smaller values of n
    X_train, y_train = shuffle(X_train, y_train)
    regularization_constant =  100 #5 # this was obtained from the tunning program
    
    
    all_accuracies = defaultdict(list)
    noise_and_weights = defaultdict(partial(defaultdict, list)) # defaultdict inside defaultdict
   
    for n in total_amount_of_data_in_interval:
          
        clf = LogisticRegression(penalty="l2", C=1 / regularization_constant)
        clf.fit(X_train[:n], y_train[:n])
        if debugg:
            print(clf.score(X_test, y_test))
            print(len(X_train))
            scikit_proba = clf.predict_proba(X_test)
            scikit_predict = clf.predict(X_test)
        
        weights = clf.coef_[0]
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
        # we put the 9999 to make the wweight be the last when sorted with the epsilons
        all_accuracies[(9999, 'Without DP')].append(1 - clf.score(X_test, y_test)) 
        
            
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
        
                accur = num_correct_predictions / len(y_test)

            # first index has the lowest n and then it increases
            all_accuracies[(epsilon, '$\epsilon$ = {}'.format(epsilon))].append(1 - accur)
            noise_and_weights[n][epsilon] = noise.tolist()
        noise_and_weights[n][99999999999999999999] = weights.tolist() # add the weights at the end for plottting

    print("leaving!!!")
    return (all_accuracies, noise_and_weights)
            
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
    #epsilons = [10, 50, 100, 200, 500, 1000, 10000000000000000]
    dimensionality = len(X_train[0])
    number_of_training_samples = len(X_train)
    
    # Select The colorschemme for the plots  
    sns.set_style('darkgrid')
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
    results,  noise_and_weights = zip(*results_and_weights_perturb)
    

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
        # result of 1 is the string represantation of the result
        ax.plot(total_amount_of_data_in_interval, average_results[result], '-*', label = result[1]) 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Shrink current axis by 25%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        
    plt.ylabel('Error rate')
    plt.xlabel('Amount of training data [N]')
    plt.title('Differentialy private Logistic Regression')
    plt.savefig('error_rate.png')
    plt.show()

    
    
    #%%
    ###################### write statistics to excel for generating table in latex ##################
    ###################### Lets make two tables, one for mean and one fore variance ############
    ######################    Also make box plot of the means and the variances     #########
    
    # combine all the thread values...
    noise_and_weights_combined = defaultdict(lambda: defaultdict(list))
    for i, noise in enumerate(noise_and_weights):
        for n in noise:
            item = noise[n]
            for eps in item:
                noise_and_weights_combined[n][eps] = noise_and_weights_combined[n][eps] + noise_and_weights[i][n][eps]


    # plot the magnitude and the distributtion of all the weights and noises generated
    x_labels = ['$\epsilon = {}$'.format(eps) for eps in epsilons]
    x_labels.append('weights')
    for n in noise_and_weights_combined:
        item = noise_and_weights_combined[n]
        if n == total_amount_of_data_in_interval[0] or n == total_amount_of_data_in_interval[int(num_splits / 2)] or n == total_amount_of_data_in_interval[-1]:
            noise_and_weights_distribution = []
            noise_and_weights_magnitude = []
            for eps in item:
                noise_and_weights_distribution.append(noise_and_weights_combined[n][eps])
                noise_and_weights_magnitude.append([abs(value) for value in noise_and_weights_combined[n][eps]])
            num_labels = len(noise_and_weights_distribution)
            plt.title('Distribution of noise and weights for n = {}'.format(n))
            ax = sns.boxplot(data=noise_and_weights_distribution)
            plt.xticks(range(num_labels), x_labels, rotation=45)
            plt.savefig('distributionOfNoiseWeights_n={}.png'.format(n))
            plt.show()
            
            plt.title('Distribution of noise and weights for n = {} with log axis'.format(n))
            ax = sns.boxplot(data=noise_and_weights_distribution)
            plt.yscale('log')
            plt.xticks(range(num_labels), x_labels, rotation=45)
            plt.savefig('distributionOfNoiseWeightsLog_n={}.png'.format(n))
            plt.show()
            
            plt.title('Magnitude off noise and the weights.. n = {} with log axis'.format(n))
            ax = sns.barplot(data=noise_and_weights_magnitude , estimator = sum)
            plt.yscale('log')
            plt.xticks(range(num_labels), x_labels, rotation=45)
            plt.savefig('magnitudeOfNoiseAndWeights_n_{}'.format(n))
            plt.show()
            
    
    
    # write variances of the noise and mean of the weights to pandas inorder to make
    # a excel file to copy into latex.....
    
    statistics = [] 
    for n in noise_and_weights_combined:
        item = noise_and_weights_combined[n]
        statistics.append([])
        for i, eps in enumerate(item):
            name = x_labels[i]
            if name != 'weights':
                # get the variance of all the noise's
                statistics[-1].append(np.var(noise_and_weights_combined[n][eps]))
            else:
                # get the variance and the mean of the weights
                statistics[-1].append(np.mean(noise_and_weights_combined[n][eps]))
                statistics[-1].append(np.var(noise_and_weights_combined[n][eps]))
    
    names = x_labels + ['\bar{weights}']
    statistics = pd.DataFrame(statistics, columns = names) 
    writer = pd.ExcelWriter('output.xlsx')
    statistics.to_excel(writer, 'Sheet1')
    writer.save()

