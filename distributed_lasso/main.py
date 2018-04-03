"""
Created on Fri Mar  9 16:34:27 2018

@author: helga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:07:13 2018

@author: Ingvar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from multiprocessing import Pool
from computer import Computer
#%%
sns.set_style(style = 'darkgrid')
def main(args):
    print('starting..!.')
    # get all the relevant data from the preprocessing
    #X, y, num_splits = args
    #!!!!!!!X_train, X_test, y_train, y_test, X_train1, y_train1, X_train2, y_train2, total_amount_of_data_intervals = args
    X_train, y_train, X_test, y_test, num_splits, tunned_parameters = args
    

    num_test_samples = len(y_test)
    num_dimensions = len(X_train[0])
    
    # we split to train and test before we do feature selection
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    
    # shuffle to introduce randomness in the smaller samples
    X_train, y_train = shuffle(X_train, y_train)

        
    # in each itreation we use different amount of data to see how the model improvese with increased data    
    total_amount_of_data = [int(len(y_train)/num_splits) for i in range(num_splits)] 
    total_amount_of_data_intervals = np.cumsum(total_amount_of_data)
    total_amount_of_data_intervals = [2000, 4000, 7000, len(y_train)]
    #!!! muna ad eyda!!!!!!!!!!!!!!!
    #total_amount_of_data_intervals = total_amount_of_data_intervals[0:-1:6]
    
    # now we perform the distributed lasso regression
    num_rounds = 3000
    weight_decay = 10**(-9) # for the lasso regression - they tried values from 0.0001-0.1
    learning_rate = 10**(-1)
    #Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    
    accuracies_distributed = []
    accuracies_single = []
    accuracies_central = []
    for n in total_amount_of_data_intervals:
        #print('n',n)
        
        # split the data differently according to if it is odd or even
        if len(X_train) % 2 == 0:
            m = int(n / 2)
            # we split the data for the two computers
            X_train1, y_train1 = X_train[:m], y_train[:m]
            X_train2, y_train2 = X_train[m:], y_train[m:]
            
        else:
            # if odd, we make the first data set have 1 more record than the second
            m = int((n + 1) / 2)
            X_train1, y_train1 = X_train[:m], y_train[:m]
            X_train2, y_train2 = X_train[m:], y_train[m:]

            
         # make the two computer centers that run the program
        theta = np.zeros(num_dimensions)
        computer_1 = Computer(m)
        computer_2 = Computer(m)
        
        #learning_rate = tunned_parameters['distributed'][str(n)]['learning_rate']
        #weight_decay = tunned_parameters['distributed'][str(n)]['weight_decay']
        Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
        #print('Iteration n:',n)       
        for i in range(num_rounds):
            is_converged_computer_1, computer_1_gradient = computer_1.get_gradients(X_train1, y_train1, theta )
            is_converged_computer_2, computer_2_gradient = computer_2.get_gradients(X_train2, y_train2, theta )
            total_gradients = computer_1_gradient + computer_2_gradient
            theta  =  Gamma(theta - learning_rate * total_gradients)
            
            if is_converged_computer_1 and is_converged_computer_2:
                #!!! if either of the computers has converge we stopp
                break
        print('Number for rounds = {}'.format(i))
        # Evaluate the model -- check for error rate
        total_correct_distributed = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            #print(prediction, y_test[i])
            if prediction == y_test[i]:
                total_correct_distributed += 1
        
        print('\ntotal correct distributed lasso: ', total_correct_distributed)
        print('\nlen(X_train1): ', len(X_train1))
        accuracies_distributed.append(1 - total_correct_distributed / num_test_samples)
        
    ############## If only one computer did the analysis on there own data ############################
        theta = np.zeros(num_dimensions)
        #learning_rate = tunned_parameters['distributed'][str(n)]['learning_rate']
        #weight_decay = tunned_parameters['distributed'][str(n)]['weight_decay']
        computer_1.set_cost = 0.0
        theta = computer_1.lasso_gradiants(X_train1, y_train1, theta,
                                 learning_rate, num_rounds, weight_decay) 
        
        # Evaluate the model -- check for error rate
        total_correct_single = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_single += 1
        
        print('\ntotal correct single computer: ', total_correct_single)
        accuracies_single.append(1 - total_correct_single / num_test_samples)
    
    ############ If all data was at a centeralized location ###########################################
    
        theta = np.zeros(num_dimensions)
        #learning_rate = tunned_parameters['distributed'][str(n)]['learning_rate']
        #weight_decay = tunned_parameters['distributed'][str(n)]['weight_decay']
        computer_1.set_cost = 0.0
        computer_1.set_m = 2*m
        theta = computer_1.lasso_gradiants(X_train[:2*m], y_train[:2*m], theta,
                                 learning_rate, num_rounds, weight_decay)

        # Evaluate the model -- check for error rate
        total_correct_all_data = 0
        for i in range(num_test_samples):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_all_data += 1
        
        print('\ntotal correct - central location: ', total_correct_all_data)
        accuracies_central.append(1 - total_correct_all_data / num_test_samples)
    print('Leaving..!.1')
    return {'distributed':np.array(accuracies_distributed),
           'single':np.array(accuracies_single),
           'central_all_data':np.array(accuracies_central),
           'total_amount_of_data_intervals':np.array(total_amount_of_data_intervals)}

if __name__ == '__main__':

    #!!! passa
    num1 = 4
    num2 = 9
    
    y_train = []
    X_train_without_bias = []
    with open("../../data/mnist_train.csv") as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_train.append(label)
                X_train_without_bias.append(features) 
    

    y_train = np.asarray(y_train)
    X_train_without_bias = np.asarray(X_train_without_bias)
    
    X_train_without_bias = normalize(X_train_without_bias)
    y_train[y_train == num1] = -1
    y_train[y_train == num2] = 1
    


    y_test = []
    X_test_without_bias = []        
    with open("../../data/mnist_test.csv") as l:
        for i , line in enumerate(l):
            line = line.split(",")
            label = int(line[0])
            if label == num1 or label == num2:
                features = [float(i) for i in line[1:]]
                y_test.append(label)
                X_test_without_bias.append(features) 
    
    
    y_test = np.asarray(y_test)
    X_test_without_bias = np.asarray(X_test_without_bias)
    
    X_test_without_bias = normalize(X_test_without_bias)
    y_test[y_test == num1] = -1
    y_test[y_test == num2] = 1
    
    
    
    # load the parameters from the tunned model
    with open('parameters_tunned_new3_pink3.json') as f:
        tunned_parameters = json.load(f)
    
    print(tunned_parameters['distributed'].keys())
    print(len(tunned_parameters['distributed'].keys()))
    
    print('data has been loaded')
    print('length of n={}'.format(len(y_train)))
    
    
    
    # we reduce the dimensionality with PCA since the gradient is very sensable in
    # high dimensional space..
# =============================================================================
#     if is_pca:
#         pca = PCA(n_components=10)
#         pca.fit(X_without_bias)
#         X_without_bias = pca.transform(X_without_bias)
# =============================================================================
    
    # we add bias term in front -- done for the gradient calculations
    records, attributes = np.shape(X_train_without_bias)
    X_train = np.ones((records, attributes + 1))
    X_train[:,1:] = X_train_without_bias
    
    
    
    records, attributes = np.shape(X_test_without_bias)
    X_test = np.ones((records, attributes + 1))
    X_test[:,1:] = X_test_without_bias
    
   
    # applay multiprocessing to make the analysis faster
    # here we do some number of total instances to average and see how the model
    # behaves, this is due to the randomness in the train, test split
    t1 = time.time()
    total_instances = 3
    p = Pool(total_instances)
    num_splits = 4
    args = [(X_train, y_train, X_test, y_test, num_splits, tunned_parameters)]*total_instances#[(X_train, X_test, y_train, y_test, X_train1, y_train1, X_train2, y_train2, total_amount_of_data_intervals)]*total_instances
    results = p.map(main, args)
    p.close()
    p.join()
    print('Time taken for multiprocessing: {}'.format(time.time() - t1))
    average_results = {
            'distributed': np.zeros(num_splits),
            'single': np.zeros(num_splits),
            'central_all_data': np.zeros(num_splits),
            'total_amount_of_data_intervals': np.zeros(num_splits)
            }

    # calculate the average of all the instances
    for res in results:
        for key in res:
            average_results[key] += res[key]

    for key in average_results:
        average_results[key] /= total_instances
            
        
    ########### plot the data #########
    for res in average_results:
        if not res == 'total_amount_of_data_intervals':
            plt.plot(average_results['total_amount_of_data_intervals'], average_results[res], '*--', alpha = 0.85, label=res)
    plt.ylabel('Error rate')
    plt.xlabel('Amount of data [N]')
    plt.title('The average results from ' + str(total_instances) + ' runs')
    plt.legend()
    plt.show()