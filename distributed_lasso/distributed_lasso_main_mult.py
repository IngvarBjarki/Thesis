# -*- coding: utf-8 -*-
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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from optimization_algorithms import  gradientDescent, gradientDescent_converge

def main(args):
    # get all the relevant data from the preprocessing
    X, y, num_splits = args
    #X_train, X_test, y_train, y_test, X_train1, y_train1, X_train2, y_train2, total_amount_of_data_intervals = args
    
    def computer_1(X, y, theta):
        # X is the data attributes and y are the targets, theta is the gradient
        learning_rate = 0.0005
        m = len(y) 
        numIterations = 500000
        return(gradientDescent_converge(X, y, theta, learning_rate, m, numIterations))
        
    def computer_2(X, y, theta):
        # X is the data attributes and y are the targets, theta is the gradient
        learning_rate = 0.0005
        m = len(y) 
        numIterations = 500000
        return(gradientDescent_converge(X, y, theta, learning_rate, m, numIterations))
    
    



     
    # we split to train and test before we do feature selection
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # split the data differently according to if it is odd or even
    if len(X_train) % 2 == 0:
        n = int(len(X_train) / 2)
        # we split the data for the two computers
        X_train1, y_train1 = X_train[:n], y_train[:n]
        X_train2, y_train2 = X_train[n:], y_train[n:]
        print('test1')
        
    else:
        # if odd, we make the first data set have 1 more record than the second
        # split the data for the two computers
        n = int((len(X_train) + 1) / 2)
        X_train1, y_train1 = X_train[:n], y_train[:n]
        X_train2, y_train2 = X_train[n:], y_train[n:]
        
    # in each itreation we use different amount of data to see how the model improvese with increased data
    #num_splits = 15    
    total_amount_of_data = [int(n/num_splits) for i in range(num_splits)] #not lin space i numpy..
    total_amount_of_data_intervals = np.cumsum(total_amount_of_data)
    


    
    # now we perform the distributed lasso regression
    num_rounds = 200
    weight_decay = 0.005 # for the lasso regression - they tried values from 0.0001-0.1
    Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
    accuracies_distributed = []
    accuracies_mean = []
    accuracies_single = []
    accuracies_central = []
    for n in total_amount_of_data_intervals:  
        theta = np.array([0.0 for i in range(len(X_train[0]))])
        print('Iteration n:',n)
            
        for i in range(num_rounds):
            computer_1_gradient = computer_1(X_train1[:n], y_train1[:n], theta )
            computer_2_gradient = computer_2(X_train2[:n], y_train2[:n], theta)
            total_gradients = computer_1_gradient + computer_2_gradient
            
            theta  =  Gamma(theta - 0.0001 *total_gradients)
            
            #if i % 10 == 0:
                #print('**********{}***********'.format(i))
                #total_correct = 0
                #for i in range(len(y_test)):
                    #prediction = np.sign(np.dot(theta, X_test[i]))
                    #if prediction == y_test[i]:
                        #total_correct += 1
    
                #print('\ntotal correct: ', total_correct)
    
                
        # Evaluate the model -- check for error rate
        total_correct_distributed = 0
        for i in range(len(y_test)):
            prediction = np.sign(np.dot(theta, X_test[i]))
            #print(prediction, y_test[i])
            if prediction == y_test[i]:
                total_correct_distributed += 1
        
        print('\ntotal correct distributed lasso: ', total_correct_distributed)
        accuracies_distributed.append(1 - total_correct_distributed/len(y_test))
    
    
    ############## take the average ################################################
        theta = np.array([0.0 for i in range(len(X_train[0]))])
        for i in range(num_rounds):
            computer_1_gradient = computer_1(X_train1[:n], y_train1[:n], theta )
            computer_2_gradient = computer_2(X_train2[:n], y_train2[:n], theta)
            total_gradients = computer_1_gradient + computer_2_gradient
                
            theta  =  Gamma(0.5 *total_gradients)
            
            # Evaluate the model -- check for error rate
            total_correct_mean = 0
            for i in range(len(y_test)):
                prediction = np.sign(np.dot(theta, X_test[i]))
                #print(prediction, y_test[i])
                if prediction == y_test[i]:
                    total_correct_mean += 1
            
        print('\ntotal correct mean lasso: ', total_correct_mean)
        accuracies_mean.append(1 - total_correct_mean/len(y_test))
        
    ############## If only one computer did the analysis on there own data ############################
        theta = np.array([0.0 for i in range(len(X_train[0]))])
        for i in range(num_rounds):       
            theta = computer_1(X_train1[:n], y_train1[:n], theta )
            theta  =  Gamma(theta - 0.0001 *theta)
        
        # Evaluate the model -- check for error rate
        total_correct_single = 0
        for i in range(len(y_test)):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_single += 1
        
        print('\ntotal correct single computer: ', total_correct_single)
        accuracies_single.append(1 - total_correct_single/len(y_test))
    
    ############ If all data was at a centeralized location ###########################################
    
        theta = np.array([0.0 for i in range(len(X_train[0]))])
        for i in range(num_rounds):
            theta = computer_1(X_train, y_train, theta)
            theta  =  Gamma(theta - 0.0001 *theta)
    
        # Evaluate the model -- check for error rate
        total_correct_all_data = 0
        for i in range(len(y_test)):
            prediction = np.sign(np.dot(theta, X_test[i]))
            if prediction == y_test[i]:
                total_correct_all_data += 1
        
        print('\ntotal correct - central location: ', total_correct_all_data)
        accuracies_central.append(1 - total_correct_all_data/len(y_test))
    
    return {'distributed':np.array(accuracies_distributed),
           'mean':np.array(accuracies_mean),
           'single':np.array(accuracies_single),
           'central_all_data':np.array(accuracies_central),
           'total_amount_of_data_intervals':np.array(total_amount_of_data_intervals)}

if __name__ == '__main__':

    # get the data and preprocess it
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X_without_bias = digits.images.reshape((n_samples, -1))
    y = digits.target
     
    # now we only want to do binary classification of two numbers
    # so we take only number 0 and 2 ---- 9 and 4 are probably most similar
    index_of_zeros =  np.flatnonzero( y == 4 ) #returns the indexes
    index_of_tows = np.flatnonzero( y == 9 )
     
    # merge the two together and  sort them
    new_indexes = np.concatenate((index_of_zeros, index_of_tows), axis=0)
    new_indexes = np.sort(new_indexes)
    y = y[new_indexes]
    X_without_bias = X_without_bias[new_indexes]
    # since we are classifying with the sign - we translate the y vector  to -1 to 1
    y[y == 4] = -1
    y[y == 9] = 1
    # we add bias term in front -- done for the gradient decent
    records, attributes = np.shape(X_without_bias)
    X = np.ones((records, attributes + 1))
    X[:,1:] = X_without_bias
   


    p = Pool(4)
    total_instances = 30
    num_splits = 15
    args = [(X, y, num_splits)]*total_instances#[(X_train, X_test, y_train, y_test, X_train1, y_train1, X_train2, y_train2, total_amount_of_data_intervals)]*total_instances
    results = p.map(main, args, chunksize = 5)
    p.close()
    p.join()
    print('getting ready for plotting results...')
    average_results = {
            'distributed': np.zeros(num_splits),
            'mean':np.zeros(num_splits),
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
