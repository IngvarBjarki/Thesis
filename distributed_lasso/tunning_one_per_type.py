# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:07:13 2018

@author: Ingvar
"""

import numpy as np
import time
import json
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from multiprocessing import Pool
from computer import Computer
from collections import defaultdict, Counter

#%%
def get_best_param(all_instances):
    # all_instances_distributed is a list dict with two keyts -- parameters and error_rates
    most_comon_params = Counter(all_instances['parameters']).most_common()    
    
    # we know the first one is the biggest, but do we have ties
    biggest_indexes = [0]
    i = 1
    while True:        
        if i  == len(most_comon_params):
            break
        
        if most_comon_params[i-1][1] > most_comon_params[i][1]:
            break
        else:
            biggest_indexes.append(i)
        i += 1
            
    if len(biggest_indexes) == 1:
        print('Boom')
        error_rate = None
        weight_decay, learning_rate = most_comon_params[0][0]
        for i, value in enumerate(all_instances['parameters']):
            if value[0] == weight_decay and value[1] == learning_rate:
                error_rate = all_instances['error_rates'][i]
        best_param = (weight_decay, learning_rate)
        return (best_param, error_rate)
    else:
        #finna minsta
        possible_best_params = [(all_instances['error_rates'][i], i) for i in biggest_indexes]
        error_rate, index = min(possible_best_params)
        best_param = all_instances['parameters'][index]
        return (best_param, error_rate)



#%%

def main(args):
    print('starting..!.')
    # get all the relevant data from the preprocessing
    X, y = args

    # shuffle the data so we get different data for each run of this program
    # Then the program us ryb multiple times to get a good esitmate    
    X, y= shuffle(X, y)
    max_iterations = 3000 
    # in each itreation we use different amount of data to see how the model improvese with increased data    
    #total_amount_of_data = [int(num_samples/num_splits) for i in range(num_splits)] 
    #total_amount_of_data_in_intervals = np.cumsum(total_amount_of_data)
    #!!! muna ad eyda!!!!!!!!!!!!!!!
    #total_amount_of_data_in_intervals = total_amount_of_data_in_intervals[0:-1:6]
    
    
    # initialize dictonaries that contain information from the cross validation
 
    distributed_solution = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
    single_solution = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
    all_data_solution = {'weight_decays': [], 'learning_rates': [], 'error_rates':[]}
    
    # keep the param
    #weight_decays = [10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10), 10**(-11), 10**(-12)]
    #learning_rates = [10**(-4), 10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10), 10**(-11)]
    #weight_decays = [10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-9), 10**(-10), 10**(-11), 10**(-12), 10**(-13), 10**(-14), 10**(-15), 10**(-16)]
    #learning_rates = [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6)]
    
    weight_decays = [10**(-6), 10**(-8), 10**(-10), 10**(-12)]#, 10**(-9), 10**(-10)]
    learning_rates = [10**(-1), 10**(-2), 10**(-3), 10**(-4)]#, 10**(-3), 10**(-4)]
    all_results_for_each_n_distributed = []
    all_results_for_each_n_single = []
    all_results_for_each_n_central = []
    
    for weight_decay in weight_decays:
        for learning_rate in learning_rates:
            kf = KFold(n_splits = 5)
            error_rates_distributed = []
            error_rates_single = []
            error_rates_central = []
            # spurning med tehtta vegna tess ad testid eykst alltaf......
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                num_test_samples = len(y_test)
                num_dimensions = len(X_train[0])
                # split the data differently according to if it is odd or even
                if len(y_train) % 2 == 0:
                    m = int(len(y_train) / 2)
                    # we split the data for the two computers
                    X_train1, y_train1 = X_train[:m], y_train[:m]
                    X_train2, y_train2 = X_train[m:], y_train[m:]
                    
                else:
                    # if odd, we make the first data set have 1 more record than the second
                    m = int((len(y_train) + 1) / 2)
                    X_train1, y_train1 = X_train[:m], y_train[:m]
                    X_train2, y_train2 = X_train[m:], y_train[m:]
                    
                # now we perform the distributed lasso regression

                Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
            

                # initalize for the distributed algorithm
                theta = np.zeros(num_dimensions)
        
                 # make the two computer centers that run the program
                computer_1 = Computer(m)
                computer_2 = Computer(m)
        
                #print('Iteration n:',n)       
                for i in range(max_iterations):
                    is_converged_computer_1, computer_1_gradient = computer_1.get_gradients(X_train1, y_train1, theta )
                    is_converged_computer_2, computer_2_gradient = computer_2.get_gradients(X_train2, y_train2, theta )
                    total_gradients = computer_1_gradient + computer_2_gradient
                    theta  =  Gamma(theta - learning_rate * total_gradients)
                    
                    if is_converged_computer_1 or is_converged_computer_2:
                        # if either of the computers has converge we stopp
                        print('distributed finsihed with i = {}'.format(i))
                        break
                        
                # Evaluate the model -- check for error rate
                total_correct_distributed = 0
                for i in range(num_test_samples):
                    prediction = np.sign(np.dot(theta, X_test[i]))
                    #print(prediction, y_test[i])
                    if prediction == y_test[i]:
                        total_correct_distributed += 1

                error_rates_distributed.append(1 - total_correct_distributed/num_test_samples)
                
            ############## If only one computer did the analysis on there own data ############################
                theta = np.zeros(num_dimensions)
                computer_1.set_cost = 0.0
                theta = computer_1.lasso_gradiants(X_train1, y_train1, theta,
                                         learning_rate, max_iterations, weight_decay) 
                
                # Evaluate the model -- check for error rate
                total_correct_single = 0
                for i in range(num_test_samples):
                    prediction = np.sign(np.dot(theta, X_test[i]))
                    if prediction == y_test[i]:
                        total_correct_single += 1
                
                error_rates_single.append(1 - total_correct_single/num_test_samples)
            
            ############ If all data was at a centeralized location ###########################################
            
                theta = np.zeros(num_dimensions)
                computer_1.set_cost = 0.0
                computer_1.set_m = 2*m
                theta = computer_1.lasso_gradiants(X_train[:2*m], y_train[:2*m], theta,
                                         learning_rate, max_iterations, weight_decay)
        
                # Evaluate the model -- check for error rate
                total_correct_all_data = 0
                for i in range(num_test_samples):
                    prediction = np.sign(np.dot(theta, X_test[i]))
                    if prediction == y_test[i]:
                        total_correct_all_data += 1
                
                error_rates_central.append(1 - total_correct_all_data/num_test_samples)
            
            # After cross validation is finsihed for a pair of weight decay and learning rate we save the results
            # we use tuples so we can find the min error rate, and at the same time have the correct weight decay and learning rate
            all_results_for_each_n_distributed.append((np.mean(error_rates_distributed), weight_decay, learning_rate))
            all_results_for_each_n_single.append((np.mean(error_rates_single), weight_decay, learning_rate))
            all_results_for_each_n_central.append((np.mean(error_rates_central), weight_decay, learning_rate))
    #print('len(all_results_for_each_n_distributed)', len(all_results_for_each_n_distributed))
    #print('all_results_for_each_n_distributed', all_results_for_each_n_distributed)
    error_rate, weight_decay, learning_rate = min(all_results_for_each_n_distributed) 
    distributed_solution['weight_decays'].append(weight_decay)
    distributed_solution['learning_rates'].append(learning_rate)
    distributed_solution['error_rates'].append(error_rate)
    
    error_rate, weight_decay, learning_rate = min(all_results_for_each_n_single) 
    single_solution['error_rates'].append(error_rate)
    single_solution['weight_decays'].append(weight_decay)
    single_solution['learning_rates'].append(learning_rate)
    
    error_rate, weight_decay, learning_rate = min(all_results_for_each_n_central) 
    all_data_solution['error_rates'].append(error_rate)
    all_data_solution['weight_decays'].append(weight_decay)
    all_data_solution['learning_rates'].append(learning_rate)
    
    #print('len(distributed_solution[n][error_rates])', len(distributed_solution[n]['error_rates']))
    #print('distributed_solution[n][error_rates]', distributed_solution[n]['error_rates'])
    print('Leaving..!.1')
    return (distributed_solution, single_solution, all_data_solution)
if __name__ == '__main__':

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

    print('data has been loaded')
            
    
    # we add bias term in front -- done for the gradient calculations
    records, attributes = np.shape(X_train_without_bias)
    X_train = np.ones((records, attributes + 1))
    X_train[:,1:] = X_train_without_bias

   
    # applay multiprocessing to make the analysis faster
    # here we do some number of total instances to average and see how the model
    # behaves, this is due to the randomness in the train, test split
    t1 = time.time()
    total_instances = 5#45 
    p = Pool(total_instances)
    num_splits = 10
    args = [(X_train, y_train)]*total_instances#[(X_train, X_test, y_train, y_test, X_train1, y_train1, X_train2, y_train2, total_amount_of_data_intervals)]*total_instances
    result = p.map(main, args)
    
    
    distributed, single, central = zip(*result)
    p.close()
    p.join()
    #print(result)
    print('Time taken for multiprocessing: {}'.format(time.time() - t1))
    
    # flatten the data structure so it is wasier to work with and compair between diferenet runs
    all_instances_distributed = defaultdict(list)
    for distributed_item in distributed:
        all_instances_distributed['error_rates'] += distributed_item['error_rates']
        all_instances_distributed['parameters'] += list(zip(distributed_item['weight_decays'], distributed_item['learning_rates']))
    
    
    all_instances_single = defaultdict(list)
    for single_item in single:
        all_instances_single['error_rates'] += single_item['error_rates']
        all_instances_single['parameters'] += list(zip(single_item['weight_decays'], single_item['learning_rates']))
    
    all_instances_central= defaultdict(list)
    for  central_item in central:
        all_instances_central['error_rates'] += central_item['error_rates']
        all_instances_central['parameters'] += list(zip(central_item['weight_decays'], central_item['learning_rates']))
    
   
   
    
    
    
    
    # we pcik the pair of parameters which come most often -- if there is a tie we select the one with
    # the lowest error rate..
    tunned_params = {'distributed':{}, 'single':{}, 'central':{}}
    
    parameters, error_rate = get_best_param(all_instances_distributed)
    weight_decay, learning_rate = parameters
    # key needs to be string for the json object
    tunned_params['distributed'] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}
    
    
    parameters, error_rate= get_best_param(all_instances_single)
    weight_decay, learning_rate  = parameters
    # key needs to be string for the json object
    tunned_params['single'] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}


    parameters, error_rate= get_best_param(all_instances_central)
    weight_decay, learning_rate = parameters
    # key needs to be string for the json object
    tunned_params['central'] = {'weight_decay': weight_decay, 'learning_rate': learning_rate, 'error_rate': error_rate}        

        
    # load all the best parameters into json so we can easly acess them in our main program    
    with open('parameters_tunned_new3_pink3_one_per_type.json', 'w') as f:
        json.dump(tunned_params, f)
        
    print('data has been loaded to parameters.json')
    
    print('\n The total time of tunning was: {}'.format(time.time() - t1))
    

