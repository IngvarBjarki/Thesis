# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:33:31 2018

@author: Ingvar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:58:52 2018

@author: Ingvar Bjarki Einarsson
"""

################################################# IMPORT LIBRARIES #########################################################################
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import time
#from optimization_algorithms import gradientDescentLasso, gradientDescent

is_Linux = False
if is_Linux:
    from optimization_algorithms_c import gradientDescent
else:
    from optimization_algorithms import  gradientDescent
    
######################################################################################################################################



def computer_1(X, y, theta):
    # X is the data attributes and y are the targets, theta is the gradient
    learning_rate = 0.0005
    m = len(y) 
    numIterations = 10000
    return(gradientDescent(X, y, theta, learning_rate, m, numIterations))
    

def computer_2(X, y, theta):
    # X is the data attributes and y are the targets, theta is the gradient
    learning_rate = 0.0005
    m = len(y)
    numIterations = 10000
    return(gradientDescent(X, y, theta, learning_rate, m, numIterations))
    
        
def evaluate(X_train, y_train, X_test, y_test, neural_net, knn):
    # neural_net is a boolen value, if user wants to get the results from a nural network
    # knn is a boolen value, if users want to get the reulsts from a nural network
    # if both values are false an information about the function will be stated
    print('X_train', len(X_train))
    print('y_train', len(y_train))
    print('X_test', len(X_test))
    print('y_test', len(y_test))
    if neural_net:
        mlp = MLPClassifier(hidden_layer_sizes=(30,30))
        mlp.fit(X_train, y_train)
        error_rate = 1 -  mlp.score(X_test, y_test)
        return(error_rate)        
        
    if knn:
        error_rates = []
        for i in range(0,100,10):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(X_train, y_train)
            error_rates.append(1-neigh.score(X_test, y_test))
        return(min(error_rates))
        
    if not neural_net and not knn:
        return("at least one argument must be true, to get neural_net or/and knn predictions")
    pass




def main(necessities):
    X, y, total_amount_of_data_intervals, weight_decay  = necessities
    # X is the data, y the respone variable, num splits is the number of splits we will cut the data
    # to simulate data sets with different amount of data
    
    ###########  we want to split the data set into 2 different data set to simmulate ##########
    ########### 2 different users - but use different amount of data each time           ##########
    n = len(y)
    if n % 2 == 0:
        n_half = int(n / 2)
        # we split the data for the two computers
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
        
    
    else:
        # if odd, we make the first data set have 1 more record than the second
        n_half = int((n + 1) / 2)
        # we split to train and test before we do feature selection
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:n_half], y[:n_half], test_size=0.1)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X[n_half:], y[n_half:], test_size=0.1)
        #VID BREYTUM TESSU TANNIG AD TAD VERDI ALLTAF JAFN MIKID I TEST.... SAMA FYRIR OFAN
        
    
    
    ################# The heart of the program ############################################
        
    score_feature_selection = []
    score_without_feature_selection = []
    # We check if our data set has even or odd numbers, and decide how to split the data
    
    for n in total_amount_of_data_intervals:  
        theta = np.array([0.0 for i in range(len(X[0]))])
        num_rounds = 100
        #weight_decay = 0.001 # for the lasso regression - they tried values from 0.0001-0.1
        Gamma = lambda x: np.sign(x)*(abs(x) - weight_decay)
        print('Iteration n:',n)
        
    
        for i in range(num_rounds):
            #laga nparray
            computer_1_gradient = computer_1(X_train1[:n], y_train1[:n], theta )
            computer_2_gradient = computer_2(X_train2[:n], y_train2[:n], theta)
            theta  = Gamma(0.5 * (computer_1_gradient + computer_2_gradient))
        
        # check what features to use
        threshold = 0.00001
        features = [1 if abs(feature) > threshold else 0 for feature in theta ]
        
        # now we remove the feature from the data set.. before we train the ML model
        features_to_remove = [i for i, feature in enumerate(features) if feature == 0]
        X_train1_with_feature_selection = X_train1
        X_test1_with_feature_selection = X_test1
        num_removed = 1
        print('features to remove', features_to_remove)
        for i, feature_to_remove in enumerate(features_to_remove):
            if i > 0:
                # sincce now the matrix is not as bigg as in the first run
                feature_to_remove -= num_removed
                num_removed += 1
            X_train1_with_feature_selection = np.delete(X_train1_with_feature_selection, feature_to_remove, axis = 1)
            X_test1_with_feature_selection = np.delete(X_test1_with_feature_selection, feature_to_remove, axis = 1)
            
        ############################ Evaluate the result by running classifiers from computer 1 ###########################
        ############################ on the data with and without the feature selection         ###########################  
        score_feature_selection.append(evaluate(X_train1_with_feature_selection[:n], y_train1[:n], X_test1_with_feature_selection, y_test1, True, False))
        score_without_feature_selection.append(evaluate(X_train1[:n], y_train1[:n], X_test1, y_test1, True, False))
        # held tad vanti n herna...
    return({'score_feature_selection':np.array(score_feature_selection), 'score_without_feature_selection': np.array(score_without_feature_selection)})





if __name__ == '__main__':
    start_time = time.time()
    #=============================== Data processing ===========================================================
    #===========================================================================================================
    is_model = 'digits' #'iris' 
    if is_model == 'digits':    
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        X_without_bias = digits.images.reshape((n_samples, -1))
        # we add bias term in front -- done for the gradient decent
        _records, _attributes = np.shape(X_without_bias)
        X = np.ones((_records, _attributes + 1))
        X[:,1:] = X_without_bias
        y = digits.target
        
    elif is_model == 'iris':
        iris = datasets.load_iris()
        X = []
        for attribute in iris.data:
            # we add bias term in the front -- done fo the gradient decent
            attribute = np.insert(attribute, 0, 1)
            X.append(attribute)
        X = np.asanyarray(X)
        y = iris.target    
    
    
    weight_decays = np.linspace(0.001, 0.1, num = 5)#values from 0.0001-0.1 .linspace(start, stop, num=50
    for weight_decay in weight_decays:
        num_splits = 15
        # 0.5 because we are spliting our data between computers..
        total_amount_of_data = [int(0.5*len(y)/num_splits) for i in range(num_splits)] #not lin space i numpy.. 
        total_amount_of_data_intervals = np.cumsum(total_amount_of_data)
        total_run_for_avg = 5
        args = [((X, y, total_amount_of_data_intervals, weight_decay))] * total_run_for_avg
        processes = Pool(total_run_for_avg)
        results = processes.map(main, args)
        
        score_feature_selection = np.zeros(num_splits)
        score_without_feature_selection = np.zeros(num_splits)
        for result in results:
            score_feature_selection += result['score_feature_selection'] 
            score_without_feature_selection += result['score_without_feature_selection']
        score_feature_selection *= 1/num_splits
        score_without_feature_selection *= 1/num_splits
        
        if not is_Linux:
            print('weightdecay', weight_decay)
            line_up, = plt.plot(total_amount_of_data_intervals, score_feature_selection, '--o', color = 'red', alpha = 0.6, label = 'With feature selection')
            line_down, = plt.plot(total_amount_of_data_intervals, score_without_feature_selection, '--o', color = 'blue', alpha = 0.6, label = 'Without feature selection')
            plt.legend(handles=[line_up, line_down])
            plt.xlabel('N')
            plt.ylabel('Error rate')
            plt.title('Distributed Lasso',fontsize = 18)
            fig_name = 'results_lambda_{}'.format(weight_decay)
            plt.savefig('img/' + fig_name + '.png', format = 'png')
            plt.savefig('img/' + fig_name + '.eps', format = 'eps')
            plt.clf()
           
        	#plt.savefig('result.png')
          #plt.show()
        else:
            line_up, = plt.plot(total_amount_of_data_intervals, score_feature_selection, '--o', color = 'red', alpha = 0.6, label = 'With feature selection')
            line_down, = plt.plot(total_amount_of_data_intervals, score_without_feature_selection, '--o', color = 'blue', alpha = 0.6, label = 'Without feature selection')
            plt.legend(handles=[line_up, line_down])
            plt.xlabel('N')
            plt.ylabel('Error rate')
            plt.title('Distributed Lasso',fontsize = 18)
            fig_name = 'result_lambda_{}'.format(weight_decay)
            plt.savefig('img/' + fig_name + '.png', format = 'png')
            plt.savefig('img/' + fig_name + '.eps', format = 'eps')
            plt.clf()
          #plt.show()
        
    print('time:',time.time() - start_time)
