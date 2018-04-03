# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:37:54 2018

@author: s161294
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

sns.set_context("paper")

# load the data and put it into a pandas data frame
train = []
train_names = []
X_train = []
y_train = []
X_train_four_or_nine = []
y_train_four_or_nine = []
X_train_zero_or_one = []
y_train_zero_or_one = []
school = "C:/Users/s161294/OneDrive - Danmarks Tekniske Universitet/"
home = '../../data/'
with open(school + 'mnist_train.csv') as l:
    for i , line in enumerate(l):
        line = line.split(",")
        features = [float(i) for i in line[1:]]
        target = int(line[0]) 
        row = [target] + features
        train.append(row)
        X_train.append(features)
        y_train.append(target)
        if target == 4 or target == 9:
            X_train_four_or_nine.append(features)
            y_train_four_or_nine.append(target)
        elif target == 0 or target ==1:
            X_train_zero_or_one.append(features)
            y_train_zero_or_one.append(target)


test = []
test_names = []
with open(school + 'mnist_test.csv') as l:
    for i , line in enumerate(l):
        line = line.split(",")
        row = [int(line[0])] + [float(i) for i in line[1:]]
        test.append(row)



names = ['y'] + ['X{}'.format(i) for i in range(784)] # we know there are 784 features
df_train = pd.DataFrame(train, columns = names )
df_test = pd.DataFrame(test, columns = names )

#%%
# plot the images to see how they look

images_to_show = list(range(10))
for i, target in enumerate(y_train):
    if target in images_to_show:
        sns.set_style("ticks")
        pxles = np.array(X_train[i])
        pxles = pxles.reshape((28, 28))        
        plt.imshow(pxles, cmap = 'gray')
        plt.savefig('number_{}.eps'.format(target), format='eps')
        plt.show()
        images_to_show.remove(target)

#%%

sns.set_style("darkgrid")


colors = ['#e6194b',
          '#0082c8',
          '#d2f53c',
          '#3cb44b',
          '#f032e6',
          '#911eb4',
          '#46f0f0',
          '#f58231', 
          '#008080',
          '#ffe119']

X_train = normalize(X_train)
# visulize all attributes in the data set
num_components = 2
pca = PCA(n_components = num_components)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)) 

total_explained_variance = sum(pca.explained_variance_ratio_)

train_pca = pca.transform(X_train) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


#sns.pairplot(df_pca, hue="y")
sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False, palette = colors)
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_all.eps', format='eps')
plt.show()
#%%
# visulize only 4 and 9
X_train_four_or_nine = normalize(X_train_four_or_nine)
num_components = 2
pca = PCA(n_components = num_components)
pca.fit(X_train_four_or_nine)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)) 

train_pca = pca.transform(X_train_four_or_nine) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train_four_or_nine



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


#sns.pairplot(df_pca, hue="y")
sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False, palette = [colors[4], colors[9]])
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_four_and_nine.eps', format='eps')
plt.show()

#%%
# visulize only 0 and 1

num_components = 2
X_train_zero_or_one = normalize(X_train_zero_or_one)
pca = PCA(n_components = num_components)
pca.fit(X_train_zero_or_one)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)) 

train_pca = pca.transform(X_train_zero_or_one) # make this more pretty l8ter
records, attributes = np.shape(train_pca)
train_pca_ones = np.ones((records, attributes + 1))
train_pca_ones[:,1:] = train_pca
train_pca_ones[:,0] = y_train_zero_or_one



names = ['y'] # we know y is the first one
for i in range(num_components):
    names.append('pca{}'.format(i+1))

df_pca = pd.DataFrame(train_pca_ones, columns = names)


#sns.pairplot(df_pca, hue="y")
sns.lmplot("pca1", "pca2", data=df_pca, hue='y', fit_reg=False, palette = [colors[0], colors[1]])
#plt.title('explained variance = {}'.format(sum(pca.explained_variance_ratio_)))
plt.savefig('pca_plot_zero_and_one.eps', format='eps')
plt.show()


#%%
# the evalution of pca

sns.set_style("ticks")
pxles = np.array(train_pca[0])
#pxles = pxles.reshape((28, 28))        
plt.imshow(pxles, cmap = 'gray')
plt.savefig('number_{}.eps'.format(target), format='eps')
plt.show()
images_to_show.remove(target)


#%%
# find out how many data points are in each category of the targets
print('Categories in train set')
print(df_train['y'].value_counts())
print('\n Categories in test set')
print(df_test['y'].value_counts())

#%%
# create histogram of the number of times a value comes upp in pandas..
###! spa i pixlunum!!!!
sns.set_style("darkgrid")
sns.distplot(np.asarray(X_train).flatten())
plt.title('Number of pixels associated with there color value')
plt.xlabel('color value')
plt.ylabel('rate of number of pixels in each bin')
plt.show()
plt.hist(np.asarray(X_train).flatten(), bins = 254)
plt.show()




