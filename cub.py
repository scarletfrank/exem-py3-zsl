import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVR, SVR, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
# different regressor test
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet



import pandas as pd # process txt
from scipy.io import loadmat
from PIL import Image

from collections import defaultdict

folder = "G:/dataset/cub/CUB_200_2011/"
standard_path = "G:/dataset/standard_split/CUB/"
proposed_path = "G:/dataset/proposed_split/CUB/"

cls_to_idx = {}
with open(folder + "classes.txt", "r", encoding='utf-8') as f:
     for row in f.readlines():
         row = row.rstrip()
         idx, name = row.split()
         cls_to_idx[name] = int(idx) - 1 # custom will -1

sstrain, sstest = [], []
pstrain, pstest = [], []

# Standard Split 
with open(standard_path + "trainvalclasses.txt", "r", encoding='utf-8') as f:
    for row in f.readlines():
        row = row.rstrip()
        sstrain.append(cls_to_idx[row])

with open(standard_path + "testclasses.txt", "r", encoding='utf-8') as f:
    for row in f.readlines():
        row = row.rstrip()
        sstest.append(cls_to_idx[row])

print("standard_split:", len(sstrain), len(sstest))
# transform List(str) -> List(int)

# Proposed Split
with open(proposed_path + "trainvalclasses.txt", "r", encoding='utf-8') as f:
    for row in f.readlines():
        row = row.rstrip()
        pstrain.append(cls_to_idx[row])

with open(proposed_path + "testclasses.txt", "r", encoding='utf-8') as f:
    for row in f.readlines():
        row = row.rstrip()
        pstest.append(cls_to_idx[row])

print("proposed_split:", len(pstrain), len(pstest))


# Random Train & Test Class Split

train_class, test_class = [], [] # (5994, 5794)
X_class = list(range(200)) # label start from 0, you can adjust it
train_class, test_class = train_test_split(X_class, test_size=0.2)

train_class, test_class = pstrain, pstest


class cubRead:
    def __init__(self, p, train_split):
        """
        p: 数据集存放路径
        train_split: 给出训练集
        """
        X_class = list(range(200)) # custom: range(200)
        train_class = train_split
        test_class = list(filter(lambda i: i not in train_class, X_class))
        self.path = p
        # labels
        yp = self.path + "custom-cub-labels.txt"
        y = np.loadtxt(yp, delimiter=" ", encoding='utf-8')
        # visual features 2048 dimensions
        xp = self.path + "custom-cub-features.txt"
        x = np.loadtxt(xp, delimiter=" ", encoding='utf-8')
        i1 = np.isin(y, train_class)
        i2 = np.isin(y, test_class)
        self.X_train, self.X_test = x[i1], x[i2]
        self.y_train, self.y_test = y[i1], y[i2]
    def train_data(self):
        return self.X_train, self.y_train
    def test_data(self):
        return self.X_test, self.y_test
    def test(self): # 11788 all, add them to test
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)

cubReader = cubRead(folder, train_class)
cubReader.test() # 8855 + 2933 = 11788
# their cub data, 8823 + 2965, 有点问题

X_train, y_train = cubReader.train_data()
X_test, y_test = cubReader.test_data()
#X_train.shape, y_train.shape, X_test.shape, y_test.shape
## Hyper Parameters

# dimension of PCA

pca_d = 500

exemPCA = decomposition.PCA(n_components=pca_d)
exemPCA.fit(X_train)
X_train = exemPCA.transform(X_train)
X_test = exemPCA.transform(X_test)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# group up PCA projections

exem_train_group = defaultdict(list)

for c in train_class:
    exem_train_group[c] = []
        
for x, y in zip(X_train, y_train):
    exem_train_group[y].append(x)

# Average
exem_train = {}
std_train = {}
# standard deviation

k = 0

for item in exem_train_group.items():
    y, ary = item
    exem_train[y] = np.mean(ary, axis=0) # Key Sentence
    std_train[y] = np.std(ary, axis=0)

del exem_train_group



v_c_train = [exem_train[i] for i in train_class] # dict uses classes to index
sd_train = [std_train[i] for i in train_class]
sd_train = np.mean(sd_train, axis=0)

# class to index, the same
# normal
# semantic embedding of CUB, 200 birds, 312 attributes
se = folder + "attributes/class_attribute_labels_continuous.txt"
semat = np.loadtxt(se, delimiter=" ", encoding='utf-8')
semat = normalize(semat, norm='l2')
a_c_train, a_c_test = semat[train_class], semat[test_class]


print("SVR")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    #regressor = MLPRegressor(hidden_layer_sizes=(50, 10))
    regressor = SVR()
    regressor.fit(X, y)
    regress_group.append(regressor)

v_c_test = np.zeros((50, pca_d)) # 提前定义好exemplar矩阵, 200 * 0.2, standard split=50 test classes
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

print(v_c_test.shape, len(v_c_train), len(v_c_train[0]))
# also add up to 200 here

exem_X, exem_y = [], []

for i, c in enumerate(test_class):
    exem_X.append(v_c_test[i])
    exem_y.append(c)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(exem_X, exem_y)
print("1NN:{} ".format(neigh.score(X_test, y_test)))

sneigh = KNeighborsClassifier(n_neighbors=1, metric='seuclidean', 
                    metric_params={'V':sd_train})
sneigh.fit(exem_X, exem_y)
print("1NNs:{}".format(sneigh.score(X_test, y_test)))


print("Ridge")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    regressor = Ridge()
    regressor.fit(X, y)
    regress_group.append(regressor)

v_c_test = np.zeros((50, pca_d)) # 提前定义好exemplar矩阵, 200 * 0.2, standard split=50 test classes
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

print(v_c_test.shape, len(v_c_train), len(v_c_train[0]))
# also add up to 200 here

exem_X, exem_y = [], []

for i, c in enumerate(test_class):
    exem_X.append(v_c_test[i])
    exem_y.append(c)
    
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(exem_X, exem_y)
print("1NN:{} ".format(neigh.score(X_test, y_test)))

sneigh = KNeighborsClassifier(n_neighbors=1, metric='seuclidean', 
                    metric_params={'V':sd_train})
sneigh.fit(exem_X, exem_y)
print("1NNs:{}".format(sneigh.score(X_test, y_test)))


se = folder + "attributes/class_attribute_labels_continuous.txt"
semat = np.loadtxt(se, delimiter=" ", encoding='utf-8')
#semat = normalize(semat, norm='l2')
a_c_train, a_c_test = semat[train_class], semat[test_class]


print("Lasso")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    #regressor = MLPRegressor(hidden_layer_sizes=(50, 10))
    regressor = Lasso()
    regressor.fit(X, y)
    regress_group.append(regressor)

v_c_test = np.zeros((50, pca_d)) # 提前定义好exemplar矩阵, 200 * 0.2, standard split=50 test classes
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

print(v_c_test.shape, len(v_c_train), len(v_c_train[0]))
# also add up to 200 here

exem_X, exem_y = [], []

for i, c in enumerate(test_class):
    exem_X.append(v_c_test[i])
    exem_y.append(c)
    
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(exem_X, exem_y)
print("1NN:{} ".format(neigh.score(X_test, y_test)))

sneigh = KNeighborsClassifier(n_neighbors=1, metric='seuclidean', 
                    metric_params={'V':sd_train})
sneigh.fit(exem_X, exem_y)
print("1NNs:{}".format(sneigh.score(X_test, y_test)))



print("Elastic Net")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    regressor = ElasticNet()
    regressor.fit(X, y)
    regress_group.append(regressor)

v_c_test = np.zeros((50, pca_d)) # 提前定义好exemplar矩阵, 200 * 0.2, standard split=50 test classes
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

print(v_c_test.shape, len(v_c_train), len(v_c_train[0]))
# also add up to 200 here

exem_X, exem_y = [], []

for i, c in enumerate(test_class):
    exem_X.append(v_c_test[i])
    exem_y.append(c)
    
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(exem_X, exem_y)
print("1NN:{} ".format(neigh.score(X_test, y_test)))

sneigh = KNeighborsClassifier(n_neighbors=1, metric='seuclidean', 
                    metric_params={'V':sd_train})
sneigh.fit(exem_X, exem_y)
print("1NNs:{}".format(sneigh.score(X_test, y_test)))