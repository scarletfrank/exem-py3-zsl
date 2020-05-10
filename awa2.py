import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVR, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
# different regressor test
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression


import pandas as pd # process txt
from scipy.io import loadmat
from PIL import Image

from collections import defaultdict

folder = "G:/dataset/awa/"

standard_path = "G:/dataset/standard_split/AWA2/"
proposed_path = "G:/dataset/proposed_split/AWA2/" 

cls_to_idx = {}
with open(folder + "classes.txt", "r", encoding='utf-8') as f:
     for row in f.readlines():
         row = row.rstrip()
         idx, name = row.split()
         cls_to_idx[name] = int(idx)

sstrain, sstest = [], []
pstrain, pstest = [], []

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
X_class = list(range(1, 51))
train_class, test_class = train_test_split(X_class, test_size=0.2)

train_class, test_class = pstrain, pstest

# semantic embedding of AwA2, 85 attibutes
seb = folder + "predicate-matrix-binary.txt" 
se = folder + "predicate-matrix-continuous.txt" 
# np.loadtxt(seb, delimiter=" ", encoding='utf-8'), binary works perfectly
# np.loadtxt(se, delimiter="  ", encoding='utf-8') # single, double, triple space exists
semat = np.zeros((50, 85))
with open(se, 'r', encoding='utf-8') as f:
    rows = f.readlines()
    cnt = 0
    for row in rows:
        row = row.strip()
        semat[cnt, :] = np.array(row.split(), dtype='float64')
        cnt = cnt + 1

semat.shape # unnormalize

# 写成一个类方便其他人使用？
# combine x_n and y_n to a class
class awaRead:
    def __init__(self, p, train_split):
        """
        p: 数据集存放路径
        train_split: 给出训练集
        """
        X_class = list(range(1, 51))
        train_class = train_split
        test_class = list(filter(lambda i: i not in train_class, X_class))
        self.path = p
        # labels
        yp = self.path + "ResNet101/AwA2-labels.txt"
        y = np.loadtxt(yp, delimiter=" ", encoding='utf-8')
        # visual features 2048 dimensions
        xp = self.path + "ResNet101/AwA2-features.txt"
        x = np.loadtxt(xp, delimiter=" ", encoding='utf-8')
        i1 = np.isin(y, train_class)
        i2 = np.isin(y, test_class)
        self.X_train, self.X_test = x[i1], x[i2]
        self.y_train, self.y_test = y[i1], y[i2]
    def train_data(self):
        return self.X_train, self.y_train
    def test_data(self):
        return self.X_test, self.y_test
    def test(self):
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)



#len(test_class), len(train_class) # (10, 40)
awaReader = awaRead(folder, train_class)
awaReader.test()

X_train, y_train = awaReader.train_data()
X_test, y_test = awaReader.test_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
## Hyper Parameters

# dimension of PCA
pca_d = 500

exemPCA = decomposition.PCA(n_components=pca_d) # 1024 => 500
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
k = 0

for item in exem_train_group.items():
    y, ary = item
    exem_train[y] = np.mean(ary, axis=0) # Key Sentence
    std_train[y] = np.std(ary, axis=0)

del exem_train_group

trai, tesi = [c-1 for c in train_class], [c-1 for c in test_class] # class to index, start from 0
a_c_train, a_c_test = semat[trai], semat[tesi]
v_c_train = [exem_train[i] for i in train_class] # dict uses classes to index
sd_train = [std_train[i] for i in train_class]
sd_train = np.mean(sd_train, axis=0)



print("SVR")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    #regressor = Ridge()
    regressor = SVR()
    regressor.fit(X, y)
    regress_group.append(regressor)


v_c_test = np.zeros((10, pca_d)) # 提前定义好exemplar矩阵
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

v_c_test.shape, len(v_c_train), len(v_c_train[0])

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
    #regressor = Ridge()
    regressor = Ridge()
    regressor.fit(X, y)
    regress_group.append(regressor)


v_c_test = np.zeros((10, pca_d)) # 提前定义好exemplar矩阵
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

v_c_test.shape, len(v_c_train), len(v_c_train[0])

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


print("Lasso")

regress_group = []
k = 0
for j in range(pca_d):
    X = a_c_train # 
    y = [vc[j] for vc in v_c_train]
    #regressor = Ridge()
    regressor = Lasso()
    regressor.fit(X, y)
    regress_group.append(regressor)


v_c_test = np.zeros((10, pca_d)) # 提前定义好exemplar矩阵
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

v_c_test.shape, len(v_c_train), len(v_c_train[0])

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
    #regressor = Ridge()
    regressor = ElasticNet()
    regressor.fit(X, y)
    regress_group.append(regressor)


v_c_test = np.zeros((10, pca_d)) # 提前定义好exemplar矩阵
# 对每一个维度进行预测
for j in range(pca_d):
    v_c_test[:, j] =  regress_group[j].predict(a_c_test) # 10 dimension , assign to column

v_c_test.shape, len(v_c_train), len(v_c_train[0])

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