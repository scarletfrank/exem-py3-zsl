"""
task: 'train' 'test'
opt: opt.C the regularizer coefficient of nu-SVR (e.g, 2.0)
     opt.nu the nu-SVR parameter (e.g 2^[-10:0])
     opt.pca_d the PCA dimensionality (e.g. 500)
     opt.ind_split: AWA: []; CUB: choose one from 1:4; SUN: choose one from 1:10
"""
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

import pandas as pd # process txt
from scipy.io import loadmat
from PIL import Image
from sklearn.svm import NuSVR # nu-SVR, implement from libsvm

nr_fold = 5
# training & validation
def train():
    for f in range(nr_fold):
        Xbase = Xtr
        # do_pca
        Xbase = []
        pca = decomposition.PCA(n_components=opt.pca_d)
        pca.fit(Xbase)
        Xbase = pca.transform(Xbase) # 20s
        Xval = Xtr[:] # a part of training
        Xval = pca.transform(Xbase)
        for g in range(opt.gamma):
            # SVR kernel
            Ker_base, Ker_val = [], []
            #for d in range(opt.pca_d):
                #for c in range(opt.C):
                    #for n in range(opt.nu): unknown iteration
            for j in range(opt.pca_d):
                regressor = NuSVR(nu=opt.nu, C=2.0, kernel=Ker_base)
                regressor.fit()
                mean_rc = regressor.predict() # Xbase, Xval

# testing
def test():
    pass





                


