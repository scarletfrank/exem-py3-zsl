import numpy as np 

from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVR 
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd # process txt
from scipy.io import loadmat

from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(description="python implementation of EXEM")

parser.add_argument('name', type=str, help='dataset')
args = parser.parse_args()
print(args.name)
if　args.name == 'awa':
    folder = "G:/dataset/awa/"
    standard_path = "G:/dataset/standard_split/AWA2/"
    proposed_path = "G:/dataset/proposed_split/AWA2/" 
elif args.name == 'cub':
    folder = "G:/dataset/cub/CUB_200_2011/"
    standard_path = "G:/dataset/standard_split/CUB/"
    proposed_path = "G:/dataset/proposed_split/CUB/"
elif args.name == 'sun':
    folder = "G:/dataset/sun/"
    standard_path = "G:/dataset/standard_split/SUN/"
    proposed_path = "G:/dataset/proposed_split/SUN/"
else:
    print("The dataset you input is not supported. -h for more details")
    return


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

train_class, test_class = sstrain, sstest
train_class, test_class = pstrain, pstest

## Semantic Embedding Reading

## 


