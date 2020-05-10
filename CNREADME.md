# Title

## Abstract


## Introduction

## Related Works




## Our Method



## Experiment

*视觉特征*

采用 ResNet-101提取特征，最后一层卷积网络输出为2048维。（原本是池化层接softmax）

AwA2: 37322个样本，矩阵大小为37322x2048
CUB: 11788个样本
SUN: 14340个样本

*语义特征*

- AwA2: 50种动物，85个属性。矩阵大小为(50x85)
- CUB: 200种鸟类，312种二元属性(200x312)
- SUN: 717种场景，102种属性

## Conclusion

## log

### TODO

3.22 iterator

3.23 torch reshape tensor

4.x Add Standard Split and Proposed Split


[transfer](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)