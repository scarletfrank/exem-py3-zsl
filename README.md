# Implementation of Visual Exemplars in Python3

> 代码还是一团糟的状态

## Methods

**EXEM** is abbreviation for exemplars, which is originated from "predicting visual exemplars of unseen classes for zero-shot learning".  Check the references for more details, 

**Dependencies**

- Numpy
- Matplotlib
    - pyplot
- Scikit-learn
    - decomposition(PCA)
    - manifold(t-SNE)
    - svm(NuSVR)
    - linear_model(Lasso, Ridge)
    - neighbors(KNeighborClassifier)
- Scipy
    - io(loadmat)
- PyTorch
    - nn.Linear
    - nn.Conv1d
    - nn.Dropout(p=0.5)
- Pillow
    - Image

## TODO

1. .ipynb => .py
2. 解决写死(hard code)的问题

## References

[EXEM](http://openaccess.thecvf.com/content_ICCV_2017/papers/Changpinyo_Predicting_Visual_Exemplars_ICCV_2017_paper.pdf)

[EXEM Original Matlab](https://github.com/pujols/Zero-shot-learning-journal)
