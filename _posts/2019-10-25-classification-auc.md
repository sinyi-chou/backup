---
layout: post
title: AUC & classification
date: 2019-10-25
Author: Sin-Yi Chou
categories: [Data Science]
tags: [data science, classification, machine learning]
comments: true
---

While taking courses of machine learning, you would always hear that AUC is a useful metric to evaluate the model. The higher the values (range from 0 to 1), the better the model is.  However, what is AUC? What's the theory behind to make it a great metric? We'll have a deep dive in this post.

# AUC - Aread under ROC curve
AUC is short for Area under ROC(Receiver operating characteristics) curve. AUC serves as a general performance metrics to measure the binary classification model performance. It is a better measurement than accuracy, error rate, etc. with imbalanced class distribution. AUC could only be calculated while the model is able to provide probability prediction. In order to understand AUC, let's talk about ROC curve first.

# ROC Curve - Receiver operating characteristics

Let's start with confusion matrix with binary classification. As the figure shown below, confusion matrix summarizes all the conditions in a binary classification model. One thing to note here is that the confusion matrix is based on certain threshold to identify positive and negative class. False positive and false negative indicate that the case is identified by model incorrectly.

![confusion matrix](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/confusion_matix.png)
<p align="center">
<img src="/assets/classification/metric_definition.png">
<img src="/assets/classification/metric_definition.png" src="#" width="50%"  />
<img src="/images/classification/confusion_matix.png"/>
</p>
All those statistics are used to compute metrics to show the model performance in different perspective, as the figure shown below. Among all those metrics, false positive rate(x-axis) and true positive rate(y-axis) are used to plot ROC curve.

![definition](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/confusion_matix.png)

Since there are countless values for the cut-off threshold to generate the predicted labels, AUC provides an aggregated view across all threshold.

 The purpose is to allow the viewer to assess the accuracy of the test M for any possible value of the cutoff c. This aids in deciding what cutoff to use in practice, comparing different tests for the same thing, and for evaluating the overall accuracy. A key advantage of our approach is that the values of the cutoffs are visible!

![threshold animation](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/classification_threshold.gif)

![proba table](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/prob_table.png)

![roc animation](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/roc_plot_animation.gif)

![roc plot](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/ROC_plot.png)

# Statistic Interpretation of AUC
Rank
positive over negative

# Visualization of AUC
threshold + dot
$$AUC = P(f(x+)>f(x-)|class(x+)=1, class(x-)=0)$$
two distribution separation

# Summary
1. auc can only be used in two class -> if you have multiple class, one to all AUC would be an option/ micro macro  

1. Even if auc is powerful, auc is not a cure-all.

### Reference
1. [An introduction to ROC analysis](https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X)
2. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
3. http://www.medicalbiostatistics.com/roccurve.pdf

. For a binary test and a binary disease state, the following table summarizes the possible errors that one can make using the test as a prediction of a disease state measured by some gold standard

If the test is continuous, say M, then a test positive is defined as M>c. Now we consider measures of accuracy as functions of c, i.e.

TPF(c)=Pr{M>c|D=1}
FPF(c)=Pr{M>c|D=0}

The ROC curve is a plot of FPF(c) versus TPF(c).
