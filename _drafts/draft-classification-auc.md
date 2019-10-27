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
AUC is short for Area under ROC(Receiver operating characteristics) curve. AUC serves as a general performance metrics to measure the binary classification model performance without specific threshold. AUC could only be calculated while the model is able to provide probability prediction. In order to understand AUC, let's talk about ROC curve first.

# ROC Curve - Receiver operating characteristics

Let's start with confusion matrix with binary classification. As the figure shown below, confusion matrix summarizes all the conditions - true positive, false positive, true negative and false negative in a binary classification model. False positive and false negative indicate that the case is identified by model incorrectly. True positive and True negative show the predictions are correct. One thing to note here is that all the measurements are based on individual threshold (cutoffs) of the probability to identify positive and negative class. In other words, all of the measures are single-threshold measures. They couldn't provide an overview of the performance with varying threshold.

<p align="center">
<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/metric_definition.png" src="#" width="50%"  />
<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/confusion_matix.png" src="#" width="50%" />
</p>


 Threshold-invariant metrics are able to provide the overall performance of model despite of the chosen threshold, such as ROC/AUC & logloss. The figure below is a plot of ROC curve, which false positive rate -FPR, also called recall, (x-axis) and true positive rate - TPR (y-axis) are used for plotting. Each point in ROC curve is computed based on one cutoff threshold, labelled on the plot. AUC is the area under the curve. While ROC curve moves toward left-top, the value of AUC would increase with higher TPR and lower FPR for any threshold.

<p align="center">
<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/ROC_plot.png" src="#" width="70%" />
</p>

Let's look at the animation below with changing cut-off threshold. At first, FPR is 0 and TPR/Recall is 1 with higher threshold. While the threshold become smaller, TPR decreases and FPR increases with more false positive are classified. TPR would reach 1 until all the positive classes are identify correctly. If the goal is to find all the positive instances correctly, the higher TPR/recall is favored. On the other hand, FPR is favored if the interest is to find all the negative instances correctly.

There is no single criteria to determine the optimal threshold of the classifier. The optimal threshold would be chosen heavily based on domain knowledge and application. Take cancer problem for example, the model is to predict if the patient has cancer or not. we want catch as many patience with cancer as possible even if some patiences are false identified. Also, other measures, such as F1-measure, are also used to find the optimal threshold. F1-measure is able to manage the trade off between precision and recall rate to a balanced threshold.

![threshold animation](https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/classification_threshold.gif)

<p align="center">
<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/prob_table.png" src="#" width="20%" /><img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/roc_plot_animation.gif" src="#" width="70%" />
</p>



# Mathematical/Statistical Interpretation of AUC

In the [probabilistic perspective](https://www.alexejgossmann.com/auc), AUC is the probability of the score of a randomly chosen class + is higher than the score of randomly chosen class -.
$$AUC = P(f(x+)>f(x-)|class(x+)=1, class(x-)=0)$$
In other words, it measures how well the probability ranked based on their true classes. Thus, it is a threshold-invariant and scale-invariant metrics. Only the sequence matters in the predicted probabilities. Based on this property, model with higher AUC indicates better discrimination of two classes. However, the probabilities output from model with higher AUC didn't guarantee the well-calibrated probability. You could find more information in this link: [Safe Handling Instructions for Probabilistic Classification](https://www.youtube.com/watch?v=RXMu96RJj_s).


# Visualization

To illustrate how AUC affected by different level of sepearation/discrimination, there are two distributions of probability for positive class and negative class respectively. When the more overlap between two class increase, the harder to separate two classes. It would lead to the decrease of AUC - random separation. Interestedly, it means that the classifer is a good classifier with flipped prediction if the roc curves lying in the right-bottom corner.

<p align="center">
<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/prob_dist_animation.gif" src="#" width="60%" />
</p>

<img src="https://github.com/sinyi-chou/sinyi-chou.github.io/blob/master/images/classification/multiple_ROCs_plot.png" src="#" width="60%" />


# Summary
1. AUC is a threshold-free metrics to measure the performance of binary classifier.

2. auc can only be used in two class -> if you have multiple class, one to all AUC would be an option/ micro macro  

3. AUC would be a good metric for you if you care the rank of output probabilities and not the absolute probability.

3. Even if auc is powerful, auc is not a cure-all. not good for well-calibrated
and heavily imbalanced class distribution
care equally about

4. AUC provided the equally weight between positive and negative class.


### Reference
1. [An introduction to ROC analysis](https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X)
2. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
3. http://www.medicalbiostatistics.com/roccurve.pdf
https://www.dataschool.io/roc-curves-and-auc-explained/
https://www.alexejgossmann.com/auc
. For a binary test and a binary disease state, the following table summarizes the possible errors that one can make using the test as a prediction of a disease state measured by some gold standard


If the test is continuous, say M, then a test positive is defined as M>c. Now we consider measures of accuracy as functions of c, i.e.

TPF(c)=Pr{M>c|D=1}
FPF(c)=Pr{M>c|D=0}

The ROC curve is a plot of FPF(c) versus TPF(c).

Take credit card risk problem for example, the model is to predict if the credit card application would approve or not. we want to be very sure about the prediction or it may lead to customer dissatisfaction.


. It measures how well predictions are ranked, rather than their absolute values. So, for example, if you as a marketer want to find a list of users who will respond to a marketing campaign. AUC is a good metric to use since the predictions ranked by probability is the order in which you will create a list of users to send the marketing campaign.
