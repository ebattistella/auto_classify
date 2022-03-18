### Author: Enzo Battistella
### Date: 3/17/2022
####################################
### Keywords: Classification; Ensemble;
### Description: <Define ensemble methods from a list of models with performance>
### Input: data, ground truth, learned distance
### Ouput: predictions, assessment of the classification, distance matrix
###################################

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, f1_score, confusion_matrix, \
    roc_auc_score
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from functools import cmp_to_key
import numpy as np

# Tool function to compare two models given their cross-validation performance
# metric: criterion used to evaluate the performance
# th: maximal difference between training and validation performance allowed
def compare(model1, model2, th, metric):
    model1_val = float(model1[1][metric + 2])
    model2_val = float(model2[1][metric + 2])
    model1_train = float(model1[1][metric])
    model2_train = float(model2[1][metric])
    if model1_val > model2_val and (0 < model1_train - model1_val < th or model2_train - model2_val > th):
        return 1
    elif model1_val < model2_val and (model1_train - model1_val > th or 0 < model2_train - model2_val < th):
        return -1
    elif model1_val == model2_val:
        return np.sign(model2_train - model1_train)
    else:
        return 0

# Return the k best models according to the ranking rule defined in compare
def select_best(results, th, metric, k):
    res_idx = [idx for idx in range(len(results)) if
               (0 < float(results[idx][metric]) - float(results[idx][metric + 2]) < th)]
    # If no estimator fulfill the criterion, we take arbitrarily the first one
    if len(res_idx) > 0:
        res = [results[idx] for idx in range(len(results)) if idx in res_idx]
        sorted_pairs = sorted(enumerate(res), key=cmp_to_key(lambda x, y: compare(x, y, th, metric)), reverse=True)
        return [res_idx[sorted_pairs[i][0]] for i in range(min(len(res), k))]
    else:
        return [0]

# Take the k best models according the ranking rule defined in compare to define a majority voting method
def ensemble_majority(models, results, th=0.3, metric=3, k=5):
    indexes = select_best(results, th, metric, k)
    selection = [(results[i][0], models[i]) for i in indexes]

    return VotingClassifier(selection)

# Take the k best models according the ranking rule defined in compare to define a stacking method
# max_depth: is the maximal depth of the decision tree used to ensemble the results of the models
def ensemble_stacking(models, results, th=0.3, metric=3, k=5, max_depth=2):
    indexes = select_best(results, th, metric, k)
    selection = [(results[i][0], models[i]) for i in indexes]
    return StackingClassifier(selection, final_estimator=DecisionTreeClassifier(max_depth=max_depth))

# Take the k best models according the ranking rule defined in compare to define a weighted majority voting method
# The weights correspond to the performance on validation of the considered evaluation metric
def ensemble_weighted_majority(models, results, th=0.3, metric=3, k=5):
    indexes = select_best(results, th, metric, k)
    selection = [(results[i][0], models[i]) for i in indexes]
    weights = [results[i][metric + 2] for i in indexes]
    aux = 'soft'
    try:
        for i in indexes:
            models[i].predict_proba(np.zeros(10))
    except:
        aux = 'hard'
    return VotingClassifier(selection, weights=weights)

