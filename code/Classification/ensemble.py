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
# Rule: allows to change the function considered to assess the training and validation discrepancy,
# Default function is identity, another natural function is absolute value.
def compare(model1, model2, constraint, metric_optim):
    # We sum the validation performance of the metrics we want to maximize
    optim = lambda f, step: sum([float(f[idx + step]) for idx in metric_optim])
    # Performances are formatted as [average metric train, std metric train, average metric validation, std metric validation]
    model1 = model1[1]
    model2 = model2[1]
    if optim(model1, 2) > optim(model2, 2) and (all(constraint(model1)) or not all(constraint(model2))):
        return 1
    elif optim(model1, 2) < optim(model2, 2) and (not all(constraint(model1)) or all(constraint(model2))):
        return -1
    elif optim(model1, 2) == optim(model2, 2):
        return np.sign(optim(model1, 0) - optim(model2, 0))
    else:
        return 0


# Return the k best models according to the ranking rule defined in compare
def select_best(results, function, metric_constraint, metric_optim, threshold, k):
    # We ensure that the function applied to the difference is positive and not too important to prevent overfitting
    constraint = lambda f: [0 <= function(float(f[metric_idx]) - float(f[metric_idx + 2])) < threshold for
                            metric_idx in metric_constraint]
    res_idx = [idx for idx in range(len(results)) if all(constraint(results[idx]))]
    # If no estimator fulfill the criterion, we take arbitrarily the first one
    if len(res_idx) > 0:
        res = [results[idx] for idx in range(len(results)) if idx in res_idx]
        sorted_pairs = sorted(enumerate(res),
                              key=cmp_to_key(lambda x, y: compare(x, y, constraint, metric_optim)),
                              reverse=True)
        return [res_idx[sorted_pairs[i][0]] for i in range(min(len(res), k))]
    else:
        return [0]


# Take the k best models according the ranking rule defined in compare to define a majority voting method
def ensemble_majority(models, results, selection=[lambda x: x, [2], [2], 0.05], k=5):
    function, metric_constraint, metric_optim, threshold = selection[0], selection[1], selection[2], selection[3]
    indexes = select_best(results, function, metric_constraint, metric_optim, threshold, k)
    selection = [(results[i][0], models[i]) for i in indexes]

    return VotingClassifier(selection)


# Take the k best models according the ranking rule defined in compare to define a stacking method
# max_depth: is the maximal depth of the decision tree used to ensemble the results of the models
def ensemble_stacking(models, results, selection=[lambda x: x, [2], [2], 0.05], k=5, max_depth=2):
    function, metric_constraint, metric_optim, threshold = selection[0], selection[1], selection[2], selection[3]
    indexes = select_best(results, function, metric_constraint, metric_optim, threshold, k)
    selection = [(results[i][0], models[i]) for i in indexes]
    return StackingClassifier(selection, final_estimator=DecisionTreeClassifier(max_depth=max_depth))


# Take the k best models according the ranking rule defined in compare to define a weighted majority voting method
# The weights correspond to the performance on validation of the considered evaluation metric
def ensemble_weighted_majority(models, results, selection=[lambda x: x, [2], [2], 0.05], k=5):
    function, metric_constraint, metric_optim, threshold = selection[0], selection[1], selection[2], selection[3]
    indexes = select_best(results, function, metric_constraint, metric_optim, threshold, k)
    selection = [(results[i][0], models[i]) for i in indexes]
    weights = [sum([float(results[i][metric + 2]) for metric in metric_optim]) for i in indexes]
    aux = 'soft'
    try:
        for i in indexes:
            models[i].predict_proba(np.zeros(10))
    except:
        aux = 'hard'
    return VotingClassifier(selection, weights=weights, voting=aux)
