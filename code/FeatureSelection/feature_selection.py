### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Feature Selection; Cross-validation;
### Description: <Implements a cross-validation framework with various common feature selection techniques>
### Input: data, ground truth, number of iterations of the cross-validation
### Ouput: matrix of co-selection and of co-importance
###################################

import os.path

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import truncnorm, expon, randint, uniform
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, mutual_info_classif, f_classif, chi2, \
    VarianceThreshold
import numpy as np
import pandas as pd
import pickle


# Combine the co-selection matrices of the different feature selection techniques
def merge(cooc, idx_map, single_clf=-1):
    if single_clf != -1:
        cooc = [cooc[single_clf]]
        idx_map = [idx_map[single_clf]]
    num_genes = [i for i in idx_map[0].keys()]
    for i in range(1, len(idx_map)):
        for key in idx_map[i].keys():
            if key not in num_genes:
                num_genes.append(key)
    # sorted initial positions
    sorted_keys = sorted(num_genes)
    num_genes = len(num_genes)
    merged_map = {}
    merged = np.zeros((num_genes, num_genes))
    for key1 in range(num_genes):
        for key2 in range(key1, num_genes):
            if key2 not in merged_map.keys():
                merged_map[key2] = sorted_keys[key2]
            for id in range(len(cooc)):
                # merged associate initial pos to pos in merged matrix
                if sorted_keys[key1] in idx_map[id].keys() and sorted_keys[key2] in idx_map[id].keys():
                    merged[key1, key2] += cooc[id][idx_map[id][sorted_keys[key1]], idx_map[id][sorted_keys[key2]]]
                    merged[key2, key1] = merged[key1][key2]
    return merged, merged_map


# Performs a pre-filtering of the data to keep only the most correlated variables
def pre_filter_anova(x_train, y_train, threshold_stat, name):
    v = VarianceThreshold()
    sfm = SelectPercentile(score_func=f_classif, percentile=threshold_stat)
    x_cur = MinMaxScaler().fit_transform(v.fit_transform(x_train))
    # associate novel feature position idx to the initial position
    initial_map = {}
    counter = 0
    for idx in range(np.shape(x_cur)[1]):
        while idx + counter < len(v.get_support()) and not v.get_support()[idx + counter]:
            counter += 1
        initial_map[idx] = idx + counter
    y_cur = y_train.values.ravel()
    if not os.path.exists(name + "anova.npy"):
        sfm.fit(x_cur, y_cur)
        support = sfm.get_support(indices=True)
        # transpose the support into the initial space
        support = [initial_map[s] for s in support]
        np.save(name + "anova", support)
    else:
        support = list(np.load(name + "anova.npy"))
    return support


# Tune and Train the feature selection techniques before performing the selection
def feature_selector(x_train, y_train, criterion="balanced_accuracy", n_iter=50, cv_num=10, clf_only=False, n_jobs=1,
                     thresholds=["1*mean", "5*median", "2*mean", "0.5*mean", "3*mean"], threshold_stat=.25,
                     name="test", seed=82):
    features_selectors = []
    pips = []
    # any classifier can be used
    clfs = [DecisionTreeClassifier, SVC, GradientBoostingClassifier, AdaBoostClassifier]
    cv = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=seed)
    y_cur = y_train.values.ravel()
    # Ranges in which the random search is applied to find the best parameters
    # Can be replaced by the actual values of the parameters in case of manual tuning
    params = [
        {
            'Decision_Tree__class_weight': ['balanced'],
            'Decision_Tree__max_depth': randint(2, 30),
            'Decision_Tree__min_samples_split': randint(2, 10),
            'Decision_Tree__random_state': [None]
        },
        {
            'SVM_linear__kernel': ['linear'],
            'SVM_linear__C': uniform(0, 50),
            'SVM_linear__class_weight': ['balanced'],
            'SVM_linear__random_state': [None]
        },
        {
            'Gradient_Boosting__n_estimators': randint(2, 60),
            'Gradient_Boosting__learning_rate': uniform(0, 1),
            'Gradient_Boosting__max_depth': randint(2, 10),
            'Gradient_Boosting__random_state': [None]
        },
        {
            'AdaBoost__n_estimators': randint(2, 60),
            'AdaBoost__learning_rate': uniform(0, 1),
            'AdaBoost__random_state': [None]
        }
    ]
    params_names = [str(th).replace("*", "") for th in thresholds]
    names = ['Decision_Tree', 'SVM_linear', 'Gradient_Boosting', 'AdaBoost']

    # Create the pipelines with the feature selection techniques, variance threshold and data normalization (here
    # minmax but standard normalization can also be used) are applied for every data split to prevent data leakage
    # from the validation or the test
    for i in range(len(names)):
        pips.append(Pipeline(steps=[('Variance Threshold', VarianceThreshold()),
                                    ('scaler', MinMaxScaler()), (names[i], clfs[i]())]))

    for i in range(len(pips)):
        if not os.path.exists(name + "_" + names[i] + ".sav"):
            # Tune the feature selectors, the tuning should not be overfitting to much to not miss important features
            # the assessment metric used is balanced accuracy but it can be modified to consider the same combination
            # of metrics as in the classification part
            print("tuning", names[i])
            gs = RandomizedSearchCV(pips[i], params[i], verbose=0, refit=criterion, n_iter=n_iter,
                                    random_state=seed, n_jobs=n_jobs,
                                    scoring=criterion, cv=cv)
            gs = gs.fit(x_train, y_cur)
            # The best parameters are selected
            model = gs.best_estimator_[-1]

            pickle.dump(model, open(name + "_" + names[i] + ".sav", "wb"))
        else:
            model = pickle.load(open(name + "_" + names[i] + ".sav", 'rb'))
        features_selectors.append(SelectFromModel(model, threshold=thresholds[i]))
    # Add the lasso method is required (as a regression technique, it cannot be tuned using balanced accuracy)
    features_selectors.append(
        SelectFromModel(LassoCV(max_iter=1000, eps=0.005, n_alphas=400), threshold=thresholds[-1]))
    names_complete = list(names) + ["Lasso"]
    # Eventually add feature selection techniques based on statistical tests
    if not clf_only:
        aux_selectors = [
            SelectPercentile(score_func=chi2, percentile=threshold_stat),
            SelectPercentile(score_func=f_classif, percentile=threshold_stat),
            SelectPercentile(score_func=mutual_info_classif, percentile=threshold_stat)
        ]
        aux_names = ["chi2", "f_classif", "mutual_info"]
        features_selectors += aux_selectors
        names_complete += aux_names
        params_names += [str(threshold_stat) for _ in range(len(names_complete))]
    cooc = [np.zeros((0, 0)) for _ in features_selectors]
    idx_map = [{} for _ in features_selectors]
    local_map = [{} for _ in features_selectors]
    weights = np.zeros(np.shape(x_train)[1])
    cooc_weights = np.zeros((np.shape(x_train)[1], np.shape(x_train)[1]))
    idx_split = 0
    number_div = 0
    # Perform the feature selection on a cv_num-fold cross-validation split
    for train, _ in cv.split(x_train, y_cur):
        v = VarianceThreshold()
        x_cur = MinMaxScaler().fit_transform(v.fit_transform(x_train.iloc[train, :]))
        # Be careful if the minmax scaler is replaced by another technique, chi2 need positive values to be computed
        initial_map = {}
        counter = 0
        # dictionnary associating novel feature position idx to the initial position
        for idx in range(np.shape(x_cur)[1]):
            while idx + counter < len(v.get_support()) and not v.get_support()[idx + counter]:
                counter += 1
            initial_map[idx] = idx + counter
        y_cur_train = y_cur[train]
        for i in range(len(features_selectors)):
            print(names_complete[i])
            aux = x_cur
            if not os.path.exists(
                    name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split) + ".npy"):
                sfm = features_selectors[i]
                sfm.fit(aux, y_cur_train)
                support = sfm.get_support(indices=True)
                # Transpose the support into the initial space
                support = [initial_map[s] for s in support]
                # Get the feature importance / coeffs from the feature selectors enabling it
                if names_complete[i] not in ["chi2", "f_classif", "mutual_info", "Lasso"]:
                    if names_complete[i] != "SVM_linear":
                        aux_weights = sfm.estimator_.feature_importances_
                    else:
                        aux_weights = np.sum(sfm.estimator_.coef_, axis=0)

                    if np.max(aux_weights, axis=0) != np.min(aux_weights, axis=0):
                        aux_weights = (aux_weights - np.min(aux_weights, axis=0)) / (
                                np.max(aux_weights, axis=0) - np.min(aux_weights, axis=0))
                    else:
                        print(names_complete[i], "abnormal weights", aux_weights)
                else:
                    aux_weights = np.array([0. for _ in support])
                np.save(name + "_" + names_complete[i] + "_" + params_names[i] + "support" + str(idx_split), support)
                np.save(name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split),
                    aux_weights)
            else:
                support = list(np.load(
                    name + "_" + names_complete[i] + "_" + params_names[i] + "support" + str(idx_split) + ".npy"))
                aux_weights = np.load(
                    name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split) + ".npy")
            idx_split += 1
            for id_feature in support:
                if id_feature not in idx_map[i].keys():
                    # idx_map stores for each selector i the position of each feature in the cooc matrix
                    idx_map[i][id_feature] = len(cooc[i])
                    local_map[i][len(cooc[i])] = id_feature
                    cooc[i] = np.append(cooc[i], np.zeros((1, np.shape(cooc[i])[1])), axis=0)
                    cooc[i] = np.append(cooc[i], np.zeros((np.shape(cooc[i])[0], 1)), axis=1)
            aux_support = np.array([id_feature for id_feature in range(len(cooc[i]))
                                    if local_map[i][id_feature] in support])
            # Build the co-importance matrix by adding the importance weights of the features
            if not np.isnan(aux_weights).any():
                transp_weights = np.array([0. for _ in weights])
                for idx in range(len(aux_weights)):
                    transp_weights[initial_map[idx]] = aux_weights[idx]
                weights = weights + transp_weights
                non_z = [i for i in range(len(transp_weights)) if transp_weights[i] > 0]
                # We consider as co-importance of feature i and j the minimum importance weight assigned to i or j
                for idx_1 in non_z:
                    for idx_2 in non_z:
                        cooc_weights[idx_1, idx_2] += min(transp_weights[idx_1], transp_weights[idx_2])
                number_div += 1
            if len(support) > 0:
                cooc[i][aux_support[..., None], aux_support] += 1
    return cooc, idx_map, len(features_selectors) * cv_num, weights / number_div, cooc_weights / number_div


# Bootstrap version of the previous feature selection, each selection is performed over a subset of features
def feature_selector_bootstrap(x_train, y_train, criterion="balanced_accuracy", n_iter=50, cv_num=10, boot_num=10, test_size=0.4, clf_only=False,
                               n_jobs=1, thresholds=["1*mean", "5*median", "2*mean", "0.5*mean", "3*mean"],
                               threshold_stat=.25,
                               name="test", seed=82):
    features_selectors = []
    pips = []
    clfs = [DecisionTreeClassifier, SVC, GradientBoostingClassifier, AdaBoostClassifier]
    boosting = ShuffleSplit(n_splits=boot_num, test_size=test_size, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=seed)
    params = [
        {
            'Decision_Tree__class_weight': ['balanced'],
            'Decision_Tree__max_depth': randint(2, 30),
            'Decision_Tree__min_samples_split': randint(2, 30),
            'Decision_Tree__random_state': [None]
        },
        {
            'SVM_linear__kernel': ['linear'],
            'SVM_linear__C': expon(scale=2),
            'SVM_linear__class_weight': ['balanced'],
            'SVM_linear__random_state': [None]
        },
        {
            'Gradient_Boosting__n_estimators': randint(2, 100),
            'Gradient_Boosting__learning_rate': truncnorm(a=0, b=10, loc=0, scale=2.),
            'Gradient_Boosting__max_depth': randint(2, 10),
            'Gradient_Boosting__random_state': [None]
        },
        {
            'AdaBoost__n_estimators': randint(2, 100),
            'AdaBoost__learning_rate': truncnorm(a=0, b=10, loc=0, scale=2.),
            'AdaBoost__random_state': [None]
        }
    ]
    params_names = [str(th).replace("*", "") for th in thresholds]
    names = ['Decision_Tree', 'SVM_linear', 'Gradient_Boosting', 'AdaBoost']
    print("pipeline creation")
    y_cur = y_train.values.ravel()
    for i in range(len(names)):
        pips.append(Pipeline(steps=[('Variance Threshold', VarianceThreshold()),
                                    ('scaler', MinMaxScaler()), (names[i], clfs[i]())]))

    for i in range(len(pips)):
        if not os.path.exists(name + "_" + names[i] + ".sav"):
            print("tuning", names[i])
            gs = RandomizedSearchCV(pips[i], params[i], verbose=0, refit=criterion, n_iter=n_iter,
                                    random_state=seed, n_jobs=n_jobs,
                                    scoring=criterion, cv=cv)
            gs = gs.fit(x_train, y_cur)
            # We only take the classifier without the scaler
            model = gs.best_estimator_[-1]
            """if names[i] != 'SVM_linear':
                print(names[i], model.feature_importances_, np.sum([i != 0 for i in model.feature_importances_]),
                      np.mean(model.feature_importances_))
            else:
                print(names[i], model.coef_, np.sum([i != 0 for i in model.coef_]),
                      np.mean(model.coef_))"""
            pickle.dump(model, open(name + "_" + names[i] + ".sav", "wb"))
        else:
            model = pickle.load(open(name + "_" + names[i] + ".sav", 'rb'))
        features_selectors.append(SelectFromModel(model, threshold=thresholds[i]))
    features_selectors.append(
        SelectFromModel(LassoCV(max_iter=1000, eps=0.005, n_alphas=400), threshold=thresholds[-1]))
    names_complete = list(names) + ["Lasso"]
    if not clf_only:
        """features_selectors += [
            SelectPercentile(score_func=chi2, percentile=threshold_stat),
            SelectPercentile(score_func=f_classif, percentile=threshold_stat),
            SelectPercentile(score_func=mutual_info_classif, percentile=threshold_stat)
        ]"""
        features_selectors += [
            SelectPercentile(score_func=f_classif, percentile=threshold_stat),
            SelectPercentile(score_func=mutual_info_classif, percentile=threshold_stat)
        ]
        # names_complete += ["chi2", "f_classif", "mutual_info"]
        names_complete += ["f_classif", "mutual_info"]
        params_names += [str(threshold_stat) for _ in range(len(names_complete))]
    cooc = [np.zeros((0, 0)) for _ in features_selectors]
    idx_map = [{} for _ in features_selectors]
    local_map = [{} for _ in features_selectors]
    weights = np.array([0 for _ in range(np.shape(x_train)[1])])
    print("selection")
    idx_split = 0
    number_div = 0
    for train, _ in cv.split(x_train, y_train):
        y_cur = y_train.iloc[train].values.ravel()
        v = VarianceThreshold()
        x_cur = MinMaxScaler().fit_transform(v.fit_transform(x_train.iloc[train, :]))
        initial_map = {}
        counter = 0
        # associate novel feature position idx to the initial position
        for idx in range(np.shape(x_cur)[1]):
            if not v.get_support()[idx + counter]:
                counter += 1
            initial_map[idx] = idx + counter
        for _, features_idx in boosting.split(x_cur):
            for i in range(len(features_selectors)):
                aux = x_cur[:, features_idx]
                # chi2 needs positive values
                """if i == len(names) + 1:
                    aux = (x_cur[:, features_idx] - x_cur[:, features_idx].min()) /\
                          (x_cur[:, features_idx].max() - x_cur[:, features_idx].min())"""
                if not os.path.exists(
                        name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split) + ".npy"):
                    sfm = features_selectors[i]
                    sfm.fit(aux, y_cur)
                    support = sfm.get_support(indices=True)
                    # transpose the support into the initial space
                    support = [initial_map[features_idx[s]] for s in support]
                    if names_complete[i] not in ["f_classif", "mutual_info"]:
                        if names_complete[i] != 'SVM_linear' and names_complete[i] != "Lasso":
                            aux_weights = sfm.estimator_.feature_importances_
                        else:
                            aux_weights = sfm.estimator_.coef_
                    else:
                        aux_weights = np.array([0 for _ in range(aux.shape[1])])
                    aux_weights = (aux_weights - min(aux_weights)) / (max(aux_weights) - min(aux_weights))
                    np.save(name + "_" + names_complete[i] + "_" + params_names[i] + "support" + str(idx_split),
                            support)
                    np.save(name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split),
                            aux_weights)
                else:
                    support = list(np.load(
                        name + "_" + names_complete[i] + "_" + params_names[i] + "support" + str(idx_split) + ".npy",
                        allow_pickle=True))
                    aux_weights = np.load(
                        name + "_" + names_complete[i] + "_" + params_names[i] + "weights" + str(idx_split) + ".npy",
                        allow_pickle=True)
                idx_split += 1
                for id_feature in support:
                    if id_feature not in idx_map[i].keys():
                        # idx_map stores for each selector i the position of each feature in the cooc matrix
                        idx_map[i][id_feature] = len(cooc[i])
                        local_map[i][len(cooc[i])] = id_feature
                        cooc[i] = np.append(cooc[i], np.zeros((1, np.shape(cooc[i])[1])), axis=0)
                        cooc[i] = np.append(cooc[i], np.zeros((np.shape(cooc[i])[0], 1)), axis=1)
                aux_support = np.array([id_feature for id_feature in range(len(cooc[i]))
                                        if local_map[i][id_feature] in support])
                """print("cooc shape", [np.shape(aux) for aux in cooc])"""
                if not np.isnan(aux_weights).any():
                    transp_weights = np.array([0 for _ in weights])
                    for idx in range(len(aux_weights)):
                        transp_weights[initial_map[features_idx[idx]]] = aux_weights[idx]
                    weights = weights + transp_weights
                    number_div += 1
                if len(support) > 0:
                    cooc[i][aux_support[..., None], aux_support] += 1
    return cooc, idx_map, len(features_selectors) * cv_num, weights / number_div
