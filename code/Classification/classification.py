### Author: Enzo Battistella
### Date: 3/16/2022
####################################
### Keywords: Classification; Models Tuning; Cross-validation;
### Description: <Leverage a learned higher-order distance to perform a prediction on a K-NN basis>
### Input: data, ground truth, learned distance
### Ouput: predictions, assessment of the classification, distance matrix
###################################

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, balanced_accuracy_score, f1_score, \
    confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, ShuffleSplit
from scipy.stats import truncnorm, expon, randint, uniform
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
import pickle, os
from math import floor


# Identify the best model using a rule defined with a function (as abs or the identity), metrics (accuracy, precision, recall...)
# and a given threshold.
# We compare results on validation and train for each metric.
# We recommend using balanced accuracy and precision with a threshold of 0.05 and the abs function.
def rule_model_selection(results, function, metric_constraint, metric_optim, threshold):
    # We sum the validation performance of the metrics we want to maximize
    optim = lambda f: sum([float(f[idx]) for idx in metric_optim])
    # We ensure that the function applied to the difference is positive and not too important to prevent overfitting
    constraint = lambda f: [0 <= function(float(f[metric_idx]) - float(f[metric_idx + 2])) < threshold for
                            metric_idx in metric_constraint]
    # The best model is the one maximizing optim and abiding the constraints.
    scores = [optim(l) if all(constraint(l)) else 0 for l in results]
    idx = np.argmax(scores)
    return idx, scores[idx]


# Define scorer with the wanted evaluation metrics, weighted versions of the metrics are used here to enable
# multiclass tasks
def default_scorer():
    balanced_accuracy = make_scorer(balanced_accuracy_score)
    weighted_precision = make_scorer(precision_score, average='weighted', zero_division=0)
    weighted_recall = make_scorer(recall_score, average='weighted', zero_division=0)
    weighted_f1 = make_scorer(f1_score, average='weighted', zero_division=0)
    # Dictionnary of metrics to use for the tuning
    score_dict = {"balanced_accuracy": balanced_accuracy, "precision": weighted_precision, "recall": weighted_recall,
                  "f1": weighted_f1}
    # Format the score metrics output
    scores = [['mean_train_' + i, 'std_train_' + i, 'mean_test_' + i, 'std_test_' + i] for i in
              list(score_dict.keys())]
    scores = ['params'] + [score for score_list in scores for score in score_list]
    return score_dict, scores


# Tune the model on a given labelled training set according to a chosen evaluation metric
def models_tuning(x_train, y_train, criterion="balanced_accuracy", n_iter=200, n_jobs=1, cv_num=10, name="./", seed=0):
    score_dict, scores = default_scorer()
    # Definition of the classifiers names, their respectives calling functions and parameters' ranges for the tuning
    names = ['Log_Reg', 'Stoch_Grad',
             'SVM_lin', 'SVM_poly', 'SVM_rbf', 'SVM_sig',
             'K_Neighbors', 'Gaussian_Proc', 'Gaussian_NB',
             'Multi_Gaussian_NB', 'Dec_Tree',
             'Ada', 'Random_Forest', 'Bagging', 'Grad_Boost']
    clfs = [LogisticRegression, SGDClassifier, SVC, SVC, SVC, SVC,
            KNeighborsClassifier, GaussianProcessClassifier, GaussianNB, MultinomialNB,
            DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier,
            BaggingClassifier, GradientBoostingClassifier]
    params = [
        {
            'Log_Reg__penalty': ['l1', 'l2', 'elasticnet'],
            'Log_Reg__C': uniform(0, 50),
            'Log_Reg__class_weight': ['balanced'],
            'Log_Reg__solver': ['saga'],
            'Log_Reg__max_iter': [10000],
            'Log_Reg__l1_ratio': [0.5],
            'Log_Reg__random_state': [None]
        },
        {
            'Stoch_Grad__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                                                  'perceptron', 'huber', 'epsilon_insensitive',
                                                  'squared_epsilon_insensitive'],
            'Stoch_Grad__penalty': ['l2', 'l1', 'elasticnet'],
            'Stoch_Grad__alpha': uniform(0, 1),
            'Stoch_Grad__l1_ratio': uniform(0, 1),
            'Stoch_Grad__class_weight': ['balanced'],
            'Stoch_Grad__random_state': [None]
        },
        {
            'SVM_lin__kernel': ['linear'],
            'SVM_lin__C': uniform(0, 50),
            'SVM_lin__class_weight': ['balanced'],
            'SVM_lin__random_state': [None],
            'SVM_lin__probability': [True]
        },
        {
            'SVM_poly__kernel': ['poly'],
            'SVM_poly__C': uniform(0, 50),
            'SVM_poly__coef0': uniform(0, 10),
            'SVM_poly__class_weight': ['balanced'],
            'SVM_poly__degree': randint(1, 8),
            'SVM_poly__random_state': [None],
            'SVM_poly__probability': [True]
        },
        {
            'SVM_rbf__kernel': ['rbf'],
            'SVM_rbf__C': uniform(0, 50),
            'SVM_rbf__class_weight': ['balanced'],
            'SVM_rbf__random_state': [None],
            'SVM_rbf__probability': [True]
        },
        {
            'SVM_sig__kernel': ['sigmoid'],
            'SVM_sig__C': uniform(0, 50),
            'SVM_sig__coef0': uniform(0, 10),
            'SVM_sig__class_weight': ['balanced'],
            'SVM_sig__random_state': [None],
            'SVM_sig__probability': [True]
        },
        {
            'K_Neighbors__n_neighbors': randint(2, 50),
            'K_Neighbors__weights': ['uniform', 'distance']
        },
        {
            'Gaussian_Proc__n_restarts_optimizer': randint(0, 10),
            'Gaussian_Proc__random_state': [None]
        },
        {
            'Gaussian_NB__var_smoothing': truncnorm(a=0, b=1, loc=1e-9, scale=1e-3)
        },
        {
            'Multi_Gaussian_NB__alpha': uniform(0.01, 10)
        },
        {
            'Dec_Tree__class_weight': ['balanced'],
            'Dec_Tree__max_depth': randint(2, 30),
            'Dec_Tree__min_samples_split': randint(2, 10),
            'Dec_Tree__random_state': [None]
        },
        {
            'Ada__n_estimators': randint(2, 60),
            'Ada__learning_rate': uniform(0, 1),
            'Ada__random_state': [None]
        },
        {
            'Random_Forest__n_estimators': randint(2, 100),
            'Random_Forest__max_depth': randint(2, 10),
            'Random_Forest__random_state': [None]
        },
        {
            'Bagging__n_estimators': randint(2, 60),
            'Bagging__random_state': [None]
        },
        {
            'Grad_Boost__n_estimators': randint(2, 60),
            'Grad_Boost__learning_rate': uniform(0, 1),
            'Grad_Boost__max_depth': randint(2, 10),
            'Grad_Boost__random_state': [None]
        }
    ]
    pips = []
    results = []
    best_models = []
    cv = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=seed)
    y_cur = y_train.values.ravel()
    # Create the pipelines with the classification techniques, data normalization (here minmax but standard normalization can also be used)
    # is applied for every data split to prevent data leakage from the validation or the test
    for i in range(len(names)):
        pips.append(Pipeline([('scaler', MinMaxScaler()), (names[i], clfs[i]())]))

    for i in range(len(pips)):
        print(i, names[i])
        # The models and their results are saved to avoid having to recompute them
        # Saves computation cost for time consuming tasks at the expense of disk memory
        if not os.path.exists(name + "_" + names[i] + "_results.npy"):
            gs = RandomizedSearchCV(pips[i], params[i], verbose=0, refit=criterion, n_iter=n_iter,
                                    random_state=seed,
                                    scoring=score_dict,
                                    n_jobs=n_jobs, cv=cv, return_train_score=True)
            if isinstance(x_train, pd.Series):
                x_train = x_train.values.reshape((-1, 1))
            gs = gs.fit(x_train, y_cur)
            # Formate the results
            results.append([names[i]])
            for score in scores:
                results[-1].append(str(gs.cv_results_[score][gs.best_index_]).replace(",", ";"))
            # Get and save the model maximizing the chosen metric corresponding to criterion
            model = gs.best_estimator_
            pickle.dump(model, open(name + "_" + names[i] + ".sav", "wb"))
            np.save(name + "_" + names[i] + "_results", results[-1])
        else:
            model = pickle.load(open(name + "_" + names[i] + ".sav", 'rb'))
            results.append(list(np.load(name + "_" + names[i] + "_results.npy", allow_pickle=True)))
        best_models.append(model)
    return (results, best_models)


# Compute cross-validation performance for a set of already tuned models
def cross_val(x_train, y_train, models, names, select_k, criterion, select_th=0.1, cv_num=10, seed=82):
    score_dict, scores = default_scorer()
    pips = []
    results = []
    cv = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=seed)
    y_cur = y_train.values.ravel()
    # Use of a minmax scaler in case the data is not normalized and the model is not a pipeline including normalization
    # In case there is already a minmax scaler in the model or the data has already been minmax normalized it is ok
    # BUT, to be removed if a standard normalization has been performed or is in the model
    for i in range(len(names)):
        pips.append(Pipeline([('scaler', MinMaxScaler()), (names[i], models[i])]))

    for i in range(len(pips)):
        print(names[i])
        if isinstance(x_train, pd.Series):
            x_train = x_train.values.reshape((-1, 1))
        aux_res = cross_validate(pips[i], x_train, y_cur, verbose=0, scoring=score_dict, cv=cv,
                                 return_train_score=True)
        # Recover the names of the estimators when possible
        try:
            estimators = list(models[i].named_estimators.keys())
        except:
            estimators = [""]
        # Add the name and the parameters of the model at the beginning of the results
        results.append([names[i] + ":" + ";".join(
            [j for j in estimators]),
                        ";".join(["k=" + str(select_k), "metric=" + str(criterion), "th=" + str(select_th)])])
        # Keep only the actual score
        for score in scores:
            if "mean" in score:
                results[-1].append(str(aux_res[score.split("mean_")[-1]].mean()))
            elif "std" in score:
                results[-1].append(str(aux_res[score.split("std_")[-1]].std()))

        if "mean_train_AUC" not in scores:
            results[-1] += ["0", "0", "0", "0"]
    return results


# Retrain a list of models on the whole training set and assesss their performance on training and test
def test_results(models, x_train, y_train, x_test, y_test):
    labels = list(np.unique(y_train))
    if isinstance(x_train, pd.Series):
        x_train = x_train.values.reshape((-1, 1))
        x_test = x_test.values.reshape((-1, 1))
    # Use of a minmax scaler in case the data is not normalized and the model is not a pipeline including normalization
    # In case there is already a minmax scaler in the model or the data has already been minmax normalized it is ok
    # BUT, to be removed if a standard normalization has been performed or is in the model
    # The scaler has to be trained on x_train only to avoid data leakage
    scaler = MinMaxScaler().fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)
    y_cur = y_train.values.ravel()
    y_cur_test = y_test.values.ravel()
    score_dict, _ = default_scorer()

    results = []
    confusion_matrices = []
    # train each model and assess its performance
    for model in models:
        model.fit(x_train_norm, y_cur)
        y_pred_train = model.predict(x_train_norm)
        y_pred_test = model.predict(x_test_norm)
        results.append([sub for _, score in score_dict.items() for sub in [score(model, x_train_norm, y_cur),
                                                                           score(model, x_test_norm, y_cur_test)]])
        confusion_matrices.append([confusion_matrix(y_train, y_pred_train, labels=labels),
                                   confusion_matrix(y_test, y_pred_test, labels=labels)])
    return results, confusion_matrices


# Return the results for a given prediction, labels are the names of the classes and are used for the confusion matrix
def assessment(y_true, y_pred):
    score_dict, _ = default_scorer()
    results = [score(y_true.values.ravel(), y_pred) for _, score in score_dict.items()]

    return results


# Save the results in a csv file
# step: 2 for cross-validation results to account for the average and standard deviation and 1 for test
# offset: number of model details to write (in the params list of the main file)
def performance_saving(f, results, best_model_idx, step, offset):
    counter = 0
    for res in results:
        # Write model name and parameters
        f.write(",".join([str(res[idx]) for idx in range(offset)]) + ",")
        # Write the results
        for i in range(offset, len(res), step):
            if step == 2:
                f.write(str(res[i]) + "+/-" + str(res[i + 1]) + ",")
            else:
                f.write(str(res[i]) + ",")
        # Mark the model which get the best performance according to the chosen criteria
        if counter == best_model_idx:
            f.write("****,")
        f.write("\n")
        counter += 1


# Save the confusion matrices in a csv file
# labels: names of the classes
def confusion_matrix_saving(f, confusion, params, best_model_idx, labels):
    f.write("," * floor(len(confusion[0]) / 2) + "Train" +
            "," * floor(len(confusion[0]) / 2) + ",,,,,Test\n")
    counter = 0
    for aux in confusion:
        f.write(params[counter] + "\n")
        # Write Train and Test confusion matrices side by side.
        f.write("," * ((len(params[counter].split(",")) - 1) + floor(len(labels) / 2)) + "Predicted Labels" +
                "," * ((len(params[counter].split(",")) - 1) + 3 + len(labels)) + "Predicted Labels" + "\n")
        f.write(
            "," * (len(params[counter].split(","))) + ",".join([str(i) for i in labels]) +
            "," * (len(params[counter].split(",")) + 3)
            + ",".join([str(i) for i in labels]) + "\n")
        for j in range(np.shape(aux[0])[1]):
            if j == floor(len(labels) / 2):
                f.write("Actual Labels,")
            else:
                f.write(",")
            f.write(str(labels[j]) + ",")
            for i in range(np.shape(aux[0])[0]):
                f.write(str(aux[0][i][j]) + ",")
            if j == floor(len(labels) / 2):
                f.write(",,Actual Labels,")
            else:
                f.write("," * 3)
            f.write(str(labels[j]) + ",")
            for i in range(np.shape(aux[1])[0]):
                f.write(str(aux[1][i][j]) + ",")
            f.write("\n")
        if counter == best_model_idx:
            f.write("****")
        f.write("\n")
        counter += 1
    f.write("\n")
