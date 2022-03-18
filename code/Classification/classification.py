### Author: Enzo Battistella
### Date: 3/16/2022
####################################
### Keywords: Classification; Models Tuning; Cross-validation;
### Description: <Leverage a learned higher-order distance to perform a prediction on a K-NN basis>
### Input: data, ground truth, learned distance
### Ouput: predictions, assessment of the classification, distance matrix
###################################

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
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

# Tune the model on a given labelled training set according to a chosen evaluation metric
def models_tuning(x_train, y_train, criterion="balanced_accuracy", n_iter=200, n_jobs=1, cv_num=10, name="./", seed=0):
    # Define scorer with the wanted evaluation metrics, weighted versions of the metrics are used here to enable
    # multiclass tasks
    balanced_accuracy = make_scorer(balanced_accuracy_score)
    weighted_precision = make_scorer(precision_score, average='weighted', zero_division=0)
    weighted_recall = make_scorer(recall_score, average='weighted', zero_division=0)
    weighted_f1 = make_scorer(f1_score, average='weighted', zero_division=0)
    # Definition of the classifiers names, their respectives calling functions and parameters' ranges for the tuning
    names = ['Logistic_Regression', 'Stochastic_Gradient_Descent',
             'SVM_linear', 'SVM_polynomial', 'SVM_rbf', 'SVM_sigmoid',
             'K_Neighbors_Classifier', 'Gaussian_Process', 'Gaussian_Naive_Bayes',
             'Multinomial_Gaussian_Naive_Bayes', 'Decision_Tree',
             'AdaBoost', 'Random_Forest', 'Bagging_Classifier', 'Gradient_Boosting']
    clfs = [LogisticRegression, SGDClassifier, SVC, SVC, SVC, SVC,
            KNeighborsClassifier, GaussianProcessClassifier, GaussianNB, MultinomialNB,
            DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier,
            BaggingClassifier, GradientBoostingClassifier]
    params = [
        {
            'Logistic_Regression__penalty': ['l1', 'l2', 'elasticnet'],
            'Logistic_Regression__C': uniform(0, 50),
            'Logistic_Regression__class_weight': ['balanced'],
            'Logistic_Regression__solver': ['saga'],
            'Logistic_Regression__max_iter': [10000],
            'Logistic_Regression__l1_ratio': [0.5],
            'Logistic_Regression__random_state': [None]
        },
        {
            'Stochastic_Gradient_Descent__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                                                  'perceptron', 'huber', 'epsilon_insensitive',
                                                  'squared_epsilon_insensitive'],
            'Stochastic_Gradient_Descent__penalty': ['l2', 'l1', 'elasticnet'],
            'Stochastic_Gradient_Descent__alpha': uniform(0, 1),
            'Stochastic_Gradient_Descent__l1_ratio': uniform(0, 1),
            'Stochastic_Gradient_Descent__class_weight': ['balanced'],
            'Stochastic_Gradient_Descent__random_state': [None]
        },
        {
            'SVM_linear__kernel': ['linear'],
            'SVM_linear__C': uniform(0, 50),
            'SVM_linear__class_weight': ['balanced'],
            'SVM_linear__random_state': [None],
            'SVM_linear__probability':[True]
        },
        {
            'SVM_polynomial__kernel': ['poly'],
            'SVM_polynomial__C': uniform(0, 50),
            'SVM_polynomial__coef0': uniform(0, 10),
            'SVM_polynomial__class_weight': ['balanced'],
            'SVM_polynomial__degree': randint(1, 8),
            'SVM_polynomial__random_state': [None],
            'SVM_polynomial__probability':[True]
        },
        {
            'SVM_rbf__kernel': ['rbf'],
            'SVM_rbf__C': uniform(0, 50),
            'SVM_rbf__class_weight': ['balanced'],
            'SVM_rbf__random_state': [None],
            'SVM_rbf__probability':[True]
        },
        {
            'SVM_sigmoid__kernel': ['sigmoid'],
            'SVM_sigmoid__C': uniform(0, 50),
            'SVM_sigmoid__coef0': uniform(0, 10),
            'SVM_sigmoid__class_weight': ['balanced'],
            'SVM_sigmoid__random_state': [None],
            'SVM_sigmoid__probability':[True]
        },
        {
            'K_Neighbors_Classifier__n_neighbors': randint(2, 50),
            'K_Neighbors_Classifier__weights': ['uniform', 'distance']
        },
        {
            'Gaussian_Process__n_restarts_optimizer': randint(0, 10),
            'Gaussian_Process__random_state': [None]
        },
        {
            'Gaussian_Naive_Bayes__var_smoothing': truncnorm(a=0, b=1, loc=1e-9, scale=1e-3)
        },
        {
            'Multinomial_Gaussian_Naive_Bayes__alpha': uniform(0.01,10)
        },
        {
            'Decision_Tree__class_weight': ['balanced'],
            'Decision_Tree__max_depth': randint(2, 30),
            'Decision_Tree__min_samples_split': randint(2, 10),
            'Decision_Tree__random_state': [None]
        },
        {
            'AdaBoost__n_estimators': randint(2, 60),
            'AdaBoost__learning_rate': uniform(0, 1),
            'AdaBoost__random_state': [None]
        },
        {
            'Random_Forest__n_estimators': randint(2, 100),
            'Random_Forest__max_depth': randint(2, 10),
            'Random_Forest__random_state': [None]
        },
        {
            'Bagging_Classifier__n_estimators': randint(2, 60),
            'Bagging_Classifier__random_state': [None]
        },
        {
            'Gradient_Boosting__n_estimators': randint(2, 60),
            'Gradient_Boosting__learning_rate': uniform(0, 1),
            'Gradient_Boosting__max_depth': randint(2, 10),
            'Gradient_Boosting__random_state': [None]
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
    # Dictionnary of metrics to use for the tuning
    score_dict = {"balanced_accuracy": balanced_accuracy, "precision": weighted_precision, "recall": weighted_recall,
                  "f1": weighted_f1, "roc_auc": "roc_auc"}
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
            # Format the score metrics output and combine them in results
            scores = [['mean_train_' + i, 'std_train_' + i, 'mean_test_' + i, 'std_test_' + i] for i in
                      list(score_dict.keys())]
            scores = ['params'] + [score for score_list in scores for score in score_list]
            results.append([names[i]])
            for score in scores:
                results[-1].append(str(gs.cv_results_[score][gs.best_index_]).replace(",", ";"))
            # Get and save the model maximizing the chosen metric corresponding to criterion
            model = gs.best_estimator_
            pickle.dump(model, open(name + "_" + names[i] + ".sav", "wb"))
            np.save(name + "_" + names[i] + "_results", results)
        else:
            model = pickle.load(open(name + "_" + names[i] + ".sav", 'rb'))
            results.append(list(np.load(name + "_" + names[i] + "_results.npy", allow_pickle=True)))
        best_models.append(model)
    return (results, best_models)

# Compute cross-validation performance for a set of already tuned models,
def cross_val(x_train, y_train, models, names, select_k, criterion, select_th=0.1, cv_num=10, seed=82):
    # Define scorer with the wanted evaluation metrics, weighted versions of the metrics are used here to enable
    # multiclass tasks
    balanced_accuracy = make_scorer(balanced_accuracy_score)
    weighted_precision = make_scorer(precision_score, average='weighted', zero_division=0)
    weighted_recall = make_scorer(recall_score, average='weighted', zero_division=0)
    weighted_f1 = make_scorer(f1_score, average='weighted')
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
        # Test if the model implements the predict_proba function and if so compute the auc score
        try:
            pips[i].predict_proba(x_train)
            score_dict = {"BA": balanced_accuracy, "WP": weighted_precision, "WR": weighted_recall, "WF1": weighted_f1,
                          "AUC": 'roc_auc'}
            aux_res = cross_validate(pips[i], x_train, y_cur, verbose=0, scoring=score_dict, cv=cv,
                                     return_train_score=True)
            scores = [['mean_train_' + i, 'std_train_' + i, 'mean_test_' + i, 'std_test_' + i] for i in
                      list(score_dict.keys())]
            scores = [score for score_list in scores for score in score_list]
        except:
            score_dict = {"BA": balanced_accuracy, "WP": weighted_precision, "WR": weighted_recall, "WF1": weighted_f1}
            aux_res = cross_validate(pips[i], x_train, y_cur, verbose=0, scoring=score_dict, cv=cv,
                                     return_train_score=True, error_score='raise')
            scores = [['mean_train_' + i, 'std_train_' + i, 'mean_test_' + i, 'std_test_' + i] for i in
                      list(score_dict.keys())]
            scores = [score for score_list in scores for score in score_list]
            #add default values 0 for the auc
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
            else:
                results[-1].append(str(aux_res[score.split("std_")[-1]].std()))
        if "mean_train_AUC" not in scores:
            results[-1] += ["0", "0", "0", "0"]
    return (results)


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
    scores = [
        lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
        lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    ]
    results = []
    confusion_matrices = []
    #train each model and assess its performance
    for model in models:
        auc = [0, 0]
        model.fit(x_train_norm, y_cur)
        y_pred_train = model.predict(x_train_norm)
        y_pred_test = model.predict(x_test_norm)
        # Test if the model implements a predict_proba function and so if the AUC can be computed
        try:
            proba_train = model.predict_proba(x_train_norm)
            proba_test = model.predict_proba(x_test_norm)
            if len(labels) == 2:
                proba_train = proba_train[:, 1]
                proba_test = proba_test[:, 1]
                auc = [roc_auc_score(y_cur, proba_train),
                       roc_auc_score(y_cur_test, proba_test)]
        except:
            auc = [0, 0]
        results.append([sub for score in scores for sub in [score(y_cur, y_pred_train),
                                                            score(y_cur_test, y_pred_test)]]
                       + auc)
        confusion_matrices.append([confusion_matrix(y_train, y_pred_train, labels=labels),
                                   confusion_matrix(y_test, y_pred_test, labels=labels)])
    return results, confusion_matrices


# Return the results for a given prediction, labels are the names of the classes and are used for the confusion matrix
def assessment(y_true, y_pred):
    scores = [
        lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
        lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        lambda y_true, y_pred: confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    ]
    results = [score(y_true.values.ravel(), y_pred) for score in scores]

    return results

# Save the results in a csv file
# step: 2 for cross-validation results to account for the average and standard deviation and 1 for test
# offset: number of model details to write (in the params list of the main file)
def performance_saving(f, results, best_model_idx, step, offset):
    for res in results:
        print(res)
        # Write model name and parameters
        f.write(",".join([str(res[idx]) for idx in range(offset)]) + ",")
        # Write the results
        for i in range(offset, len(res), step):
            if step == 2:
                f.write(str(res[i]) + "+/-" + str(res[i + 1]) + ",")
            else:
                f.write(str(res[i]) + ",")
        # Mark the model which get the best performance according to the chosen criteria
        if res[1] == best_model_idx:
            f.write("****,")
        f.write("\n")

def confusion_matrix_saving(f, confusion, params, best_model_idx, labels):
    f.write("," * floor(len(confusion[0]) / 2) + "Train" +
            "," * floor(len(confusion[0]) / 2) + ",,,,,Test\n")
    counter = 0
    for aux in confusion:
        f.write(params[counter] + "\n")
        f.write("," * ((len(params[counter].split(","))-1) + floor(len(labels) / 2)) + "Predicted Labels" +
                "," * ((len(params[counter].split(","))-1) + 3 + len(labels)) + "Predicted Labels" + "\n")
        f.write(
            "," * (len(params[counter].split(","))) + ",".join([str(i) for i in labels]) +
            "," * (len(params[counter].split(","))+3)
            + ",".join([str(i) for i in labels]) + "\n")
        for j in range(np.shape(aux[0])[1]):
            if j == floor(len(labels) / 2):
                f.write("True Labels,")
            else:
                f.write(",")
            f.write(str(labels[j]) + ",")
            for i in range(np.shape(aux[0])[0]):
                f.write(str(aux[0][i][j]) + ",")
            if j == floor(len(labels) / 2):
                f.write(",,True Labels,")
            else:
                f.write("," * 3)
            f.write(str(labels[j]) + ",")
            for i in range(np.shape(aux[1])[0]):
                f.write(str(aux[1][i][j]) + ",")
            f.write("\n")
        if params[counter] == best_model_idx:
            f.write("****")
        f.write("\n")
        counter += 1
    f.write("\n")