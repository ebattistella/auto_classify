### Author: Enzo Battistella
### Date: 3/16/2022
####################################
### Keywords: Classification; Main; Example;
### Description: <Main function to perform classification using diverse classifiers and ensemble techniques>
### Input: data,  target truth, features to be used, number of tuning iterations
### Ouput: predictions for the test set, files with evaluation metrics
###################################

from . import ensemble
from . import classification
from sklearn.model_selection import train_test_split
import os
import numpy as np
from pathlib import Path

# data: data considered, pandas DataFrame
# target: labels corresponding to data, pandas Series
# features: set of features to use
# signature_name: the name used to identify the signature (e.g. the output name of Feature_Selection\main_selection.py)
# n_iter_classif: number of iterations for the tuning of selectors
# criterion: metric to be used  for the models tuning, it has to be an authorized string from sklearn metrics
# path: where to save the intermediary and result files
# seed: random seed
# splitted: if True, whole data and target sets are to be used for training
# The last parameters are used for the model selection part (rule_model_selection in classification.py for more details).
# ssd: sample size determination, allows to perform the experiments on the influence fo the sample size, if not None,
# ssd characterizes the type of experiment with a triplet: the sample size percent to use "p", the fact of keeping a same test set for all
# sample sizes ("partition") or a different test set for each sample size, and a second seed "subseed" to keep.
def main_classification(data, target, features, signature_name, criterion="balanced_accuracy", n_iter_classif=10,
                        path='./', seed=0, splitted=False, function="id", metric_constraint=["ba"], metric_optim=["ba"],
                        threshold=0.05, ssd=None):
    np.random.seed(seed)

    # Number of models to select to build the ensemble methods
    select_k_model = 5

    # Path template for the classification files
    name = "_".join(["it_cl", str(n_iter_classif), "feat", signature_name, "seed", str(seed),
                     str(splitted), "f", function, "th", str(threshold), "const",
                     "_".join([metric for metric in metric_constraint]),
                     "optim", "_".join([metric for metric in metric_optim]), "model_num", str(select_k_model)])
    # Number of folds in the cross-validations
    cv_num = 10

    # Number of subprocesses
    n_jobs = 35
    # Details about the models which will be at the beginning of each model performance
    params = ["Model", "Params"]
    # Assessment metrics considered, if it has to be modified, the functions in classification have to be modified too
    metrics = ["Balanced Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
    # Convert metrics initials into their position in the results for the model selection constraints
    dict_metrics = {"ba": 2, "wp": 4, "wr": 6, "wf": 8}
    metric_constraint = [dict_metrics[metric] for metric in metric_constraint]
    metric_optim = [dict_metrics[metric] for metric in metric_optim]

    dict_function = {"id": lambda x: x, "abs": lambda x: np.abs(x)}
    function = dict_function[function]

    directory_data = os.path.dirname(path + "preprocess/") + "/"
    results_directory = os.path.dirname(path + "results/") + "/"
    classification_directory = results_directory + "classification/"

    Path(results_directory).mkdir(parents=True, exist_ok=True)
    Path(directory_data).mkdir(parents=True, exist_ok=True)
    Path(classification_directory).mkdir(parents=True, exist_ok=True)

    partition = False
    if ssd:
        p, partition, subseed = ssd
        path_add = "_".join(["ssd", str(p), "partition" * partition, str(subseed)])
        directory_data += path_add
        results_directory += path_add
        classification_directory += path_add
    # Split the data for performing the selection on the training set only
    if not splitted:
        if not os.path.exists(directory_data + str(seed) + "_x_train.pkl"):
            # If ssd, we have to keep p% of the data for training, we define a unique test set per seed if partition
            # otherwise the test set is dependent of the sample size.
            if not ssd or partition:
                x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed,
                                                                    shuffle=True,
                                                                    stratify=target)
            if ssd:
                if partition:
                    x = x_train
                    y = y_train
                else:
                    x, _, y, _ = train_test_split(data, target, train_size=p, random_state=subseed,
                                                  shuffle=True,
                                                  stratify=target)
                    p = 0.25
                    subseed = seed
                x_train, x, y_train, y = train_test_split(x, y, train_size=p, random_state=subseed,
                                                          shuffle=True,
                                                          stratify=y)
                if not partition:
                    x_test = x
                    y_test = y
            x_train.to_pickle(directory_data + str(seed) + "_x_train")
            x_test.to_pickle(directory_data + str(seed) + "_x_test")
            y_train.to_pickle(directory_data + str(seed) + "_y_train")
            y_test.to_pickle(directory_data + str(seed) + "_y_test")
        else:
            x_train = pd.read_pickle(directory_data + str(seed) + "_x_train")
            y_train = pd.read_pickle(directory_data + str(seed) + "_y_train")
            x_test = pd.read_pickle(directory_data + str(seed) + "_x_test")
            y_test = pd.read_pickle(directory_data + str(seed) + "_y_test")
    else:
        x_train = data
        y_train = target
    # Tune all the models and get the best parameters and the cross-validation performance
    results, models = classification.models_tuning(x_train.iloc[:, features], y_train, criterion,
                                                   n_iter=n_iter_classif,
                                                   cv_num=cv_num,
                                                   name=results_directory + name, n_jobs=n_jobs, seed=seed)
    # Ensemble technique names and instanciation
    ensemble_names = ["Majority", "Weighted Majority", "Stacking"]
    ensemble_models = [
        ensemble.ensemble_majority(models, results, selection=[function, metric_constraint, metric_optim, threshold],
                                   k=select_k_model),
        ensemble.ensemble_weighted_majority(models, results,
                                            selection=[function, metric_constraint, metric_optim, threshold],
                                            k=select_k_model),
        ensemble.ensemble_stacking(models, results, selection=[function, metric_constraint, metric_optim, threshold],
                                   k=select_k_model)]
    # Ensemble models cross-validation
    results += classification.cross_val(x_train.iloc[:, features], y_train, ensemble_models,
                                        ensemble_names, select_k_model,
                                        criterion=criterion,
                                        cv_num=cv_num, seed=seed)
    # Identify the best model: best average performance on validation for the selected metric and difference of performacne
    # between training and validation < threshold, difference rules can be experimented,
    # The best model has to be selected on cross-validation results only: No test overfit!
    best_model_idx, _ = classification.rule_model_selection(results, function, metric_constraint, metric_optim, threshold)
    models += ensemble_models
    if not splitted:
        aux, confusion_matrices = classification.test_results(models, x_train.iloc[:, features], y_train,
                                                              x_test.iloc[:, features], y_test)
        # Distinguish the ensemble models from the others to give more details about their parameters
        results_test = [[results[i][idx] for idx in range(len(params))] + aux[i] for i in
                        range(len(aux) - len(ensemble_names))]
        results_test += [[ensemble_names[i - len(results)] + ":" + ";".join(
            [j for j in list(models[i].named_estimators_.keys())]),
                          ";".join([str(len(features)), "k=" + str(select_k_model)])] + aux[i]
                         for i in range(len(aux) - len(ensemble_names), len(aux))]

    with open(classification_directory + name + "_results.csv", 'w') as f:
        # Write the number of samples for each class
        f.write("\n".join(
            [",".join([str(np.unique(target, return_counts=True)[i][j]) for i in range(len(np.unique(target)))])
             for j in range(len(np.unique(target)))]))
        # Write the number of features and their names
        f.write(str(len(features)) + "\n")
        f.write(",".join([str(data.columns[g]) for g in features]) + "\n")
        f.write("Cross-Validation\n")
        f.write(",".join(params) + "," + ",,".join(metrics) + "\n")
        f.write("," * len(params) + "Train,Validation," * len(metrics) + "\n")
        classification.performance_saving(f, results, best_model_idx, 2, len(params))
        if not splitted:
            f.write("\n\n")
            f.write("Test\n")
            f.write(",".join(params) + "," + ",,".join(metrics) + "\n")
            f.write("," * len(params) + "Train,Test," * len(metrics) + "\n")
            classification.performance_saving(f, results_test, best_model_idx, 1, len(params))
    if not splitted:
        with open(classification_directory + name + "_confusion.csv", 'w') as f:
            classification.confusion_matrix_saving(f, confusion_matrices,
                                                   [",".join([str(res[idx]) for idx in range(len(params))]) for res in
                                                    results_test],
                                                   best_model_idx, np.unique(target))
    # Return name to identify the selected features
    return (results, results_test, confusion_matrices, models, best_model_idx, name)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    features = [0, 1, 2, 3]
    print(data, target)
    results, results_test, confusion_matrices, models, best_model_idx, name = main_classification(data, target,
                                                                                                  features,
                                                                                                  criterion="balanced_accuracy",
                                                                                                  signature_name="test",
                                                                                                  n_iter_classif=5,
                                                                                                  path='./', seed=0,
                                                                                                  splitted=False)
    print(results, results_test, confusion_matrices)
