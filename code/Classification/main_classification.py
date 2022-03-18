### Author: Enzo Battistella
### Date: 3/16/2022
####################################
### Keywords: Classification; Main; Example;
### Description: <Main function to perform classification using diverse classifiers and ensemble techniques>
### Input: data,  target truth, features to be used, number of tuning iterations
### Ouput: predictions for the test set, files with evaluation metrics
###################################

import ensemble
import classification
from sklearn.model_selection import train_test_split
import os
import numpy as np


# data: data considered, pandas DataFrame
# target: labels corresponding to data, pandas Series
# features: set of features to use
# signature_name: the name used to identify the signature (e.g. the output name of Feature_Selection\main_selection.py)
# n_iter_classif: number of iterations for the tuning of selectors
# criterion: metric to be used  for the models tuning, it has to be an authorized string from sklearn metrics
# path: where to save the intermediary and result files
# seed: random seed
# splitted: if True, whole data and target sets are to be used for training
def main_classification(data, target, features, signature_name, n_iter_classif=10, criterion='balanced_accuracy',
                        path='./', seed=0, splitted=False):
    np.random.seed(seed)

    # Threshold used in the selection of the best models, it has to be the training and validation performance on criterion
    # metric < select_th, used to avoid overfitted models
    select_th = 0.1

    # Number of models to select to build the ensemble methods
    select_k = 5

    # Path template for the feature selection files
    name = "_".join(["num_selec", str(n_iter_classif), "features", signature_name, criterion, "seed", str(seed),
                     str(splitted), "th", str(select_th), "k", str(select_k)])
    # Number of folds in the cross-validations
    cv_num = 10

    # Number of subprocesses
    n_jobs = 35
    # Details about the models which will be at the beginning of each model performance
    params = ["Model", "Params"]
    # Assessment metrics considered, if it has to be modified, the functions in classification have to be modified too
    metrics = ["Balanced Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "AUC"]
    # Index of the metric used as criterion in the list of results
    select_metric = [i.lower().replace(" ", "_") for i in metrics].index(criterion) + len(params)

    directory_data = os.path.dirname(path + "preprocess/") + "/"
    results_directory = os.path.dirname(path + "results/") + "/"
    try:
        os.stat(results_directory)
        os.stat(directory_data)
    except:
        os.mkdir(results_directory)
        os.mkdir(directory_data)
        os.mkdir(results_directory + "performance/")

    # Split the data for performing the tuning and training on the training set only
    if not splitted:
        if not os.path.exists(directory_data + str(seed) + "_x_train.pkl"):
            x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed,
                                                                shuffle=True,
                                                                stratify=target)
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
        ensemble.ensemble_majority(models, results, th=select_th, metric=select_metric, k=select_k),
        ensemble.ensemble_weighted_majority(models, results, th=select_th, metric=select_metric,
                                            k=select_k),
        ensemble.ensemble_stacking(models, results, th=select_th, metric=select_metric, k=select_k)]
    # Ensemble models cross-validation
    results += classification.cross_val(x_train.iloc[:, features], y_train, ensemble_models,
                                        ensemble_names, select_k, select_metric, select_th,
                                        cv_num=cv_num, seed=seed)
    # Identify the best model: best average performance on validation for the selected metric and difference of performacne
    # between training and validation < select_th
    # The best model has to be selected on cross-validation results only: No test overfit!
    best_model_idx = results[np.argmax([float(i[select_metric + 2]) if
                                        0 <= float(i[select_metric]) - float(i[select_metric + 2])
                                        <= select_th else 0
                                        for i in results])][1]
    models += ensemble_models
    aux, confusion_matrices = classification.test_results(models, x_train.iloc[:, features], y_train,
                                                          x_test.iloc[:, features], y_test)
    # Distinguish the ensemble models from the others to give more details about their parameters
    results_test = [[results[i][idx] for idx in range(len(params))] + aux[i] for i in
                    range(len(aux)-len(ensemble_names))]
    results_test += [[ensemble_names[i - len(results)] + ":" + ";".join(
        [j for j in list(models[i].named_estimators_.keys())]),
                      ";".join([str(len(features)), "k=" + str(select_k),
                                "metric=" + str(select_metric), "th=" + str(select_th)])] + aux[i]
                     for i in range(len(aux) - len(ensemble_names), len(aux))]

    with open(directory_data + "classification/" + name + "_results.csv", 'w') as f:
        # Write the number of samples for each class
        f.write("\n".join(
            [",".join([str(np.unique(target, return_counts=True)[i][j]) for i in range(len(np.unique(target)))])
             for j in range(len(np.unique(target)))]))
        # Write the number of features and their names
        f.write(str(len(features)) + "\n")
        f.write(",".join([data.columns[g] for g in features]) + "\n")
        f.write("Cross-Validation\n")
        f.write(",".join(params) + ",,".join(metrics) + "\n")
        f.write("," * len(params) + "Train,Validation" * len(metrics) + "\n")
        classification.performance_saving(f, results, best_model_idx, 2, len(params))

        f.write("\n\n")
        f.write("Test\n")
        f.write(",".join(params) + ",,".join(metrics) + "\n")
        f.write("," * len(params) + "," + "Train,Test," * len(metrics) + "\n")
        classification.performance_saving(f, results_test, best_model_idx, 1, len(params))

    with open(results_directory + "classification/" + name + "_confusion.csv", 'w') as f:
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
                                                                                                  signature_name="test",
                                                                                                  n_iter_classif=5,
                                                                                                  criterion="balanced_accuracy",
                                                                                                  path='./', seed=0,
                                                                                                  splitted=False)
    print(results, results_test, confusion_matrices)
