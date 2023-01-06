### Author: Enzo Battistella
### Date: 3/17/2022
####################################
### Keywords: Classification; Feature Selection; Models Tuning; Model Selection;
### Description: <Leverage the feature selection and classification frameworks to provide a full end to end pipeline>
### Input: data, ground truth, learned distance
### Ouput: signature of features, models, predictions, assessment of the classification
###################################
import sys
from ..FeatureSelection import main_selection
from ..Classification import main_classification
import numpy as np

import pickle



def main_function(data, target, n_iter_selec=5, k_feat=10, path='./', pre_filter=False, seed=0, clf_only=False,
                  n_iter_classif=10, function="id", criterion="balanced_accuracy", metric_constraint=["ba"], metric_optim=["ba"],
                        threshold=0.05, ssd=None):
    features, name = main_selection.main_selection(data, target, criterion=criterion, n_iter_selec=n_iter_selec, k_feat=k_feat, path=path,
                                                   pre_filter=pre_filter, seed=seed,
                                                   clf_only=clf_only, splitted=False, ssd=ssd)
    ensemble_name = ["Majority", "MV", "WMV", "Densest", "k-Heavy", "W-k Heavy"]
    for idx in range(len(features)):
        _, _, _, models, best_model_idx, name_classif = main_classification.main_classification(data, target,
                                                                                                features[idx], criterion=criterion,
                                                                                                signature_name=
                                                                                                ensemble_name[
                                                                                                    idx] + "_" + name,
                                                                                                n_iter_classif=n_iter_classif,
                                                                                                path=path, seed=seed,
                                                                                                splitted=False,
                                                                                                function=function,
                                                                                                metric_constraint=metric_constraint,
                                                                                                metric_optim=metric_optim,
                                                                                                threshold=threshold, ssd=ssd)

        # Save the features and best model, fully tuned and trained, ready for predictions on an external set
        np.save(name_classif, features[idx])
        pickle.dump(models[best_model_idx], open(name + ".sav", "wb"))


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)

    n_iter_selec= int(sys.argv[1])
    k_feat = int(sys.argv[2])
    pre_filter = sys.argv[3] == "1"
    seed = int(sys.argv[4])
    clf_only = sys.argv[5] == "1"
    n_iter_classif = int(sys.argv[6])
    ssd = None
    if len(sys.argv) > 8:
        ssd = (int(sys.argv[7]), bool(sys.argv[8]), int(sys.argv[9]))
    path = "./"

    main_function(data, target, n_iter_selec=n_iter_selec, k_feat=k_feat, path=path, pre_filter=pre_filter, seed=seed, clf_only=clf_only,
                  n_iter_classif=n_iter_classif, ssd=ssd)
