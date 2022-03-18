### Author: Enzo Battistella
### Date: 3/17/2022
####################################
### Keywords: Classification; Feature Selection; Models Tuning; Model Selection;
### Description: <Leverage the feature selection and classification frameworks to provide a full end to end pipeline>
### Input: data, ground truth, learned distance
### Ouput: signature of features, models, predictions, assessment of the classification
###################################
import pickle

from ..Feature_Selection import main_selection
from ..Classification import main_classification
import pickle
import sys

def main_function(data, target, n_iter_selec=5, k_feat=10, path='./', pre_filter=False, seed=0, clf_only=False,
                  n_iter_classif=10, criterion='balanced_accuracy'):
    features, name = main_selection.main_selection(data, target, n_iter_selec=n_iter_selec, k_feat=k_feat, path=path,
                                                   pre_filter=pre_filter, seed=seed,
                                                   clf_only=clf_only, splitted=False)
    ensemble_name = ["Majority", "Threshold", "Threshold WMA", "Densest", "Heaviest Density", "Heaviest Density WMA"]
    for idx in range(len(features)):
        _, _, _, models, best_model_idx, name_classif = main_classification.main_classification(data, target,
                                                                                                features[idx],
                                                                                                signature_name=
                                                                                                ensemble_name[
                                                                                                    idx] + "_" + name,
                                                                                                n_iter_classif=n_iter_classif,
                                                                                                criterion=criterion,
                                                                                                path=path, seed=seed,
                                                                                                splitted=False)

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
    criterion = "balanced_accuracy"
    path = "./"

    main_function(data, target, n_iter_selec=n_iter_selec, k_feat=k_feat, path=path, pre_filter=pre_filter, seed=seed, clf_only=clf_only,
                  n_iter_classif=n_iter_classif, criterion=criterion)
