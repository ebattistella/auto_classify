### Author: Enzo Battistella
### Date: 3/16/2022
####################################
### Keywords: Feature Selection; Main; Example;
### Description: <Main function to perform feature selection and the ensemble feature selection>
### Input: data,  target truth, number of features to be selected, number of tuning iterations
### Ouput: prevalence of the features per feature selector, selected features for each ensemble method considered
###################################

from . import ensemble_feature_selection
from . import feature_selection
from sklearn.model_selection import train_test_split
import os
import numpy as np
from pathlib import Path

# WARNING: the cpp files have to be compiled to run the graph based ensemble techniques
# For cooc_selector, first download boost_1_63_0 in the working folder.
# Then, use g++ density_decomposition.cpp -fopenmp -fpermissive -I ./boost_1_63_0/ -o density_decomposition -O3
# For the others use gcc bb_dks.c -o bb_dks -O9
# data: data considered, pandas DataFrame
# target: labels corresponding to data, pandas Series
# n_iter_selec: number of iterations for the tuning of selectors
# k_feat: number of feature to select
# path: where to save the intermediary and result files
# pre_filter: if True, keep only the most correlated features by an anova test befor feature selection
# seed: random seed
# clf_only: if True, do not use feature selectors based on statistical metrics
# splitted: if True, whole data and target sets are to be used for training
# ssd: sample size determination, allows to perform the experiments on the influence fo the sample size, if not None,
# ssd characterizes the type of experiment with a triplet: the sample size percent to use "p", the fact of keeping a same test set for all
# sample sizes ("partition") or a different test set for each sample size, and a second seed "subseed" to keep.
def main_selection(data, target, criterion="balanced_accuracy", n_iter_selec=10, k_feat=10, path='./', pre_filter=False, seed=0, clf_only=False,
                  splitted=False, ssd=None):
    np.random.seed(seed)

    # Feature selector names
    features_selectors_names = [
        "Features", "Decision Tree Threshold", "Linear SVC Threshold",
        "Gradient Boosting Threshold", "AdaBoost Threshold", "Lasso Threshold", "chi2",
        "ANOVA F-value", "Mutual Information"
    ]

    # Ensemble technique names
    ensemble = [ensemble_feature_selection.majority_selector, ensemble_feature_selection.threshold_selector,
                ensemble_feature_selection.threshold_selector, ensemble_feature_selection.cooc_selector,
                ensemble_feature_selection.k_density_selector, ensemble_feature_selection.k_density_selector]
    ensemble_name = ["Majority", "Threshold", "Threshold WMA", "Densest", "Heaviest Density", "Heaviest Density WMA"]
    # Path template for the feature selection files
    name = "_".join(["num_selec", str(n_iter_selec), "feat_num", str(k_feat), "filter", str(pre_filter),
                            "seed", str(seed), "clf_only", str(clf_only), str(splitted)])
    # Number of folds in the cross-validation for the feature selection
    cv_num = 10
    # Level of decomposition in the density friendly decomposition, k_S>1 generally yields to many features
    k_S = 1

    # Acceptability threshold for the Majority Voting selector, by default select the features with prevalence above 25%
    th_majority = 0.25

    # Thresholds for the feature selection techniques, they should be adapted such that all methods selects similar number of features
    # Thresholds for the classifiers based feature selection techniques, to be adapted according to the number of features
    # The order of the thresholds is the same as in the list features_selectors names
    threshold_select_classif = [10**(-5), "median", "mean", "mean", "mean"]
    # Threshold for the statistical test based feature selectors, represents the percentile of feature to select
    threshold_stats = 34.
    #add the parameters to the name as the results will depend on them
    name += "_".join(["th_sel_classif", "_".join([str(i) for i in threshold_select_classif]), "th_stat", str(threshold_stats)])

    # Number of subprocesses
    n_jobs = 35

    directory_data = os.path.dirname(path + "preprocess/") + "/"
    results_directory = os.path.dirname(path + "results/") + "/"
    selection_directory = results_directory + "selection/"
    density_directory = directory_data + "density/"

    Path(results_directory).mkdir(parents=True, exist_ok=True)
    Path(directory_data).mkdir(parents=True, exist_ok=True)
    Path(density_directory).mkdir(parents=True, exist_ok=True)
    Path(selection_directory).mkdir(parents=True, exist_ok=True)

    partition = False
    if ssd:
        p, partition, subseed = ssd
        path_add = "_".join(["ssd", str(p), "partition"*partition, str(subseed)])
        directory_data += path_add
        results_directory += path_add
        density_directory += path_add
        selection_directory += path_add
    # Split the data for performing the selection on the training set only
    if not splitted:
        if not os.path.exists(directory_data + str(seed) + "_x_train.pkl"):
            # If ssd, we have to keep p% of the data for training, we define a unique test set per seed if partition
            # otherwise the test set is dependent of the sample size.
            if not ssd or partition:
                x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed, shuffle=True,
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
    else:
        x_train = data
        y_train = target

    # Perform the pre-filter using an Anova test if pre_filter is True
    map_pre_filter = {idx: idx for idx in range(len(x_train.columns))}
    features_pre_filter = list(range(len(x_train.columns)))
    if pre_filter:
        if not os.path.exists(
                results_directory + str(seed) + "_pre_filter.csv"):
            features_pre_filter = feature_selection.pre_filter_anova(x_train, y_train, threshold_stat=threshold_stats + 0.3,
                                          name=directory_data + str(seed) + "_pre_filter")
            map_pre_filter = {idx: features_pre_filter[idx] for idx in range(len(features_pre_filter))}
            np.save(directory_data + str(seed) + "_map_pre_filter",
                    map_pre_filter)
            np.save(results_directory + str(seed) + "_pre_filter_features", features_pre_filter)
            np.savetxt(results_directory + str(seed) + "_pre_filter_features.txt",
                [x_train.columns[feat] for feat in features_pre_filter], delimiter="\n", fmt="%s")
        else:
            map_pre_filter = list(np.load(directory_data + str(seed) + "_map_pre_filter.npy"))
            features_pre_filter = np.load(results_directory + str(seed) + "_pre_filter_features.npy")
    if not os.path.exists(results_directory + str(seed) + "_features.csv"):
        if not os.path.exists(directory_data + name + "_cooc_weights.npy"):
            cooc, gene_map, max_it, weights, cooc_weights = feature_selection.feature_selector(x_train.iloc[:,features_pre_filter],
                                                                                y_train, criterion=criterion, n_iter=n_iter_selec,
                                                                                cv_num=cv_num, clf_only=clf_only,
                                                                                thresholds=threshold_select_classif,
                                                                                threshold_stat=threshold_stats,
                                                                                name=directory_data + name,
                                                                                n_jobs=n_jobs, seed=seed)
            # Merge the co-selection matrices of the different feature selectors and get the map mapping the indexes
            # of the merged matrix to the initial features (after pre-filtering)
            oc, map_oc = feature_selection.merge(cooc, gene_map)
            np.save(directory_data + name + "_cooc", cooc)
            np.save(directory_data + name + "_weights", weights)
            np.save(directory_data + name + "_cooc_weights", cooc_weights)
            np.save(directory_data + name + "_oc", oc)
            np.save(directory_data + name + "_map_oc", map_oc)
            np.save(directory_data + name + "_max_it", max_it)
            np.save(directory_data + name + "_map_gene", gene_map)
        else:
            cooc = np.load(directory_data + name + "_cooc.npy", allow_pickle=True)
            weights = np.load(directory_data + name + "_weights.npy", allow_pickle=True)
            cooc_weights = np.load(directory_data + name + "_cooc_weights.npy", allow_pickle=True)
            oc = np.load(directory_data + name + "_oc.npy", allow_pickle=True)
            map_oc = np.load(directory_data + name + "_map_oc.npy", allow_pickle=True).item()
            max_it = np.load(directory_data + name + "_max_it.npy", allow_pickle=True)
            gene_map = np.load(directory_data + name + "_map_gene.npy",  allow_pickle=True)
            gene_map = [i for i in gene_map]
        # The diagonal of the co-selection matrix is the prevalence of the features
        diag = np.diag(oc)
        # Sort the features by decreasing prevalence
        sorted_idx = np.argsort(diag)[::-1]
        # Save the prevalence by feature selector and in total of each selected feature
        with open(selection_directory + name + "_features.csv", "w") as f:
            f.write(",".join(features_selectors_names) + ", Total\n")
            for feature in sorted_idx:
                # feature is in merged space put it back into space after filtering
                trans_feature = map_oc[feature]
                sum_feat = 0
                # map_pre_filter need to be applied to recover the index of the feature before pre-filtering
                f.write(str(x_train.columns[map_pre_filter[trans_feature]]) + ",")
                for i in range(len(cooc)):
                    if trans_feature in gene_map[i].keys():
                        # gene_map[i] map the feature index to the index in the co-selection matrix of feature selector i
                        count = cooc[i][gene_map[i][trans_feature], gene_map[i][trans_feature]]
                        sum_feat += count
                        f.write(str(count) + ",")
                    else:
                        f.write("0,")
                f.write(str(sum_feat) + "\n")
    else:
        max_it = np.load(directory_data + name + "_max_it.npy", allow_pickle=True)
        oc = np.load(directory_data + name + "_oc.npy", allow_pickle=True)
        map_oc = np.load(directory_data + name + "_map_oc.npy", allow_pickle=True).item()
        weights = np.load(directory_data + name + "_weights.npy", allow_pickle=True)
        diag = np.diag(oc)
    features = []
    for idx in range(len(ensemble)):
        if not os.path.exists(directory_data + name + "_" + ensemble_name[idx] + "_selected_features.npy"):
            # Select the variable corresponding to the ensemble method
            if "Dens" in ensemble_name[idx]:
                var = oc
                if "WMA" in ensemble_name[idx]:
                    var = cooc_weights
            else:
                var = diag
                if "WMA" in ensemble_name[idx]:
                    var = weights
            # All ensemble methods have the same signature for convenience, but not all variables are used for every method
            feature = ensemble[idx](var, max_it, k_feat=k_feat, k_S=k_S, th=th_majority, n_jobs=n_jobs,
                                    path=density_directory)
            # If no feature was selected, keep the one with highest prevalence
            if not feature:
                feature = [np.argmax(diag)]
            # Ensemble methods relying on co-selection/prevalence produce features in the space of selected features
            # map_oc enables to retrieve the initial index of the feature (after pre-filtering)
            if "WMA" not in ensemble_name[idx]:
                feature = [map_oc[feat] for feat in feature]
            # map_pre_filter enables to retrieve the initial feature after pre-filtering
            feature = [map_pre_filter[feat] for feat in feature]
            # Save the feature names
            np.savetxt(
                directory_data + name + "_" + ensemble_name[idx] + "_selected_features.txt",
                [x_train.columns[feat] for feat in feature], delimiter="\n", fmt="%s")
            # Save the feature indexes
            np.save(directory_data + name + "_" + ensemble_name[idx] + "_selected_features", feature)
        else:
            feature = np.load(directory_data + name + "_" + ensemble_name[idx] + "_selected_features.npy",
                              allow_pickle=True)

        features.append(feature)
    # Return name to identify the selected features
    return features, name


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    print(data, target)
    features, name = main_selection(data, target, n_iter_selec=5, k_feat=10, path='./', pre_filter=False, seed=0, clf_only=False,
                             splitted=False)
