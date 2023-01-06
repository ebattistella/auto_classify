####################################
### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Confidence Interval;
### Description: <Implement different statistical tools to better evaluate the results and propose a pipeline to
###               select the best parameter over different hyperparameters.>
### Input: scores
### Ouput: statistics on the results
###################################
import sys, os
import numpy as np
import pandas as pd
import classification
import fnmatch


# Compute a no parametric confidence interval on the results using evaluations on several experiments/seeds
def non_parametric_CI(scores, width=0.5):
    if isinstance(scores[0], pd.Series):
        scores = pd.concat(scores, axis=1)
        print("scores", scores)
        scores.fillna(0, inplace=True)
        print("scores", scores)
    # calculate lower percentile (e.g. 2.5)
    lower_p = width / 2.0
    # retrieve observation at lower percentile
    lower = max(0.0, np.percentile(scores, lower_p))

    # calculate upper percentile (e.g. 97.5)
    upper_p = 100 - lower_p
    # retrieve observation at upper percentile
    upper = min(1.0, np.percentile(scores, upper_p))
    return (lower, upper)


# The following function parses the different results files output of the main.py file.
# It selects the best model given the considered selection rule (please refer to main_classification.py for details)
# for each seed and feature selection technique.
# This function also allows to prepare a pandas Dataframe that can be used for visualization or further study.
# Notice that path is the path of a folder and thus, different experiemnts that should not be compared shoudl not be
# stored in a same folder.
def hyperparameter_selection(path,  splitted,
                             function="id", metric_constraint=["ba"], metric_optim=["ba"],
                        threshold=0.05):
    # Ensemble feature selection methos we consider.
    ensemble_name = ["Majority", "MV", "WMV", "Densest", "k-Heavy", "W-k Heavy"]

    # Path template for the feature selection files
    signature_name = "_".join(["num_selec_*_feat_num_*_filter", str(pre_filter),
                            "seed_*_clf_only_*", str(splitted), "th_sel_classif_*", "th_stat_*"])

    # add the parameters to the name as the results will depend on them
    signature_name += "_".join([str(i) for i in threshold_select_classif]) + "th_stat" + str(threshold_stats)

    # Path template for the classification files
    name = "_".join(["it_classif_*_features", signature_name, "seed_*",
                     str(splitted), "function", function, "th", str(threshold), "constraint", "_".join([metric for metric in metric_constraint]),
                     "optim", "_".join([metric for metric in metric_optim]), "model_num_*"])

    # Path template for the output file
    output_name = "_".join(["features", "filter", str(pre_filter), str(splitted), "function", function, "th",
                            str(threshold), "constraint", "_".join([metric for metric in metric_constraint]),
                     "optim", "_".join([metric for metric in metric_optim])])
    classification_directory = os.path.dirname(path + "results/") + "/classification/"
    evaluation_directory = os.path.dirname(path + "results/") + "/evaluation/"
    try:
        os.stat(evaluation_directory)
    except:
        os.mkdir(evaluation_directory)

    # Details about the models which will be at the beginning of each model performance
    params = ["Model", "Params"]
    # Assessment metrics considered, if it has to be modified, the functions in classification have to be modified too
    metrics = ["Balanced Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1", "AUC"]

    # Define the rules and constreaints to select the best models as done in classification.py
    dict_metrics = {"ba": 2, "wp": 4, "wr": 6, "wf": 8, "auc": 10}
    metric_constraint = [dict_metrics[metric] for metric in metric_constraint]
    metric_optim = [dict_metrics[metric] for metric in metric_optim]

    dict_function = {"id": lambda x: x, "abs": lambda x: np.abs(x)}
    function = dict_function[function]

    # The range of parameters we consider, 1000 is used to keep all results.
    selecs_ref = range(1000)
    clfs_ref = range(1000)
    feats_ref = list(range(40))

    # To store the different results on test and cross validation
    seeds = []
    results = []
    results_cv = []
    # To count the total number of results files.
    filtered = 0

    for filename in os.listdir(classification_directory):
        # We search over the files whose name correspond to the results we are considering.
        if fnmatch.fnmatch(filename, name + "_results.csv"):
            filtered += 1

            # Recover the different variables from the file name
            num_features = filename.split("feat_num_*_")[1].split("_features")[0]
            seed_selec = filename.split("seed_")[1].split("_clf_only_")
            seed_clf = filename.split("seed_")[-1].split(str(splitted) + "_function_")
            seed = (seed_selec, seed_clf)
            if seed not in seeds:
                seeds.append(seed)
            index_seed = seeds.index(seed)
            n_iter_selec = filename.split("num_selec_")[1].split("feat_")[0]
            n_iter_clf = int(filename.split("it_classif_")[1].split("_features")[0])
            clf_only = filename.split("clf_only_")[1].split(str(splitted) + "_th_sel")[0]
            select_k_model = filename.split("model_num_")[1].split("_results")[0]
            threshold_select_classif = filename.split["th_sel_classif"][1].split("_th_stat")[0]
            threshold_stats = filename.split["th_stat"][1].split("_seed")[0]


            if num_features not in feats_ref or n_iter_selec not in selecs_ref or n_iter_clf not in clfs_ref:
                print(filename)
                print("not in refs", num_features, n_iter_clf, n_iter_selec)
                continue


            if len(seeds) > len(results):
                results.append([[] for _ in range(len(ensemble_name))])
                results_cv.append([[] for _ in range(len(ensemble_name))])

            f = open(classification_directory + filename, "r")

            is_test = False
            for line in f.readlines():
                hyperparam = ",".join([num_features, threshold_select_classif, threshold_stats, clf_only, select_k_model])
                if "Test" in line and "Train" not in line:
                    is_test = True
                    continue
                if "**" in line:
                    if line.split(',')[0] in ensemble_name:
                        if is_test:
                            aux = [line.split(',')[0]]
                            for res in line.split(','):
                                try:
                                    l = float(res)
                                    aux.append(l)
                                except:
                                    continue
                            aux.append(hyperparam)
                            aux.append(filename)
                            results[index_seed][ensemble_name.index(line.split(',')[0])].append(aux)
                        else:
                            aux = [line.split(',')[0]]
                            for res in line.split(','):
                                try:
                                    l = float(res.split('+')[0])
                                    aux.append(l)
                                except:
                                    continue
                            aux.append(hyperparam)
                            aux.append(filename)
                            results_cv[index_seed][ensemble_name.index(line.split(',')[0])].append(aux)
            f.close()
    if len(results) == 0 or len(results[0][0]) == 0:
        print("No Results")
        return
    print("number experiments", sum([len(results[i][0]) for i in range(len(seeds))]), "filtered:", filtered)

    # Find the best model for every feature selection method on each seed.
    maxi = [[[0, 0] for _ in range(len(ensemble_name))] for _ in seeds]
    for seed in seeds:
        index_seed = seeds.index(seed)
        for method in range(len(ensemble_name)):
            if len(results_cv[index_seed][method]) > 0:
                maxi[index_seed][method] = rule_model_selection(results_cv[index_seed][method], function, metric_constraint, metric_optim, threshold)
                maxi[index_seed][method][1] = str(maxi_perso[index_seed][method][1])

    # Agregate the results
    max_summary = [[[results[index_seed][method][maxi[index_seed][method][0]][idx]
                     for idx in range(len(results[index_seed][method][0]))]
                    for method in range(len(ensemble_name))]
                   for index_seed in range(len(seeds))]
    # Results per seed to save
    max_to_write = [
        [";".join([str(results[index_seed][method][maxi[index_seed][method][0]][idx])
                   for idx in range(len(results[index_seed][method][0]))])
         + ";" + str(maxi[index_seed][method][1]) + ";" + str(seeds[index_seed])
         for method in range(len(ensemble_name))] for index_seed in range(len(seeds))]

    with open(evaluation_directory + output_name + ".csv", "w") as f:
        f.write("sep=;\n")
        # Save the performance and characteristics of the bes model for each seed and feature selection technique.
        for seed in seeds:
            f.write("Seed," + str(seed) + "\n")
            index_seed = seeds.index(seed)
            f.write(";".join(params) + ";" + ";;".join(metrics) + "\n")
            f.write(";;" + "Train;Test;" * len(metrics) + "\n")
            for idx in range(len(max_to_write[index_seed])):
                f.write(max_to_write[index_seed][idx] + "\n")
            f.write("\n")

        # Prepare two dataframes for a boxplot and a lineplot (see visualization.py)
        dict_metric = {2: "BA", 4: "WP", 6: "WR", 8: "WF", 10: "AUC"}
        plot_col = ["Method", "Metric", "Value"]
        line_col = ["Method", "Metric", "X", "Value", "Value Validation"]
        box_plot_df = pd.DataFrame(columns=plot_col)
        line_plot_df = pd.DataFrame(columns=line_col)
        for method in range(len(ensemble_name)):
            aux_df = [
                (ensemble_name[method], dict_metric[idx], max_summary[index_seed][method][idx])
                for index_seed in range(len(seeds))
                for idx in range(len(params), len(metrics) + len(params))
                if idx in dict_metric.keys()]
            aux_line_df = [
                (ensemble_name[method], dict_metric[idx], seeds[index_seed],
                 results_all[index_seed][idx][maxi_perso[index_seed][idx][0]][i],
                 results_cv_all[index_seed][idx][maxi_perso[index_seed][idx][0]][i])
                for index_seed in range(len(seeds))
                if len(results_all[index_seed][idx]) > 0 for i in range(len(params), len(metrics) + len(params))
                if i in dict_metric.keys()]
            aux_df = pd.DataFrame(aux_df, columns=plot_col)
            aux_line_df = pd.DataFrame(aux_line_df, columns=line_col)
            box_plot_df = box_plot_df.append(aux_df)
            line_plot_df = line_plot_df.append(aux_line_df)
        box_plot_df.to_pickle(evaluation_directory + 'boxplot_' + output_name)
        line_plot_df.to_pickle(evaluation_directory + 'lineplot_' + output_name)

        f.write("\n Average over seeds with Confidence Intervals\n")
        f.write(";".join(params) + ";" + ";;".join(metrics) + "\n")
        f.write(";;" + "Train;Test;" * len(metrics) + "\n")
        for method in range(len(ensemble_name)):
            CI = [cross_validation.non_parametric_CI([max_summary[index_seed][method][res]
                                                      for index_seed in range(len(seeds))])
                  for res in range(1, len(max_summary[0][method]) - 1)]

            f.write(str(ensemble_name[method]) + ";" + ";".join([str(np.mean([max_summary[index_seed][method][res]
                                                                                for index_seed in range(len(seeds))]))
                                                                   + "+/- [" +
                                                                   str(CI[res - 1][0]) + "," + str(CI[res - 1][1]) + "]"
                                                                   for res in
                                                                   range(len(params), len(metrics) + len(params))]) + "\n")
        f.write("\n")
    print("done")


if __name__ == "__main__":
    hyperparameter_selection('./', False)