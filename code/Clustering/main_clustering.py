import clustering_algorithms
import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Main function to perform the end-to-end clustering
# n_cluster_max: maximal number of clusters authorize
def launcher(n_clusters_max, name, x, seed):
    # The clustering assessment metrics considered
    score_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Dunn Index']
    # Create the directory folders and path
    directory_data = os.path.dirname("../../data/clustering/ml_processed/iris/") + "/"
    results_directory = os.path.dirname("../../data/clustering/results/iris/") + "/"
    name += "_" + str(n_clusters_max) + "_" + str(seed)
    test_name = results_directory + "test_clustering_" + name
    train_name = results_directory + "train_clustering_" + name
    try:
        os.stat(results_directory)
    except:
        os.mkdir(results_directory)
        os.mkdir(results_directory + "clustering/")
        os.mkdir(results_directory + "selection/")
        os.mkdir(results_directory + "samples/")
    try:
        os.stat(directory_data)
    except:
        os.mkdir(directory_data)
    # Split into training and test to check the generalizability of the discovered patterns
    if not os.path.exists(directory_data + "data_" + name + ".pkl"):
        x_train, x_test = train_test_split(x, test_size=0.25, random_state=seed, shuffle=True)
        x_train.to_pickle(directory_data + "x_train_" + name)
        x_test.to_pickle(directory_data + "x_test_" + name)
    else:
        x_train = pd.read_pickle(directory_data + "x_train_" + name)
        x_test = pd.read_pickle(directory_data + "x_test_" + name)
    # Normalize the data using the training set properties only to avoid data leakage
    mm = MinMaxScaler()
    x_train = mm.fit_transform(x_train).T
    x_test = mm.transform(x_test).T
    # Tune the clustering algorithms and get the training performance and clusters
    results, best_params, clusters_train = clustering_algorithms.tuning(x_train, n_clusters_max=n_clusters_max + 1, metric_idx=0,
                                                        name=train_name)
    print("Tuned")
    # Get the testing performance and clusters
    results_test, clusters = clustering_algorithms.test(x_test, best_params, test_name)
    best_id = [[] for _ in clusters_train]
    # Write the results in a csv file
    print("Tested")
    with open(results_directory + "clustering/" + name + ".csv", "w") as f:
        f.write("Training\n")
        f.write(",," + ",".join(score_names) + "\n")
        counter = 0
        for l in results:
            temp = 0
            for aux_res in l:
                f.write(str(aux_res[0]) + "," + ";".join([i[0] + ";" + str(i[1]) for i in aux_res[1]]) + ",")
                for i in range(2, len(aux_res)):
                    f.write(str(aux_res[i][1]) + ",")
                # Mark the best algorithm on training with ****
                if counter < len(best_params) and aux_res[1] == best_params[counter][1]:
                    f.write("****,")
                    best_id[counter] = temp
                    temp = 0
                    counter += 1
                temp += 1
                f.write("\n")
        f.write("Test\n")
        f.write(",," + ",".join(score_names) + "\n")
        counter = 0
        for aux_res in results_test:
            np.save(results_directory + "samples/" + aux_data + max_tissue + cpt + aux_res[0] +
                    "_clusters.csv", clusters[counter])
            f.write(str(aux_res[0]) + "," + ";".join([i[0] + ";" + str(i[1]) for i in aux_res[1]]) + ",")
            # Write down the centers/medoids of the clusters, in a dimensionality reduction context they can be used to define a signature of features
            if str(aux_res[0]) != 'LP-Stability':
                signature = clustering_algorithms.medoid(clusters_train[counter][best_id[counter]], x_train, aux_res[1][0][1])
            else:
                signature = np.load(train_name + "_exemplars.npy", allow_pickle=True)
            np.save(results_directory + "selection/signature_" + aux_data + max_tissue + cpt + "_" + str(
                aux_res[0]) + ".npy", signature)
            for i in range(2, len(aux_res)):
                f.write(str(aux_res[i][1]) + ",")
            counter += 1
            f.write("\n")

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    name = "BRCA"

    n_clusters_max = int(sys.argv[1])
    seed = int(sys.argv[2])
    launcher(n_clusters_max, name, data, seed)
