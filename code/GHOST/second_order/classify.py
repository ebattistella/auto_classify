#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Distance Learning; Second-Order; Classification;
### Description: <Leverage a learned second-order distance to perform a prediction on a K-NN basis>
### Input: data, ground truth, learned distance
### Ouput: predictions, assessment of the classification, distance matrix
###################################

import os, sys
import pandas as pd
import numpy as np
from distance_calculator import calculator
import sklearn.cluster as skclust
from scipy.spatial.distance import pdist
from sklearn import metrics
from numpy import random
import graph_nn
import lp_stability
from scipy.stats import kendalltau, pearsonr, spearmanr


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


def kmedoid(distances, k):
    random.seed(10)
    m = distances.shape[0]  # number of points
    # Pick k random medoids.
    curr_medoids = np.array([-1] * k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1] * k)

    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids


def to_distance_matrix(distance, data_train, data_test, w, make_graph, nn_numbers, clusters, dist_vector_bool=True,dist_bool=True):
    max_path = 10 ** 2 * np.max(data_train, axis=None)
    print(len(w))
    distances = ["pearson", "spearman", "kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs",
                 "kendall_abs"]
    distances += ["learned_" + i for i in distances] + ["learned"]
    methods = [5, 6, 7, 9, 10, 11, 12, 13, 5, 6, 7, 9, 10, 11, 12, 13, 0]
    if dist_bool:
        dists_names = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
                       'chebyshev', 'matching', 'yule', 'braycurtis', 'dice', 'kulsinski', 'russellrao']
        dists_vectors = [lambda x, y: pdist(np.array([x, y]), i)[0] for i in dists_names]
    else:
        dists_names = []
        dists_vectors= []
    if dist_vector_bool:
        correlation_names = [pearsonr, spearmanr, kendalltau]
        dists_vectors += [lambda x, y: 1 - corre(x, y)[0] for corre in correlation_names]
    dists = [lambda x, y: (x - y) ** 2 for _ in range(np.shape(data_train)[1])]
    print(len(dists_vectors))
    idx_aux = len(w)
    if make_graph:
        idx_aux = -1
    print(distances[distance])
    clusters_names = np.unique(clusters)
    distance_matrix = np.zeros((np.shape(data_test)[0], np.shape(data_train)[0]))
    if distances[distance] != "learned":
        data = np.concatenate((data_train, data_test))
        distance_matrix = calculator(methods[distance], "lancet", 200, (w[:idx_aux - len(dists_vectors)] * data).T)
        distance_matrix = distance_matrix[-np.shape(data_test)[0]:, :]
        distance_matrix = distance_matrix[:, :np.shape(data_train)[0]]
    if make_graph or distances[distance] == "learned":
        aux_distance_train = np.array([[[0 if p == q else dists[i](data_train[p, i], data_train[q, i]) for i in
                                         range(np.shape(data_train)[1])] + [dist(data_train[p, :], data_train[q, :]) for
                                        dist in dists_vectors]
                                        for q in range(np.shape(data_train)[0])]
                                       for p in range(np.shape(data_train)[0])])
        aux_distance_train_test = np.array([[[0 if p == q else dists[i](data_test[p, i], data_train[q, i]) for i in
                                              range(np.shape(data_test)[1])] + [dist(data_test[p, :], data_train[q, :])
                                                                                for dist in dists_vectors]
                                             for q in range(np.shape(data_train)[0])]
                                            for p in range(np.shape(data_test)[0])])
    for patient in range(data_test.shape[0]):
        if make_graph:
            graph = graph_nn.graph_building(np.concatenate((np.concatenate((np.sum(aux_distance_train, axis=-1),
                                                                            np.sum(
                                                                                aux_distance_train_test[patient, :],
                                                                                axis=-1).reshape(-1, 1)), axis=1),
                                                            np.concatenate((np.sum(
                                                                aux_distance_train_test[patient, :], axis=-1),
                                                                            [0])).reshape(1, -1)),
                                                           axis=0), nn_numbers)
            paths_dist = graph_nn.path_finding(graph)
            paths_dist[paths_dist == np.inf] = max_path
            paths_dist = w[-1] * paths_dist
        for k in range(len(clusters_names)):
            cluster_k = [i for i in range(len(clusters)) if clusters[i] == k]
            for idx_train in cluster_k:
                if distances[distance] == "learned":
                    distance_matrix[patient, idx_train] = np.sum(
                        w[:idx_aux] * aux_distance_train_test[patient, idx_train]
                    )
                else:
                    distance_matrix[patient, idx_train] += np.sum(
                        w[-1 - int(make_graph) - dist_i] * dists_vectors[dist_i](data_test[patient, :],
                                                                                 data_train[idx_train, :]) for dist_i in
                        range(len(dists_vectors)))
                if make_graph:
                    distance_matrix[patient, idx_train] += paths_dist[-1, idx_train]
    return distance_matrix


def to_centers_distance_matrix(distance, data_train, data_test, w, make_graph, nn_numbers, centers, distance_matrix,
                               dist_vector_bool=True):
    return distance_matrix[:,centers]


def to_mean_distance_matrix(distance, data_train, data_test, clusters, w, make_graph, nn_numbers,
                            dist_vector_bool=True,dist_bool=True):
    max_path = 10 ** 2 * np.max(data_train, axis=None)
    distances = ["pearson", "spearman", "kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs",
                 "kendall_abs"]
    distances += ["learned_" + i for i in distances] + ["learned"]
    idx_aux = len(w)
    if make_graph:
        idx_aux = -1
    clusters_names = np.unique(clusters)
    methods = [5, 6, 7, 9, 10, 11, 12, 13, 5, 6, 7, 9, 10, 11, 12, 13, 0]
    if dist_bool:
        dists_names = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
                       'chebyshev', 'matching', 'yule', 'braycurtis', 'dice', 'kulsinski', 'russellrao']
        dists_vectors = [lambda x, y: pdist(np.array([x, y]), i)[0] for i in dists_names]
    else:
        dists_names = []
        dists_vectors = []
    if dist_vector_bool:
        correlation_names = [pearsonr, spearmanr, kendalltau]
        dists_vectors += [lambda x, y: 1 - corre(x, y)[0] for corre in correlation_names]
    dists = [lambda x, y: (x - y) ** 2 for _ in range(np.shape(data_train)[1])]
    means = np.zeros((len(clusters_names), data_test.shape[-1]))
    for k in range(len(clusters_names)):
        means[k] = np.mean(data_train[[i for i in range(len(clusters)) if clusters[i] == clusters_names[k]], :], axis=0)
    distance_matrix = np.zeros((np.shape(data_test)[0], len(clusters_names)))
    if distances[distance] != "learned":
        data = np.concatenate((data_test, means))
        distance_matrix = calculator(methods[distance], "lancet", 200, (w[:idx_aux - len(dists_vectors)] * data).T)
        distance_matrix = distance_matrix[:np.shape(data_test)[0], :]
        distance_matrix = distance_matrix[:, -np.shape(means)[0]:]
    if make_graph:
        print(np.shape(means[0]), np.shape(means[1]), len(clusters_names), np.shape(data_test)[-1])
        aux_distance_matrix = np.array(
            [[np.sum(dists[i](means[k][i], means[l][i]) for i in range(np.shape(data_train)[-1])) + np.sum(
                [dist(means[k], means[l]) for dist in
                 dists_vectors]) for l in
              range(len(clusters_names))] for k in
             range(len(clusters_names))])
        aux_distance_train_test = np.array(
            [[np.sum(dists[i](data_test[k, i], means[l][i]) for i in range(np.shape(data_test)[-1])) + np.sum(
                [dist(data_test[k, :], means[l]) for dist in
                 dists_vectors]) for l in
              range(len(clusters_names))] for k in
             range(np.shape(data_test)[0])])
    for patient in range(data_test.shape[0]):
        for k in range(len(clusters_names)):
            data = np.zeros((2, np.shape(data_test)[1]))
            data[0, :] = data_test[patient, :]
            data[1, :] = means[k, :]
            if distances[distance] == "learned":
                distance_matrix[patient, k] = np.sum([w[i] * dists[i](data[0, i], data[1, i]) for i in
                                                      range(np.shape(data)[1])] + [w[i + np.shape(data)[1]] *
                                                                                   dists_vectors[i](data[0, :],
                                                                                                    data[1, :]) for i
                                                                                   in
                                                                                   range(len(dists_vectors))])
            if make_graph:
                graph = graph_nn.graph_building(
                    np.concatenate((np.concatenate(
                        (aux_distance_matrix, aux_distance_train_test[patient, :].reshape(-1, 1)), axis=1),
                                    np.concatenate((aux_distance_train_test[patient, :], [0])).reshape(1, -1)), axis=0),
                    nn_numbers)
                paths_dist = graph_nn.path_finding(graph)
                paths_dist[paths_dist == np.inf] = max_path
                distance_matrix[patient, k] += w[-1] * paths_dist[-1, k]
    return distance_matrix


def clustering(data, distance, algo, ground, w, make_graph, nn_numbers, dist_vector_bool=True):
    n = len(np.unique(ground))

    algos = ["lp-stab", "kmeans", "average", "complete", "spectral", "kmedoids", "ground"]
    algos_func = ["lp", skclust.KMeans(n_clusters=n),
                  skclust.AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage="average"),
                  skclust.AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage="complete"),
                  skclust.SpectralClustering(n_clusters=n,
                                             assign_labels="discretize", random_state=82,
                                             affinity="precomputed"), kmedoid]
    if algos[algo] != "ground":
        distance_matrix = to_distance_matrix(distance, data, data, w, make_graph, nn_numbers,
                                             [0 for _ in range(np.shape(data)[0])], dist_vector_bool=dist_vector_bool)
        print(np.unique(distance_matrix, return_counts=True), np.shape(distance_matrix))
    print(algos[algo])
    medoids = []
    clusters = []
    if algos[algo] == "lp-stab":
        counter = 0
        penalty = 1.0
        step = 1.0
        num_clust = n + 1
        old_num_clust = n
        while counter < 1000 and num_clust != n and old_num_clust != num_clust:
            print("penalty", penalty)
            old_num_clust = num_clust
            step /= 10
            direction = 2
            while num_clust != n and direction != 0:
                medoids = []
                try:
                    # print(counter)
                    clusters, medoids = lp_stability.lp_stab(distance_matrix, penalty,
                                                             'distance_learning_penalty_' + str(
                                                                 penalty))
                except Exception as e:
                    print("Exception", e)
                    counter = 100
                    print(penalty)
                    break
                if len(medoids) == 0:
                    counter = 10000
                    print(penalty)
                    break
                num_clust = len(np.unique(medoids))
                if num_clust > n:
                    if direction == 2:
                        direction = 1
                    else:
                        direction = (direction + 1) / 2
                    penalty += step
                else:
                    if direction == 2:
                        direction = -1
                    else:
                        direction = (direction - 1) / 2
                    penalty -= step
                counter += 1
        print(num_clust)
    elif algos[algo] == "spectral":
        clusters = algos_func[algo].fit(np.exp(-distance_matrix / distance_matrix.std())).labels_
    elif algos[algo] == "kmedoids":
        clusters, medoids = kmedoid(distance_matrix, n)
    else:
        if algos[algo] == "ground":
            clusters = ground
        else:
            if algos[algo] != "kmeans" or distances[distance] == "euclidean":
                clusters = algos_func[algo].fit(distance_matrix).labels_
        clusters_indexes = [[i for i in range(len(clusters)) if clusters[i] == k] for k in np.unique(clusters)]
        if algos[algo] != "ground":
            medoids = [np.argmin(np.sum(distance_matrix[np.ix_(clusters_indexes[k], clusters_indexes[k])], axis=0)) for k in
                   range(len(np.unique(clusters)))]
    return clusters, medoids


def aggregate_dist(dist, cluster_index, pred_index, medoid, linkage):
    if linkage == "Average":
        return np.mean(dist[pred_index, cluster_index])
    elif linkage == "Min":
        return np.min(dist[pred_index, cluster_index])
    elif linkage == "Min Max":
        return np.max(dist[pred_index, cluster_index])
    elif linkage == "Min center":
        return np.min(dist[pred_index, medoid])
    elif linkage == "Sum":
        return np.sum(dist[pred_index, cluster_index]) / len(cluster_index)
    print("Undefined Linkage")
    exit(1)


def neigbhor_fit(cluster, ground):
    predictions = []
    classes = np.unique(ground)
    clusters_id = np.unique(cluster)
    print("cluster", cluster)
    clusters_indexes = [[i for i in range(len(cluster)) if cluster[i] == k] for k in clusters_id]
    score = []
    for k in classes:
        score.append(np.array([sum(
            [1 for i in range(len(ground)) if ground[i] == k and i in clusters_indexes[l]]) / len(clusters_indexes[l])
                               for l in classes]))
    score = np.array(score)
    print(list(ground).count(0), list(ground).count(1))
    print("score", score)
    # pb = perfect coupling in bipartite graph
    # here 2 classes ex
    for i in range(len(classes)):
        class_i, clust_i = np.unravel_index(np.argmax(score), score.shape)
        score[class_i] = [0 for _ in range(score.shape[1])]
        cluster[clusters_indexes[clust_i]] = class_i
    return cluster


def neigbhor_predict(cluster, dist_to_pred, dist_to_mean, dist_to_centers, linkage, medoids, simi):
    predictions = []
    classes = np.unique(cluster)
    clusters_indexes = [[i for i in range(len(cluster)) if cluster[i] == k] for k in classes]
    medoids = [i for k in classes for i in medoids if cluster[i] == k]
    ope = np.argmin
    if simi:
        ope = np.argmax
    if linkage != "Min to Mean" and linkage != "Min to Centers":
        for i in range(np.shape(dist_to_pred)[0]):
            scores = []
            for k in range(len(classes)):
                scores.append(aggregate_dist(dist_to_pred, clusters_indexes[k], i, medoids[k], linkage))
            predictions.append(classes[ope(scores)])
    elif linkage != "Min to Mean":
        predictions = ope(dist_to_mean, axis=1)
    else:
        predictions = ope(dist_to_centers, axis=1)
    return predictions


def KNN(cluster, dist_to_pred, linkage, k, weight_knn):
    predictions = []
    classes = np.unique(cluster)
    # clusters_indexes = [[i for i in range(len(cluster)) if cluster[i] == class_k] for class_k in classes]
    for i in range(np.shape(dist_to_pred)[0]):
        closest = np.argsort(dist_to_pred[i, :])[:k]
        aux_classes = np.unique([cluster[j] for j in closest])
        if weight_knn == 0:
            aux_votes = [len([1 for j in closest if cluster[j] == class_i]) for class_i in aux_classes]
        else:
            aux_votes = [sum([1 / dist_to_pred[i, j] for j in closest if cluster[j] == class_i]) for class_i in
                         aux_classes]
        predictions.append(aux_classes[np.argmax(aux_votes)])
    return (predictions)


def classify(data_train, data_test, clustering_train, clustering_test, algo, distance, w, ensemble_thres, k,
             weight_knn, make_graph, nn_numbers, centers, dist_name, dist_vector_bool=True, distances_list=None,indexes_set=[0,0]):
    linkages = ["Average", "Min", "Min Max", "Min center", "Sum", "Min to Mean", "Min to Centers", "Ensemble", "KNN"]
    """clusters, medoids = clustering(data_train, distance, algo, clustering_train, w, make_graph, nn_numbers,
                                   dist_vector_bool=dist_vector_bool)"""
    clusters = clustering_train
    medoids = centers
    indexes = [i for i in range(len(clustering_train) + (1 - indexes_set[0]) * len(clustering_test))]
    indexes += [i for i in range(-indexes_set[0]*len(clustering_test),0)]
    if not "learned" in dist_name:
        centers = medoids
    if distances_list == None:
        distance_matrix = to_distance_matrix(distance, data_train, np.concatenate((data_train, data_test), axis=0), w,
                                             make_graph,
                                             nn_numbers, clusters, dist_vector_bool=dist_vector_bool)
        distance_to_mean = to_mean_distance_matrix(distance, data_train,
                                                   np.concatenate((data_train, data_test), axis=0),
                                                   clusters, w, make_graph, nn_numbers,
                                                   dist_vector_bool=dist_vector_bool)
        distance_to_centers = to_centers_distance_matrix(distance, data_train,
                                                         np.concatenate((data_train, data_test), axis=0),
                                                         w, make_graph, nn_numbers, centers,distance_matrix,
                                                         dist_vector_bool=dist_vector_bool)
    else:
        distance_matrix = distances_list[0]
        distance_to_mean = distances_list[1]
        distance_to_centers = distances_list[2]
        distance_matrix = distance_matrix[indexes, :]
        distance_to_mean = distance_to_mean[indexes, :]
        distance_to_centers = distance_to_centers[indexes, :]
    if len(clusters) == 0:
        return ([], [], [])
    if algo != -1:
        clusters = neigbhor_fit(clusters, clustering_train)


    results_train = [[[] for _ in range(5)] for _ in range(len(linkages))]
    results_test = [[[] for _ in range(5)] for _ in range(len(linkages))]
    clusters_train_ensemble = []
    clusters_test_ensemble = []
    confusion = []
    for linkage in range(len(linkages)):
        simi = False
        print(algo)
        print(np.shape(distance_matrix[len(clustering_train):, :len(clustering_train)]), len(clustering_train),
              len(clustering_test))
        print(np.shape(distance_to_mean), np.shape(distance_to_centers))
        if linkages[linkage] == "KNN":
            print("KNN")
            y_pred_train = KNN(clusters, distance_matrix[:len(clustering_train), :len(clustering_train)],
                               linkages[linkage], k, weight_knn)
            y_pred_test = KNN(clusters, distance_matrix[len(clustering_train):, :len(clustering_train)],
                              linkages[linkage], k, weight_knn)
        elif linkages[linkage] != "Ensemble":
            y_pred_train = neigbhor_predict(clusters, distance_matrix[:len(clustering_train), :],
                                            distance_to_mean[:len(clustering_train), :],
                                            distance_to_centers[:len(clustering_train), :],
                                            linkages[linkage], medoids, simi)
            print(len(clustering_train), len(y_pred_train), np.shape(distance_matrix))
            if metrics.balanced_accuracy_score(clustering_train, y_pred_train) < 1 / len(np.unique(clustering_train)):
                simi = True
                y_pred_train = neigbhor_predict(clusters,
                                                distance_matrix[:len(clustering_train), :],
                                                distance_to_mean[:len(clustering_train), :],
                                                distance_to_centers[:len(clustering_train), :],
                                                linkages[linkage], medoids, simi)
            y_pred_test = neigbhor_predict(clusters, distance_matrix[len(clustering_train):, :],
                                           distance_to_mean[len(clustering_train):, :],
                                           distance_to_centers[len(clustering_train):, :],
                                           linkages[linkage], medoids, simi)
            if metrics.balanced_accuracy_score(clustering_train, y_pred_train) > ensemble_thres:
                clusters_train_ensemble.append(y_pred_train)
                clusters_test_ensemble.append(y_pred_test)
        else:
            if len(clusters_train_ensemble) > 0:
                y_pred_train = [int(
                    np.sum([clusters_train_ensemble[j][i] for j in range(len(clusters_train_ensemble))]) / len(
                        clusters_train_ensemble)) for i in range(len(clusters_train_ensemble[0]))]
                y_pred_test = [
                    int(np.sum([clusters_test_ensemble[j][i] for j in range(len(clusters_test_ensemble))]) / len(
                        clusters_test_ensemble)) for i in range(len(clusters_test_ensemble[0]))]
            else:
                results_train.append([[0] for _ in range(5)])
                results_test.append([[0] for _ in range(5)])
                confusion.append(np.array([[0 for _ in range(len(np.unique(clustering_test)))] for _ in
                                           range(len(np.unique(clustering_test)))]))
        acc = metrics.balanced_accuracy_score(clustering_train, y_pred_train)

        prec = metrics.precision_score(clustering_train, y_pred_train, average='weighted')
        rec = metrics.recall_score(clustering_train, y_pred_train, average='weighted')

        """spec = len([1 for j in range(len(clustering_train))
                    if (clustering_train[j] != 1 and y_pred_train[j] != 1)]) / (
                       len(clustering_train) - np.sum(clustering_train))"""
        spec = (np.sum([list(clustering_train).count(i) * len([1 for j in range(len(clustering_train))
                                                               if (clustering_train[j] != i and y_pred_train[
                j] != i)]) / len([1 for j in clustering_train
                                  if j != i]) for i in np.unique(clustering_train)])) / len(clustering_train)

        vec = [0, acc, prec, rec, spec]
        results_train[linkage][0].append(0)
        for i in range(1, len(vec)):
            results_train[linkage][i].append(vec[i])

        acc = metrics.balanced_accuracy_score(clustering_test, y_pred_test)

        prec = metrics.precision_score(clustering_test, y_pred_test, average='weighted')
        rec = metrics.recall_score(clustering_test, y_pred_test, average='weighted')
        confusion.append(metrics.confusion_matrix(clustering_test, y_pred_test))
        spec = (np.sum([list(clustering_test).count(i) * len([1 for j in range(len(clustering_test))
                                                              if
                                                              (clustering_test[j] != i and y_pred_test[j] != i)]) / len(
            [1 for j in clustering_test
             if j != i]) for i in np.unique(clustering_test)])) / len(clustering_test)
        vec = [0, acc, prec, rec, spec]
        results_test[linkage][0].append(0)
        for i in range(1, len(vec)):
            results_test[linkage][i].append(vec[i])
    return results_train, results_test, confusion


def writer(f, f_conf, ground, list_res, conf, conf_test, set_i, types, list_res_cat, conf_cat):
    versions = ["Average", "Min", "Min Max", "Min center", "Sum", "Min to Mean", "Min to Centers" "Ensemble", "KNN"]
    set_res = ["Training", "Validation", "Test"]
    line = ["Classifier", "Balanced Accuracy", "Weighted Precision", "Weighted Sensitivity", "Weighted Specificity"]
    f.write(str(set_i) + "\n")
    f.write(str(types[0][set_i][0]) + ",," + str(
        int(types[0][set_i][1]) + int(types[1][set_i][1]) + int(types[2][set_i][1])) + "\n")

    f.write("n classes" + str(len(np.unique(ground[0][set_i]))) + "\n")
    f.write("Train" + str(len(ground[0][set_i])) + "\n")
    f.write("Val" + str(len(ground[1][set_i])) + "\n")
    f.write("Test" + str(len(ground[2][set_i])) + "\n")
    f.write("Train Val\n")
    if sum([len(list_res[i]) != 0 for i in range(len(list_res))]) != len(list_res):
        print(list_res)
        f.write("ERROR \n")
        return
    for i in line:
        if i != "Classifier":
            f.write(i + ",,,")
        else:
            f.write(i + ",")
    f.write("\n ,")
    for i in range(1, len(line)):
        for j in range(len(list_res)):
            f.write(set_res[j] + ",")
    f.write("\n")
    for version_cl in range(len(versions)):
        f.write(versions[version_cl] + ",")
        for i in range(1, len(line)):
            for j in range(len(list_res)):
                if len(list_res[j][version_cl][i]) > 0:
                    f.write(str(round(np.mean(list_res[j][version_cl][i]), 2)) + "+/-" + str(
                        round(np.std(list_res[j][version_cl][i]), 2)) + ",")
                else:
                    f.write("NA,")
        f.write("\n")

    f.write("\n\n")
    f_conf.write("Train - Val\n")
    for version in range(len(versions)):
        f_conf.write(versions[version] + "\n")
        print(conf)
        aux = np.array(
            [[conf[version][i, j] for j in range(np.shape(conf[version])[1])] for i in
             range(np.shape(conf[version])[0])])

        f_conf.write(
            "\n".join([",".join([str(round(aux[i, j], 2)) for j in range(np.shape(conf[version])[1])]) + ",,," +
                       ",".join([str(
                           round(aux[i, j] * 100 / np.sum([aux[i, j2] for j2 in range(np.shape(conf[version])[1])]),
                                 2)) for j in range(np.shape(conf[version])[1])]) for i in
                       range(np.shape(conf[version])[0])]))

        f_conf.write("\n")
    f_conf.write("Test\n")
    for version in range(len(versions)):
        f_conf.write(versions[version] + "\n")
        aux = np.array(
            [[conf_test[version][i, j] for j in range(np.shape(conf_test[version])[1])] for i in
             range(np.shape(conf_test[version])[0])])

        f_conf.write(
            "\n".join(
                [",".join([str(round(aux[i, j], 2)) for j in range(np.shape(conf_test[version])[1])]) + ",,," +
                 ",".join([str(
                     round(aux[i, j] * 100 / np.sum([aux[i, j2] for j2 in range(np.shape(conf_test[version])[1])]),
                           2)) for j in range(np.shape(conf_test[version])[1])]) for i in
                 range(np.shape(conf_test[version])[0])]))

        f_conf.write("\n")

    f_conf.write("\n\n\n")
    f.write("\n\n\n")
    f.write("Re-train - test\n")

    set_res = ["Training", "Test"]
    list_res = list_res_cat
    for i in line:
        if i != "Classifier":
            f.write(i + ",,")
        else:
            f.write(i + ",")
    f.write("\n ,")
    for i in range(1, len(line)):
        for j in range(len(list_res)):
            f.write(set_res[j] + ",")
    f.write("\n")
    for version_cl in range(len(versions)):
        f.write(versions[version_cl] + ",")
        for i in range(1, len(line)):
            for j in range(len(list_res)):
                if len(list_res[j][version_cl][i]) > 0:
                    f.write(str(round(np.mean(list_res[j][version_cl][i]), 2)) + "+/-" + str(
                        round(np.std(list_res[j][version_cl][i]), 2)) + ",")
                else:
                    f.write("NA,")
        f.write("\n")

    f_conf.write("Re-Train Test\n")
    for version in range(len(versions)):
        f_conf.write(versions[version] + "\n")
        aux = np.array(
            [[conf_cat[version][i, j] for j in range(np.shape(conf_cat[version])[1])] for i in
             range(np.shape(conf_cat[version])[0])])

        f_conf.write(
            "\n".join([",".join([str(round(aux[i, j], 2)) for j in range(np.shape(conf_cat[version])[1])]) + ",,," +
                       ",".join([str(
                           round(aux[i, j] * 100 / np.sum([aux[i, j2] for j2 in range(np.shape(conf_cat[version])[1])]),
                                 2)) for j in range(np.shape(conf_cat[version])[1])]) for i in
                       range(np.shape(conf_cat[version])[0])]))
        f_conf.write("\n")
    f_conf.write("\n")
    f.write("\n")


if __name__ == "__main__":
    genes_set = sys.argv[1]
    version = sys.argv[2]
    separation = int(sys.argv[3])
    ground_col = sys.argv[4]
    T = int(sys.argv[5])
    conv_criterion = float(sys.argv[6])
    max_it = int(sys.argv[7])
    types_selec = sys.argv[8]
    alpha = float(sys.argv[9])
    beta = float(sys.argv[10])
    tau = float(sys.argv[11])
    seed = int(sys.argv[12])
    use_centers = int(sys.argv[13])
    weighted = int(sys.argv[14])
    weighted_list = ["unweighted", "correctly_weighted2"]
    weighted_list_save = ["unweighted", "weighted2"]
    update = sys.argv[15]
    cross_val = int(sys.argv[16])
    algo = int(sys.argv[17])
    dist = int(sys.argv[18])
    k = int(sys.argv[19])
    weight_knn = int(sys.argv[20])
    make_graph = bool(int(sys.argv[21]))
    nn_numbers = int(sys.argv[22])
    regu = int(sys.argv[23])
    regu_methods = ["lasso", "ridge", "elasticnet"]

    weight_knn_list = ["unweighted", "weighted"]
    ensemble_thres = 0.6 if ground_col == "Evolution" else 0.75
    # sbatch --export=T=5,conv_criterion=0.001,max_it=500,alpha=1.0,beta=4.0,tau=1.0,update=norm,dist=-2,k=20 covid_learning.sh
    # python covid_classify.py lung_disease_left_right_heart clinical_full_all2 0 Evolution 5 0.001 500 all 1 4 .5 10 0 1 norm 1 -1 9 5 0
    file_folder = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.dirname(file_folder) + "/data/covid/"

    algos = ["lp-stab", "kmeans", "average", "complete", "spectral", "kmedoids", "ground"]

    distances = ["pearson", "spearman", "kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs",
                 "kendall_abs"]
    distances += ["learned_" + i for i in distances] + ["learned"]

    genes = np.load(data_folder + "preprocess/" + "_".join(
        ["corre", str(ground_col),
         str(types_selec), "seed", str(seed),
         str(use_centers), "cross_val" + str(cross_val)]) + "_train_val_test.npy", allow_pickle=True)
    types = np.load(data_folder + "preprocess/" + "_".join(
        ["types", "correct", "corre",
         str(types_selec), "seed", str(seed),
         str(use_centers), "cross_val" + str(cross_val)]) + "_train_val_test.npy", allow_pickle=True)
    ground = []
    features = []
    ext = ["train", "val", "test"]
    for i in ext:
        ground.append(np.load(
            data_folder + "preprocess/covid_clustering_list_types_" + "_".join(
                ["corre", str(ground_col), str(types_selec), "seed",
                 str(seed),
                 str(use_centers), "cross_val" + str(cross_val), i]) + ".npy", allow_pickle=True))

        features.append(np.load(data_folder + "preprocess/" + "_".join(
            ["corre", str(ground_col), str(types_selec), "seed", str(seed),
             str(use_centers), "cross_val" + str(cross_val),
             "classif_features", i + ".npy"]), allow_pickle=True))

    if "learned" in distances[dist]:
        w = np.load(data_folder + "preprocess/" + "_".join(
            ["corre", str(ground_col), str(types_selec), "seed", str(seed),
             str(use_centers),
             "cross_val" + str(cross_val), weighted_list[weighted], update,
             str(conv_criterion), str(max_it), str(T), str(alpha), str(beta), str(tau), "graph", str(make_graph),
             str(nn_numbers), regu_methods[regu], "w"]) + ".npy")
        centers = np.load(data_folder + "preprocess/" + "_".join(
            ["corre", str(ground_col), str(types_selec), "seed", str(seed),
             str(use_centers),
             "cross_val" + str(cross_val), weighted_list[weighted], update,
             str(conv_criterion), str(max_it), str(T), str(alpha), str(beta), str(tau), "graph", str(make_graph),
             str(nn_numbers), regu_methods[regu], "centers"]) + ".npy", allow_pickle=True)
    else:
        w = [1 for _ in range(np.shape(features[0][0])[-1] + int(make_graph) + 14)]
        centers = [[] for _ in range(len(ground[0]))]
    # w = (w - np.min(w)) / (np.max(w) - np.min(w))
    # w = (w-np.mean(w)) / (np.max(w) - np.min(w))
    # w = (w-np.mean(w)) / np.std(w)
    f = open(data_folder + "/results/classif/" + "_".join(
        ["corre", str(types_selec), "seed", str(seed), str(use_centers),
         "cross_val" + str(cross_val), weighted_list_save[weighted], update,
         str(conv_criterion), str(max_it), str(T), str(alpha), str(beta), str(tau), str(algos[algo]),
         str(distances[dist]), "ensemble", str(ensemble_thres), "KNN", weight_knn_list[weight_knn], "graph",
         str(make_graph),
         str(nn_numbers), regu_methods[regu], str(k) + ".csv"]),
             "w")
    f_conf = open(data_folder + "/results/classif/Confusion_" + "_".join(
        ["corre", str(types_selec), "seed", str(seed), str(use_centers),
         "cross_val" + str(cross_val),
         str(conv_criterion), str(max_it), str(T), str(alpha), str(beta), str(tau), str(algos[algo]),
         str(distances[dist]), "ensemble", str(ensemble_thres), "KNN", weight_knn_list[weight_knn], "graph",
         str(make_graph),
         str(nn_numbers), regu_methods[regu], str(k) + ".csv"]),
                  "w")

    results_train_list = []
    results_val_list = []
    results_test_list = []
    results_train_cat_list = []
    results_test_cat_list = []
    for set_i in range(len(ground[0])):
        print(np.shape(ground[0][set_i]), np.shape(ground[1][set_i]))
        print(np.shape(features[0][set_i]), np.shape(features[1][set_i]))
        print("centers", centers[set_i])
        distances_list = None
        if algo == -1:
            distances_list = []
            distances_list.append(to_distance_matrix(dist, features[0][set_i], np.concatenate((features[0][set_i], features[1][set_i],features[2][set_i]), axis=0), w,
                                                 make_graph,
                                                 nn_numbers, ground[0][set_i], dist_vector_bool=True))
            distances_list.append(to_mean_distance_matrix(dist, features[0][set_i],
                                                       np.concatenate((features[0][set_i], features[1][set_i],features[2][set_i]), axis=0),
                                                       ground[0][set_i], w, make_graph, nn_numbers,
                                                       dist_vector_bool=True))
            distances_list.append(to_centers_distance_matrix(dist, features[0][set_i],
                                                         np.concatenate((features[0][set_i], features[1][set_i],features[2][set_i]), axis=0),
                                                         w, make_graph, nn_numbers, centers[set_i], distances_list[0],
                                                         dist_vector_bool=True))

        results_train, results_val, conf = classify(
            features[0][set_i], features[1][set_i],
            ground[0][set_i], ground[1][set_i], algo, dist, w, ensemble_thres, k, weight_knn, make_graph, nn_numbers,
            centers[set_i], distances[dist], indexes_set =[0,len(ground[2][set_i])])

        _, results_test, conf_test = classify(features[0][set_i], features[2][set_i],
                                              ground[0][set_i], ground[2][set_i], algo, dist, w, ensemble_thres, k,
                                              weight_knn, make_graph, nn_numbers, centers[set_i], distances[dist], indexes_set = [1, len(ground[1][set_i])])
        list_res = [results_train, results_val, results_test]
        results_train_list.append(results_train)
        results_val_list.append(results_val)
        results_test_list.append(results_test)

        """aux_features = np.concatenate((features[0][set_i], features[1][set_i]), axis=0)
        results_train_cat, results_test_cat, conf_cat = classify(aux_features,
                                                                 features[-1][set_i], np.array(
                list(ground[0][set_i]) + list(ground[1][set_i])), ground[-1][set_i], algo, dist, w, ensemble_thres, k,
                                                                 weight_knn, make_graph, nn_numbers, centers[set_i],
                                                                 distances[dist])"""
        results_train_cat = results_train
        results_test_cat = results_test
        conf_cat = conf_test
        results_train_cat_list.append(results_train_cat)
        results_test_cat_list.append(results_test_cat)
        list_res_cat = [results_train_cat, results_test_cat]

        writer(f, f_conf, ground, list_res, conf, conf_test, set_i, types, list_res_cat, conf_cat)
    list_res = [results_train_list, results_val_list, results_test_list]
    list_res = [
        [[[list_i[split_i][version][i][0] for split_i in range(len(list_i))] for i in range(len(list_i[0][version]))]
         for version in range(len(list_i[0]))] for list_i in list_res]
    print("average", list_res)
    list_res_cat = [results_train_cat_list, results_test_cat_list]
    list_res_cat = [
        [[[list_i[split_i][version][i][0] for split_i in range(len(list_i))] for i in range(len(list_i[0][version]))]
         for version in range(len(list_i[0]))] for list_i in list_res_cat]
    writer(f, f_conf, ground, list_res, conf, conf_test, -1, types, list_res_cat, conf_cat)
    f_conf.close()
    f.close()
