#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Distance Learning; Higher-Order; Classification;
### Description: <Leverage a learned higher-order distance to perform a prediction on a K-NN basis>
### Input: data, ground truth, learned distance
### Ouput: predictions, assessment of the classification, distance matrix
###################################


import os
import numpy as np
from distance_calculator import calculator
from scipy.spatial.distance import pdist
from sklearn import metrics
import graph
import multiprocessing as mp
from itertools import combinations_with_replacement
from scipy.special import binom
from scipy.stats import kendalltau, pearsonr, spearmanr
from time import time


def parallel_dist_order(g, method, combination, center, w):
    methods = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]

    if w != 0:
        return w * methods[method](g, list(combination), center)
    else:
        return 0


def parallel_dist_aux(p, q, m, data1, data2, has_vector):
    dists = [lambda x, y: (x - y) ** 2 for _ in range(m)]
    dists_vectors = []
    if has_vector:
        dists_names = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
                       'chebyshev', 'matching', 'yule', 'braycurtis', 'dice', 'kulsinski', 'russellrao']
        dists_vectors = [lambda x, y: pdist(np.array([x, y]), i) for i in dists_names]
    if p != q:
        return [dists[i](data1[p, i], data2[q, i]) for i in range(m)] + [dist(data1[p, :], data2[q, :]) for dist in
                                                                         dists_vectors]
    else:
        return [0 for _ in range(m)] + [0 for _ in dists_vectors]


# compute the higher order distance to centers
def to_centers_distance_matrix(distance, data_train, data_test, w, nn_numbers, centers, clusters, order, has_vectors,
                               path, init=[], graph_train=[], cluster_metric=1):
    order += 1
    methods_func = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]
    print(w)
    distances = ["pearson", "spearman", "kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs",
                 "kendall_abs"]
    distances += ["learned_" + i for i in distances] + ["learned"]
    methods = [5, 6, 7, 9, 10, 11, 12, 13, 5, 6, 7, 9, 10, 11, 12, 13, 0]
    dists_vectors = []
    if has_vectors:
        dists_names = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'hamming', 'jaccard',
                       'chebyshev', 'matching', 'yule', 'braycurtis', 'dice', 'kulsinski', 'russellrao']
        dists_vectors = [lambda x, y: pdist(np.array([x, y]), i)[0] for i in dists_names]
        if len(init) > 0:
            correlation_names = [pearsonr, spearmanr, kendalltau]
            dists_vectors += [lambda x, y: 1 - corre(x, y)[0] for corre in correlation_names]
    idx_aux = -4
    train_size = data_train.shape[0]
    distance_matrix = np.zeros((np.shape(data_test)[0], len(centers)))
    if distances[distance] != "learned":
        data = np.concatenate((data_train[centers, :], data_test))
        distance_matrix = calculator(methods[distance], "lancet", 200, (w[:idx_aux - len(dists_vectors)] * data).T)
        distance_matrix = distance_matrix[-np.shape(data_test)[0]:, :]
        distance_matrix = distance_matrix[:, :len(centers)]
    print("aux comput")
    aux_time = time()
    n = np.shape(data_train)[0]
    n_2 = np.shape(data_test)[0]
    if not os.path.exists(path + "train.npy"):
        pool = mp.Pool(processes=10)
        aux_distance_train = np.array(pool.starmap(parallel_dist_aux,
                                                   ((p, q, np.shape(data_train)[1], data_train, data_train, has_vectors)
                                                    for p in
                                                    range(n) for q in
                                                    range(n))), dtype=object).reshape(
            (n, n, np.shape(data_train)[1] + len(dists_vectors)))
        pool.close()
        np.save(path + "train", aux_distance_train)
    else:
        aux_distance_train = np.load(path + "train.npy", allow_pickle=True)

    if not os.path.exists(path + "train_test.npy"):
        pool = mp.Pool(processes=10)
        aux_distance_train_test = np.array(pool.starmap(parallel_dist_aux,
                                                        ((p, q, np.shape(data_test)[1], data_test, data_train,
                                                          has_vectors) for
                                                         p in
                                                         range(n_2) for q in
                                                         range(n))), dtype=object).reshape(
            (n_2, n, np.shape(data_test)[1] + len(dists_vectors)))
        print(aux_distance_train_test)
        pool.close()

        np.save(path + "train_test", aux_distance_train_test)
    else:
        aux_distance_train_test = np.load(path + "train_test.npy", allow_pickle=True)

    print("aux distances ok", time() - aux_time)
    if len(init) == 0:
        length = np.shape(data_test)[-1]
        if has_vectors:
            length += len(dists_vectors)
        init = np.array([1.0 for _ in range(length)])

    for patient in range(data_test.shape[0]):
        if graph_train == []:
            g = graph.Graph.from_data(np.concatenate((np.concatenate((np.sum(init * aux_distance_train, axis=-1),
                                                                      np.sum(init *
                                                                             aux_distance_train_test[patient, :],
                                                                             axis=-1).reshape(1, -1)), axis=0),
                                                      np.concatenate((np.sum(init *
                                                                             aux_distance_train_test[patient, :],
                                                                             axis=-1),
                                                                      [0])).reshape(-1, 1)), axis=1) / sum(init),
                                      nn_numbers)
        else:
            g = graph.Graph.add_node(graph_train[0], graph_train[1][patient], graph_train[2], graph_train[3])

        for idx_train in range(len(centers)):
            cluster_k = [i for i in range(len(clusters)) if clusters[i] == clusters[centers[idx_train]]]
            if distances[distance] == "learned":
                distance_matrix[patient, idx_train] = np.sum([
                    w[:idx_aux] * aux_distance_train_test[patient, centers[idx_train]]]
                )
            else:
                distance_matrix[patient, idx_train] += np.sum([
                    w[-1 - idx_aux - dist_i] *
                    dists_vectors[dist_i](data_test[patient, :], data_train[centers[idx_train], :]) for dist_i in
                    range(len(dists_vectors))])
            if w[-1] != 0:
                aux_para = 0
                if cluster_metric > 0:
                    aux_mult = w[-1] / len(cluster_k) ** 2
                    if not os.path.exists(path + "p" + str(patient) + "center" + str(idx_train) + "graph" + ".npy"):
                        time_paths = time()
                        for p2 in cluster_k:
                            for p in [idx_train]:
                                if g.paths[p][p2] != 0:
                                    aux_para += (g.paths[train_size][p] + g.paths[p2][train_size]) / g.paths[p][p2]
                        np.save(path + "p" + str(patient) + "center" + str(idx_train) + "graph", aux_para)
                        print("paths", time() - time_paths)
                    else:
                        aux_para = np.load(path + "p" + str(patient) + "center" + str(idx_train) + "graph" + ".npy",
                                           allow_pickle=True)
                    aux_para = aux_mult * aux_para
                if cluster_metric == -1 or cluster_metric == 2:
                    aux_para += sum(
                        [g.paths[train_size][q] + sum([g.paths[p2][q] for p2 in cluster_k if p2 > q]) for q in cluster_k
                         ]) / np.sum([g.paths[p][p2]
                                      for q in cluster_k for p2 in cluster_k if p2 > q])
                distance_matrix[patient, idx_train] += aux_para
            pool = mp.Pool(processes=10)
            for method_i in range(len(methods_func)):
                if w[-1 - 1 - method_i] != 0:
                    aux_para = 0
                    if cluster_metric > 0:
                        if not os.path.exists(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(
                                method_i) + ".npy"):
                            aux_para += sum(pool.starmap(parallel_dist_order,
                                                         ((g, method_i, list(combination) + [idx_train],
                                                           train_size,
                                                           1 / binom(len(cluster_k) - 2 + current_order - 1 - 2,
                                                                     current_order - 2))
                                                          for current_order in range(2, order) for combination in
                                                          combinations_with_replacement(cluster_k, current_order - 2))))
                            np.save(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(method_i),
                                    aux_para)
                        else:
                            aux_para += np.load(path + "p" + str(patient) + "center" + str(idx_train) + "method" + str(
                                method_i) + ".npy",
                                                allow_pickle=True)
                    if cluster_metric == -1 or cluster_metric == 2:
                        aux_para += parallel_dist_order(g, method_i, cluster_k, train_size, 1)
                    distance_matrix[patient, idx_train] += w[-1 - 1 - method_i] * aux_para
            pool.close()
    return distance_matrix


def aggregate_dist(dist, cluster_index, pred_index, medoid, linkage):
    if linkage == "Average":
        return np.mean(dist[pred_index, cluster_index])
    elif linkage == "Min":
        return np.min(dist[pred_index, cluster_index])
    elif linkage == "Min Max":
        return np.max(dist[pred_index, cluster_index])
    elif linkage == "Min center":
        return np.min(dist[pred_index, medoid])
    print("Undefined Linkage")
    exit(1)


def neigbhor_predict(cluster, dist_to_centers, linkage, simi):
    predictions = []
    classes = np.unique(cluster)
    clusters_indexes = [[i for i in range(len(cluster)) if cluster[i] == k] for k in classes]
    ope = np.argmin
    if simi:
        ope = np.argmax
    predictions = ope(dist_to_centers, axis=1)
    return predictions


def KNN(cluster, dist_to_pred, k):
    predictions = []
    for i in range(np.shape(dist_to_pred)[0]):
        closest = np.argsort(dist_to_pred[i, :])[:k]
        aux_classes = np.unique([cluster[j] for j in closest])
        aux_votes = [len([1 for j in closest if cluster[j] == class_i]) for class_i in aux_classes]
        predictions.append(aux_classes[np.argmax(aux_votes)])
    return predictions


def assess(clustering, y_pred):
    acc = metrics.balanced_accuracy_score(clustering, y_pred)
    prec = metrics.precision_score(clustering, y_pred, average='weighted')
    rec = metrics.recall_score(clustering, y_pred, average='weighted')
    spec = (np.sum([list(clustering).count(i) * len([1 for j in range(len(clustering))
                                                     if (clustering[j] != i and y_pred[
            j] != i)]) / len([1 for j in clustering
                              if j != i]) for i in np.unique(clustering)])) / len(clustering)

    return [acc, prec, rec, spec]


def classify(clustering_train, clustering_test, indexes_set, distances_list=None):
    linkages = ["Min to Centers"]
    clusters = clustering_train

    indexes = [i for i in range(len(clustering_train) + (1 - indexes_set[0]) * len(clustering_test))]
    indexes += [i for i in range(-indexes_set[0] * len(clustering_test), 0)]

    distance_to_centers = distances_list[indexes, :]

    results_train = []
    results_test = []
    confusion = []
    for linkage in range(len(linkages)):
        y_pred_train = neigbhor_predict(clusters,
                                        distance_to_centers[:len(clustering_train), :],
                                        linkages[linkage])
        y_pred_test = neigbhor_predict(clusters,
                                       distance_to_centers[len(clustering_train):, :],
                                       linkages[linkage])
        results_train.append(assess(clustering_train, y_pred_train))
        results_test.append(assess(clustering_test, y_pred_test))
        confusion.append(metrics.confusion_matrix(clustering_test, y_pred_test))
    return results_train, results_test, confusion
