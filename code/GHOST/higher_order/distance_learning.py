# -*- coding: utf-8 -*-

### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Distance Learning; Higher-Order;
### Description: <Learn a higher-order distance through conditional random field energy minimization>
### Input: data, ground truth, graph structure
### Ouput: learned distance
###################################

import os, sys
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
import graph
import multiprocessing as mp
from itertools import combinations_with_replacement
from scipy.special import binom


# Function to compute the higher order graph metrics in parallel
# add here new methods you would want to experiment and implement them in graph.py
def parallel_dist_order(g, method, combination, center, w):
    methods = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]

    if w != 0:
        return w * methods[method](g, list(combination), center)
    else:
        return 0


# Check for convergence
# old_w stores the list of weights in a loop of T iterations
# conv_count contains the number of consecutive iterations for which this function considered convergence reached
# C is a coefficient weighting the convergence criterion, the lower the easier to satisfy the criterion
# There is convergence if for all the weights the variance is lower than C x the average + C^2 x average over all weights
def convergence(old_w, conv_count, C):
    l = [[w[i] for w in old_w] for i in range(len(old_w[0]))]
    std_l = [np.std(l[i]) for i in range(len(l))]
    crit_l = [C * abs(np.mean(l[i])) + C ** 2 * abs(np.mean([l[j] for j in range(len(l)) if j != i]))
              for i in range(len(l))]
    print("convergence criterion per dimension", [std_l[i] < crit_l[i] for i in range(len(l))])
    if sum([std_l[i] > crit_l[i] for i in range(len(l))]) < 1:
        conv_count += 1
    else:
        conv_count = 0
    return conv_count


# Find the cluster index of the sample idx based on the list clusters containing a list of clusters
# A cluster being represented by its list of sample indexes
def find_clust(idx, clusters):
    for clust_num in range(len(clusters)):
        if idx in clusters[clust_num]:
            return clust_num


# Implementation of the higher-order distance learning algorithm
# clustering_list: ground-truth, list of labels for the samples for the K training sets
# feature_list: tensor containing for each pair of samples their distance according to each metric we are learning on
# T: number of projected subgradient rounds
# C: coefficient for the convergence criterion, cf convergence function
# max_it: maximum number of iterations in case convergence is not reached
# alpha, beta, tau: coefficients of the constraints and of the regularization, to be tuned
# speed: coefficient influencing the speed of convergence, weight the updates
# update_name: name of the projection to be update at each iteration, in the higher-order case, the weights need to be positive
# nn_numbers: size of the neighborhood to consider when builgind graphs based on the neighborhood
# order: order to consider for the higher-order
# cpu_nb_slaves: number of cpus for available for parallelization
# regu: name of the regularizationmethod, implemented methods are "lasso", "ridge" and "elasticnet"
# init: possible pre-initialization of the weights for the second-order metrics (e.g. by first applying the second-order framework), empty list for no initialization
# graphs: list of the graphs to consider for each of the K training sets, if empty the K-nearest neighbor method is used
# cluster_metric : 0: no higher order, 1: higher-order of order "order" only, -1: cluster metric only, 2: higher-order and cluster metric
def learn(clustering_list, features_list, T, C, max_it, alpha=1, beta=1, speed=1, tau=0.5,
          update_name="id", nn_numbers=5, order=5, cpu_nb_slaves=20, regu="lasso", init=[], graphs=[],
          cluster_metric=0):
    # maximal distance to replace inf in paths length
    max_path = 10 ** 1 * np.max(features_list[0], axis=None)
    # Projection function on a given set of constraints for w update
    update = lambda x: x
    if update_name == "pos":
        update = lambda x: np.array([max(x[i], 0.0) for i in range(len(x))])
    elif update_name == "norm":
        update = lambda x: np.array([abs(x[i]) for i in range(len(x))])
    elif update_name == "thres1":
        update = lambda x: np.array([x[i] if np.abs(x[i]) >= 1 else 0 for i in range(len(x))])
    elif update_name == "min_max":
        update = lambda x: np.array([(x[i] - min(x)) / (max(x) - min(x)) for i in range(len(x))])
    elif update_name == "abs_min_max":
        update = lambda x: np.array(
            [(abs(x[i]) - min(np.abs(x))) / (max(np.abs(x)) - min(np.abs(x))) for i in range(len(x))])
    else:
        print("Undefined update method")
        raise
    n_features = np.shape(features_list[0])[-1]
    """dists3_global = [lambda x, y, z: np.sqrt(
        (z - np.mean([x, y], axis=1)).T * np.cov([x, y]) * (z - np.mean([x, y], axis=1))) if x != y else 0]"""
    # Number of clusters
    K = len(clustering_list)
    # List of lambdas, array for the point wise slaves, lists for the clusters wise slaves
    la_p = [np.zeros((len(clustering_list[k]), len(clustering_list[k]))) for k in range(K)]
    la_C = [np.zeros(len(clustering_list[k])) for k in range(K)]

    # List of graph metrics to consider in addition of the by default path length
    methods = [graph.Graph.clique_order, graph.Graph.eccentricity, graph.Graph.connectivity]
    # Distance weights initialization using or not a pre-initialization
    if len(init) == 0:
        w = np.array([1.0 for _ in range(np.shape(features_list[0])[-1] + len(methods) + 1)])
        init = np.array([1.0 for _ in range(np.shape(features_list[0])[-1])])
    else:
        w = np.array(init + [1.0 for _ in range(len(methods) + 1)])

    if len(graphs) == 0:
        # Graphs construction using K-NN approach
        graphs = [graph.Graph.from_data(np.sum(init * features_list[k], axis=2) / sum(init), nn_numbers) for k in
                  range(K)]
    for g in graphs:
        # Pre-computation of the path lengths
        g.path_finding(max_path)
    # Size of clusters for each clustering in ground truth
    counts = [np.unique(clustering_i, return_counts=True) for clustering_i in clustering_list]
    clusters_size = [dict({(counts_i[0][idx], counts_i[1][idx]) for idx in range(len(counts_i[0]))}) for counts_i in
                     counts]
    # Clustering assignment
    x = [0 for _ in range(K)]
    # beta weight for the importance of having all the points of a ground truth cluster assigned to the same exemplar
    # alpha weight for the importance of having only one exemplar per ground truth cluster
    # tau weight of the regularization of w
    # Subderivative of the considered regularization
    if regu == "lasso":
        delta_J = lambda x: np.array([tau if x_i > 0 else -tau for x_i in x])
    elif regu == "elasticnet":
        delta_J = lambda x: np.array([tau * (1 + 2 * x_i) if x_i > 0 else tau * (-1 + 2 * x_i) for x_i in x])
    elif regu == "ridge":
        delta_J = lambda x: np.array([2 * x_i for x_i in x])
    conv_count = 0
    counter = 1
    idx_update = - len(methods) - 1
    beginning = time.time()
    # Indexes of the samples in each cluster
    clusters_idx = [
        [[idx for idx in range(len(clustering_list[k])) if clustering_list[k][idx] == np.unique(clustering_list[k])[i]]
         for i in range(len(np.unique(clustering_list[k])))] for k in range(K)]
    distance_higher_cluster = []
    # Computation of the higher order part of the energy for a center by cluster, used to compute the optimal x_k
    print("cluster distance computation")
    if cluster_metric != 0:
        pool = mp.Pool(processes=cpu_nb_slaves)
        for k in range(K):
            distance_higher_cluster.append([])
            for q in range(len(clustering_list[k])):
                # We consider only the samples from the cluster of center q
                i_clust = find_clust(q, clusters_idx[k])
                aux = np.array([0. for _ in range(len(methods) + 1)])
                if cluster_metric > 0:  # If higher-order involved
                    # We consider all possible combinations with replacement for each order up to the maximal order "order"
                    # we normalize by the number of combinations with replacement
                    aux += np.array([np.sum(pool.starmap(parallel_dist_order,
                                                         ((graphs[k], method, combination, q,
                                                           1 / binom(len(
                                                               clusters_idx[k][i_clust]) - 2 + current_order - 1 - 1,
                                                                     current_order - 1)) for
                                                          current_order in
                                                          range(2, order + 1) for combination in
                                                          combinations_with_replacement(clusters_idx[k][i_clust],
                                                                                        current_order - 1)))) for method
                                     in
                                     range(len(methods))] + [np.sum([
                        (graphs[k].paths[p][q] + graphs[k].paths[p2][q]) / graphs[k].paths[p][p2]
                        for p in clusters_idx[k][i_clust] for p2 in clusters_idx[k][i_clust] if p2 > p]) / len(
                        clusters_idx[k][i_clust])])
                if cluster_metric == -1 or cluster_metric == 2:  # If the cluster-metric is involved
                    aux += np.array([parallel_dist_order(graphs[k], method, clusters_idx[k][i_clust], q, 1) for
                                     method in range(len(methods))] + [
                                        np.sum([graphs[k].paths[p][q] for p in clusters_idx[k][i_clust]]) / np.sum(
                                            [graphs[k].paths[p][p2]
                                             for p in clusters_idx[k][i_clust] for p2 in clusters_idx[k][i_clust] if
                                             p2 > p])])
                distance_higher_cluster[k].append(aux)
        pool.close()
        distance_higher_cluster = [np.array(distance_higher_cluster[k]) for k in range(K)]
    # Until convergence or max_it is reached
    while counter < max_it and conv_count < 3:
        old_w = [np.copy(w)]
        print("convergence: ", conv_count)
        # Prepare the indexes for the T steps of sgd for each cohort, at each step we discard a different split
        skf = StratifiedKFold(n_splits=T, shuffle=True, random_state=T + counter)
        splits = [[] for k in range(K)]
        for k in range(K):
            for sgd_index, _ in skf.split(np.zeros(len(clustering_list[k])), clustering_list[k]):
                splits[k].append(sgd_index)
        begin_loop = time.time()
        for k in range(K):
            clusters = clustering_list[k]
            n = len(clusters)
            # compute the second-order distance with the current weights
            distance = np.sum(old_w[-1][:idx_update] * features_list[k], axis=2)
            distance += old_w[-1][-1] * np.array([[graphs[k].paths[j][i] for j in range(n)] for i in range(n)])
            # auxiliary variable for the ground truth
            x_k = np.zeros((n, n))
            for i_clust in np.unique(clusters):
                # determine the exemplar of each cluster by considering the sum of distances minimizer
                cluster_i = clusters_idx[k][i_clust]
                distance_i = distance[cluster_i, :]
                distance_i = distance_i[:, cluster_i]

                # computation of the third order sum of distances for cluster i
                # it takes advantage of the symmetry of the distance according to p2 and q
                distance_higher = np.array([np.sum(
                    old_w[-1][-method_i - 1] * distance_higher_cluster[k][q][-method_i - 1] for method_i in
                    range(len(distance_higher_cluster[k][q]))) for q in cluster_i])
                q_i = np.argmin(np.sum(distance_i, axis=1) + distance_higher)
                # assign each point of the cluster to the exemplar
                print("center", k, i_clust, cluster_i[q_i], len(cluster_i))
                for l in cluster_i:
                    x_k[l, cluster_i[q_i]] = 1
            x[k] = x_k
        end_samples = time.time()
        print("End samples", end_samples - begin_loop)
        print("order", order)
        for i in range(T):
            # s_t: decreasing towards 0 function for the update influence, here we consider 1 / nb of iterations
            s_t = 1 / (speed * counter)
            # higher order update with only x
            test_time = time.time()
            if cluster_metric > 0:
                pool = mp.Pool(processes=cpu_nb_slaves)
                for current_order in range(3, order + 1):
                    for method in range(len(methods)):
                        for q in clusters_idx[k][i_clust]:
                            w[idx_update + method + 1] -= s_t * np.array([np.sum(pool.starmap(parallel_dist_order, (
                                (graphs[k], method, combination, q,
                                 1 / binom(len(clusters_idx[k][i_clust]) - 2 + current_order - 1 - 1,
                                           current_order - 1) if np.prod([x[k][p, q] for p in combination]) == 1 else 0)
                                for combination in
                                combinations_with_replacement(range(len(clustering_list[k])), current_order - 1))))])
                    print(time.time() - test_time)
                pool.close()
            if cluster_metric == -1 or cluster_metric == 2:
                for method in range(len(methods)):
                    for q in clusters_idx[k][i_clust]:
                        w[idx_update + method + 1] -= s_t * parallel_dist_order(graphs[k], method,
                                                                                clusters_idx[k][i_clust], q,
                                                                                1 if np.sum([x[k][p, q] for p in
                                                                                             clusters_idx[k][
                                                                                                 i_clust]]) == 1 else 0)
            # second order update with only x
            w[:idx_update] -= np.array([s_t * sum(
                sum(sum(x[k][p, q] * features_list[k][p, q, feature_i] for p in range(len(clustering_list[k])))
                    for q in range(len(clustering_list[k]))) for k in range(K)) for feature_i in range(n_features)])
            la_p_copy = [np.copy(la_p[k]) for k in range(K)]
            la_C_copy = [np.copy(la_C[k]) for k in range(K)]
            w -= s_t * delta_J(old_w[-1])
            for k in range(K):
                clusters = splits[k][i]
                clusters_idx_k = [[i for i in c if i in clusters] for c in clusters_idx[k]]
                n = len(clusters)
                aux = features_list[k][:, clusters, :]
                distance = np.sum(old_w[-1][:idx_update] * aux[clusters, :, :], axis=2)
                w[-1] -= s_t * sum(sum(x[k][p, q] * graphs[k].paths[p][q] for p in clusters)
                                   for q in clusters)
                w[-1] -= s_t * sum(sum(sum(
                    0 if p == p2 else x[k][p, q] * x[k][p2, q] * (graphs[k].paths[p][q] + graphs[k].paths[p2][q]) /
                                      graphs[k].paths[p][p2]
                    for p in clusters) for p2 in clusters)
                                   for q in clusters) / len(clusters)
                distance += old_w[-1][-1] * sum([graphs[k].paths[p][q] for p in clusters for q in clusters])
                print("First update w with graphs")
                # bars have been omitted in the notation here for conciseness' sake
                # u_k(1) in the article multiply by x_pq to have u_k(x_pq) =  distance + beta * size_cluster if p and q belongs to the same cluster else only distance
                u_k = distance + \
                      beta * np.array(
                    [[(clustering_list[k][p] == clustering_list[k][q]) * x[k][p, q] / clusters_size[k][
                        clustering_list[k][p]] for q in clusters] for p in
                     clusters])
                # computation of the minimizers per slaves (point wise and cluster wise)
                print("Beginning Ep")
                begin_Ep = time.time()
                # computation of the higher order sum of distances for all possible combinations_with_replacement of p and q for each order
                distance_higher = np.zeros((n, n, len(methods) + 1))
                distance_cluster = np.zeros((n, len(clusters_idx_k), len(methods) + 1))
                if cluster_metric > 0:
                    pool = mp.Pool(processes=cpu_nb_slaves)
                    for method in range(len(methods)):
                        distance_higher[:, :, method] += np.array(pool.starmap(parallel_dist_order, (
                            (graphs[k], method, [p], q, 1)
                            for p in clusters for q in clusters))).reshape((n, n))
                    for q in range(len(clusters)):
                        i_clust = find_clust(clusters[q], clusters_idx_k)
                        distance_higher[:, q, -1] += np.array([np.sum([
                            (graphs[k].paths[p][clusters[q]] + graphs[k].paths[p2][clusters[q]]) / graphs[k].paths[p][
                                p2]
                            for p2 in clusters_idx_k[i_clust] if p2 > p]) / len(
                            clusters_idx_k[i_clust]) if p in clusters_idx_k[i_clust] else 0 for p in clusters])
                    pool.close()
                if cluster_metric == -1 or cluster_metric == 2:
                    for method in range(len(methods)):
                        distance_cluster[:, :, method] += np.array(
                            [[parallel_dist_order(graphs[k], method, (i for i in clusters_idx_k[c]), q, 1)
                              for c in range(len(clusters_idx_k))] for q in clusters])
                    distance_cluster[:, :, -1] += np.array([[sum([graphs[k].paths[clusters[p]][q] + sum(
                        [graphs[k].paths[p2][q] for p2 in clusters_idx_k[c] if p2 > q]) for q in clusters_idx_k[c]
                                                                  ]) / np.sum([graphs[k].paths[p][p2]
                                                                               for q in clusters_idx_k[c] for p2 in
                                                                               clusters_idx_k[c] if p2 > q]) for c in
                                                             range(len(clusters_idx_k))] for p in range(len(clusters))])
                # point wise slaves optimization
                theta = [(u_k[q, q] + np.sum(old_w[-1][idx_update:] * sum(
                    [2 ** (current_order - 2) for current_order in range(2, order + 1)]) *
                                             distance_higher[q, q, :])) / (n + 1) for
                         q in range(len(clusters))]
                for p in range(len(clusters)):
                    # variable to store the higher order component
                    theta_k = [theta[q] + la_p_copy[k][clusters[p], clusters[q]] for q in range(len(clusters))]
                    theta_bar_k = [(u_k[p, q] + np.sum(old_w[-1][idx_update:] * sum(
                        [2 ** (current_order - 2) for current_order in range(2, order + 1)]) * distance_higher[p, q,
                                                                                               :]) + max(
                        theta_k[q], 0)) if q != p
                                   else theta_k[q] for q in
                                   range(len(clusters))]
                    mini = np.argmin(theta_bar_k)
                    # Corresponds to x^{k,p}_pq * f^k_pq term for p fixed and all q
                    aux_p = np.array([q == mini if p != q else 0 for q in range(len(clusters))])
                    w[:idx_update] += s_t * sum(
                        [aux_p[q] * features_list[k][clusters[p], clusters[q], :] for q in range(len(clusters))])
                    w[-1] += s_t * sum(
                        [aux_p[q] * graphs[k].paths[clusters[p]][clusters[q]] for q in range(len(clusters)) if p != q])

                    # Corresponds to x^p_qq term used to complete previous sum of x^{k,p}_pq*f^k_pq and sum over X^q_q*f^k_qq
                    for q in range(len(clusters)):
                        compo = (theta_k[q] < 0) if q != p else (q == mini)
                        aux = s_t * compo / (n + 1)
                        w[:idx_update] += aux * features_list[k][clusters[q], clusters[q], :]
                        # second term
                        w[idx_update:] += distance_higher[p, q, :] * (
                                s_t * compo * aux_p[q] * sum(
                            [2 ** (current_order - 2) - 1 for current_order in range(3, order + 1)]) + s_t * aux_p[
                                    q] * (order - 2))
                        # Higher-order term
                        w[idx_update:] += distance_higher[q, q, :] * s_t * compo * aux_p[q] * sum(
                            [2 ** (current_order - 2) for current_order in range(3, order + 1)])
                        w[-1] += aux * graphs[k].paths[clusters[q]][clusters[q]]
                        la_p[k][p, q] += compo
                        la_p[k][:, q] = np.around(la_p[k][:, q] - aux, decimals=12)
                        la_C[k][q] = np.around(la_C[k][q] - aux, decimals=12)
                print("End Ep", time.time() - begin_Ep)
                # Computation of the minimizers per cluster wise slaves
                count = 1
                print("Beginning EC")
                theta_k = [theta[q] + la_C_copy[k][clusters[q]] + np.sum(
                    old_w[-1][idx_update:] * distance_cluster[q, find_clust(clusters[q], clusters_idx_k), :]) for q in
                           range(len(clusters))]
                begin_EC = time.time()
                for i_clust in np.unique(clustering_list[k]):
                    cluster_i = [idx for idx in range(len(clusters)) if clusters[idx] in clusters_idx[k][i_clust]]
                    n_i = len(cluster_i)
                    for q in range(n_i):
                        compo = s_t * (theta_k[cluster_i[q]] < alpha if (2 * alpha + sum(
                            [min(theta_k[cluster_i[q_2]] - alpha, 0) for q_2 in range(n_i)])) < 0 else 0)
                        aux = np.float(compo / (n + 1))
                        la_C[k][clusters[cluster_i[q]]] = np.around(la_C[k][clusters[cluster_i[q]]] - (aux - compo))
                        la_p[k][:, clusters[cluster_i[q]]] = np.around(la_p[k][:, clusters[cluster_i[q]]] - aux)
                        w[:idx_update] += aux * features_list[k][clusters[cluster_i[q]], clusters[cluster_i[q]], :]
                        w[idx_update:] += aux * distance_higher[cluster_i[q], cluster_i[q], :] * (order - 2)
                        w[idx_update:] += aux * distance_cluster[cluster_i[q], i_clust, :]
                        w[-1] += aux * graphs[k].paths[clusters[cluster_i[q]]][clusters[cluster_i[q]]]
                    count += 1
                print("End EC", time.time() - begin_EC)
            print("w no max", w)
            # To have constrained values of w, update choices are defined at the beginning of the optimization process
            w = update(w)
            print("w", w)
            print("iteration " + str(counter) + " end")
            counter += 1
            old_w.append(np.copy(w))
        # Convergence check
        conv_count = convergence(old_w, conv_count, C)
        print("end of T iterations", time.time() - end_samples)
        sys.stdout.flush()
    # Compute the final cluster centers to be used for classification for instance
    centers_aux = [[q for q in range(len(clustering_list[k])) if sum(x[k][:, q]) > 0] for k in range(K)]
    print(centers_aux)
    centers = [[] for _ in range(K)]
    for k in range(K):
        for cluster_i in np.unique(clustering_list[k]):
            for q in centers_aux[k]:
                if clustering_list[k][q] == cluster_i:
                    centers[k].append(q)
    print(centers)
    print("End", time.time() - beginning)
    print("number of iterations:", counter)
    return w, la_p, la_C, counter, conv_count, centers


if __name__ == "__main__":
    # Generate Random training, validation and testing sets with:
    # d: dimension of the space
    # r: number of noisy dimensions
    # c:number of clusters
    # n: number of samples per cluster
    d = int(sys.argv[1])
    r = int(sys.argv[2])
    c = int(sys.argv[3])
    n = int(sys.argv[4])
    proba = float(sys.argv[5])
    cluster_metric = int(sys.argv[6])
    T = 2
    max_it = 50
    C = 0.1

    # Second-order metrics to consider: by default euclidean distance
    dists = [lambda x, y: (x - y) ** 2 for _ in range(d)]

    # Training set
    mean = np.array([0 for i in range(d)])
    cov = np.diag([10 if i < d - r else 1000 for i in range(d)])
    seed = 50
    np.random.seed(seed)

    data = np.concatenate(
        [np.random.multivariate_normal(i * np.array([np.random.randint(-50, 50) for j in range(d)]),
                                       cov + np.diag(
                                           [np.random.randint(0, 200) * i if j < d - r else np.random.randint(0, 200)
                                            for j in range(d)]),
                                       n + i) for i in range(c)])
    features_list = [np.array(data)]
    clustering_list = [
        [k for k in range(c) for _ in range(n + k)]]
    aux_mean = np.mean(np.concatenate(features_list, axis=0), axis=0)
    aux_std = np.std(np.concatenate(features_list, axis=0), axis=0)
    data = np.copy(features_list)
    features_list[0] = (features_list[0] - aux_mean) / aux_std
    features_list[0] = np.array(
        [[[dists[i](features_list[0][p, i], features_list[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list[0])[0])] for q in
         range(np.shape(features_list[0])[0])])

    alpha = 1.0
    beta = 1.0
    speed = 1
    tau = .5
    update_name = "pos"
    nn_numbers = 5
    order = 4
    cpu_nb_slaves = 6
    regu = "lasso"
    graphs = []
    aux_adj = [[] for i in range(c)]
    # Construct the adjacency matrix for the graph with noisy edges in proportion proba
    for i in range(c):
        for k in range(c):
            if i > k:
                aux_adj[i].append(aux_adj[k][i].T)
            else:
                adj = np.array(
                    np.array(np.random.rand(n + i, n + k) - proba >= 0, int) == int(i == k), int)
                if i == k:
                    aux_adj[i].append(np.tril(adj) + np.tril(adj).T - np.diag(adj))
                else:
                    aux_adj[i].append(adj)
    adjacency = np.block(aux_adj)
    graphs = [graph.Graph.from_adjacency(adjacency)]
    w, la_p, la_C, counter, conv_count, centers = learn(clustering_list, features_list, T, C, max_it,
                                                        alpha=alpha, beta=beta, speed=speed, tau=tau,
                                                        update_name=update_name, nn_numbers=nn_numbers, order=order,
                                                        cpu_nb_slaves=cpu_nb_slaves, regu=regu, graphs=graphs,
                                                        cluster_metric=cluster_metric)

    print("counter", counter, "conv count:", conv_count)
    print("Centers:", centers)
    print("Weights:", w)

    import classify

    # Validation Set
    mean_val = np.array([10 for i in range(d)])
    cov_val = np.diag([30 if i < d - r else 2000 for i in range(d)])

    data_val = np.concatenate(
        [np.random.multivariate_normal(i * np.array([np.random.randint(-30, 30) for j in range(d)]),
                                       cov_val + np.diag(
                                           [np.random.randint(0, 200) * i if j < d - r else np.random.randint(0, 200)
                                            for j in range(d)]),
                                       n + i) for i in range(c)])
    features_list_val = [np.array(data_val)]
    clustering_list_val = [
        [k for k in range(c) for _ in range(n + k)]]

    data_val = np.copy(features_list_val)
    features_list_val[0] = (features_list_val[0] - aux_mean) / aux_std
    features_list_val[0] = np.array(
        [[[dists[i](features_list_val[0][p, i], features_list_val[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list_val[0])[0])] for q in
         range(np.shape(features_list_val[0])[0])])

    # Testing Set
    mean_test = np.array([50 for i in range(d)])
    cov_test = np.diag([30 if i < d - r else 10000 for i in range(d)])

    data_test = np.concatenate(
        [np.random.multivariate_normal(0.5 * i * np.array([np.random.randint(-50, 50) for j in range(d)]),
                                       cov_test + np.diag(
                                           [np.random.randint(0, 200) * i if j < d - r else np.random.randint(0, 200)
                                            for j in range(d)]),
                                       n + i) for i in range(c)])
    features_list_test = [np.array(data_test)]
    clustering_list_test = [
        [k for k in range(c) for _ in range(n + k)]]

    data_test = np.copy(features_list_test)
    features_list_test[0] = (features_list_test[0] - aux_mean) / aux_std
    features_list_test[0] = np.array(
        [[[dists[i](features_list_test[0][p, i], features_list_test[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list_test[0])[0])] for q in
         range(np.shape(features_list_test[0])[0])])
    # In case one would add other training, validation, testing sets
    for set_i in range(len(clustering_list)):
        distances_list = classify.to_centers_distance_matrix(data[set_i], np.concatenate((data[set_i], data_val[set_i],
                                                                                          data_test[set_i]), axis=0), w,
                                                             nn_numbers, centers[set_i],
                                                             clustering_list[set_i], order, False, [],
                                                             cluster_metric=cluster_metric, graph_train=[adjacency,
                                                                                                         clustering_list[
                                                                                                             set_i] +
                                                                                                         clustering_list_val[
                                                                                                             set_i] +
                                                                                                         clustering_list_test[
                                                                                                             set_i],
                                                                                                         clustering_list[
                                                                                                             set_i],
                                                                                                         proba])
        results_train, results_val, conf = classify.classify(data[set_i], data_val[set_i], clustering_list[set_i],
                                                             clustering_list_val[set_i], w, nn_numbers,
                                                             centers[set_i], [0, len(clustering_list_val[set_i])],
                                                             order, distances_list=distances_list)

        _, results_test, conf_test = classify.classify(data[set_i], data_test[set_i],
                                                       clustering_list[set_i],
                                                       clustering_list_test[set_i], w,
                                                       k, nn_numbers, centers[set_i],
                                                       [1, len(clustering_list_test[set_i])],
                                                       order,
                                                       distances_list=distances_list)
        print(results_train, results_val, results_test)
