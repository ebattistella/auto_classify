# -*- coding: utf-8 -*-

### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Distance Learning; Second-Order;
### Description: <Learn a second-order distance through conditional random field energy minimization>
### Input: data, ground truth
### Ouput: learned distance
###################################

import os, sys
import numpy as np
import time
import graph_nn


#Check for convergence
#old_w stores the list of weights in a loop of T iterations
#conv_count contains the number of consecutive iterations for which this function considered convergence reached
#C is a coefficient weighting the convergence criterion, the lower the easier to satisfy the criterion
#There is convergence if for all the weights the variance is lower than C x the average + C^2 x average over all weights
def convergence(old_w, conv_count, criterion):
    l = [[w[i] for w in old_w] for i in range(len(old_w[0]))]
    std_l = [np.std(l[i]) for i in range(len(l))]
    crit_l = [criterion * abs(np.mean(l[i])) + criterion ** 2 * abs(np.mean([l[j] for j in range(len(l)) if j != i]))
              for i in range(len(l))]
    print("convergence criterion per dimension", [std_l[i] > crit_l[i] for i in range(len(l))],
          [(std_l[i], crit_l[i]) for i in range(len(l))])
    if sum([std_l[i] > crit_l[i] for i in range(len(l))]) < 1:
        conv_count += 1
    else:
        conv_count = 0
    return conv_count

#Implementation of the second-order distance learning algorithm
#clustering_list: ground-truth, list of labels for the samples for the K training sets
#feature_list: tensor containing for each pair of samples their distance according to each metric we are learning on
#T: number of projected subgradient rounds
#C: coefficient for the convergence criterion, cf convergence function
#max_it: maximum number of iterations in case convergence is not reached
#alpha, beta, tau: coefficients of the constraints and of the regularization, to be tuned
#speed: coefficient influencing the speed of convergence, weight the updates
#weighted: if we consider or not a balanced error loss
#update_name: name of the projection to be update at each iteration, in the higher-order case, the weights need to be positive
#make_graph: if path length has to be considered
#nn_numbers: size of the neighborhood to consider when builgind graphs based on the neighborhood
#regu: name of the regularizationmethod, implemented methods are "lasso", "ridge" and "elasticnet"
def learn(clustering_list, features_list, T, C, max_it, alpha=1, beta=1, speed=1, tau=0.5, weighted=False, update_name="id", make_graph = False, nn_numbers = 5, regu = "lasso"):
    # maximal distance to replace inf in paths length
    max_path = 10**2 * np.max(features_list[0], axis=None)
    #weights to apply to penalyze or not samples from smaller clusters
    classes_weights = [[1 for _ in np.unique(k)] for k in clustering_list]
    if weighted:
        classes_weights = [[len(k) / count for count in np.unique(k, return_counts=True)[1]] for k in
                           clustering_list]
        print("classes", classes_weights)
    #Projection function on a given set of constraints for w update
    update= lambda x: x
    if update_name == "pos":
        update = lambda x: np.array([max(x[i],0.0) for i in range(len(x))])
    elif update_name == "norm":
        update = lambda x: np.array([abs(x[i]) for i in range(len(x))])
    elif update_name == "thres1":
        update = lambda x: np.array([x[i] if np.abs(x[i]) >= 1 else 0 for i in range(len(x))])
    elif update_name == "min_max":
        update = lambda x: np.array([(x[i]-min(x))/(max(x)-min(x)) for i in range(len(x))])
    elif update_name == "abs_min_max":
        update = lambda x: np.array([(abs(x[i]) - min(np.abs(x))) / (max(np.abs(x)) - min(np.abs(x))) for i in range(len(x))])
    elif update_name == "abs_th_min_max":
        thresholding = lambda x: x if x > 0.001 else 0
        update = lambda x: np.array([thresholding((abs(x[i])-min(np.abs(x)))/(max(np.abs(x))-min(np.abs(x)))) for i in range(len(x))])
    K = len(clustering_list)
    # list of lambdas, array for the point wise slaves, lists for the clusters wise slaves
    la_p = [np.zeros((len(clustering_list[k]), len(clustering_list[k]))) for k in range(K)]
    la_C = [np.zeros(len(clustering_list[k])) for k in range(K)]
    # distance weights
    d = np.shape(features_list[0])[-1]
    if make_graph:
        w = np.array([1.0 for _ in range(d + 1)])
    else:
        w = np.array([1.0 for _ in range(d)])
    # ground truth with centers minimizing the energy
    x = [0 for _ in range(K)]
    # beta weight for the importance of having all the points of a ground truth cluster assigned to the same exemplar
    # alpha weight for the importance of having only one exemplar per ground truth cluster
    # tau weight of the regularization of w
    # subderivative of the considered regularization (here L1 norm)
    if regu == "lasso":
        delta_J = lambda x: np.array([tau if x_i > 0 else -tau for x_i in x])
    elif regu == "elasticnet":
        delta_J = lambda x: np.array([tau * (1 + 2 * x_i) if x_i > 0 else tau * (-1 + 2 * x_i) for x_i in x])
    elif regu == "ridge":
        delta_J = lambda x: np.array([2 * x_i for x_i in x])
    conv_count = 0
    counter = 1
    if make_graph:
        idx_update = -1
    else:
        idx_update = len(w)
    beginning = time.time()
    # until convergence
    while counter < max_it and conv_count < 3:
        old_w = [np.copy(w)]
        print("convergence: ", conv_count)
        begin_loop = time.time()
        for k in range(K):
            exemplars = []
            clusters = clustering_list[k]
            n = len(clusters)
            distance = np.sum(w[:idx_update] * features_list[k], axis=2)
            # compute the distance for the current sample, d is the vector of distances we use for each dimension of the features
            if make_graph:
                graph = graph_nn.graph_building(np.sum(features_list[k], axis=2)/d, nn_numbers)
                paths_dist = graph_nn.path_finding(graph)
                paths_dist[paths_dist == np.inf] = max_path
                distance += w[-1] * paths_dist
            # auxiliary variable for the ground truth
            x_k = np.zeros((n, n))
            for i in np.unique(clusters):
                # determine the exemplar of each cluster by considering the sum of distances minimizer
                cluster_i = [idx for idx in range(n) if clusters[idx] == i]
                distance_i = distance[cluster_i, :]
                distance_i = distance_i[:, cluster_i]
                q_i = np.argmin(np.sum(distance_i, axis=0))

                # assign each  exemplar to itself
                exemplars.append(cluster_i[q_i])
                print("exemplars", exemplars)
                for l in cluster_i:
                    x_k[l, cluster_i[q_i]] = 1
            x[k] = x_k

        end_samples = time.time()
        print("End samples", end_samples - begin_loop)

        for i in range(T):
            # s_t: decreasing towards 0 function for the updates
            s_t = 1 / (speed * counter)
            #update of w according to x_k except for the weight corresponding to the graph-based metric
            w[:idx_update] -= s_t * np.array([sum(sum(sum(x[k][p, q] * features_list[k][p, q, i] for p in range(len(clustering_list[k])))
                                   for q in range(len(clustering_list[k]))) for k in range(K)) for i in range(d)])
            la_p_copy = [np.copy(la_p[k]) for k in range(K)]
            la_C_copy = [np.copy(la_C[k]) for k in range(K)]
            w -= s_t * delta_J(old_w[-1])

            for k in range(K):
                clusters = clustering_list[k]
                n = len(clusters)
                distance = np.sum(old_w[-1][:idx_update] * features_list[k], axis=2)
                if make_graph:
                    #do not recompute the graph at each iteration if we have only one training cohort
                    if K > 1:
                        graph = graph_nn.graph_building(np.sum(features_list[k], axis=2)/d, nn_numbers)
                        paths_dist = graph_nn.path_finding(graph)
                        paths_dist[paths_dist == np.inf] = max_path
                    distance += old_w[-1][-1] * paths_dist
                    w[-1] -= s_t * sum(sum(x[k][p, q] * paths_dist[p,q] for p in range(len(clustering_list[k])))
                            for q in range(len(clustering_list[k])))
                #bars have been omitted in the notation here for conciseness'sake
                # u_k(1) in the article multiply by x_pq to have u_k(x_pq) =  distance + beta if p and q belongs to the same cluster else only distance
                #p and q belong to same cluster iff they are assigned to same center (which is the only l for which x_k(pl)>0 and so the argmax)
                #we can weight the penalty for points belonging to a same cluster in  ground truth but not in same predicted cluster or conversely according to their ground truth cluster size, to favor small classes
                u_k = distance + beta * np.array(
                    [[classes_weights[k][clusters[p]] * (clusters[p] == clusters[q]) +
                      classes_weights[k][clusters[p]] * classes_weights[k][clusters[q]] * ((np.argmax(x[k][p, :]) == np.argmax(x[k][q, :])) != (clusters[p] == clusters[q]))
                      for q in range(n)] for p in range(n)])
                # computation of the minimizers per slaves (point wise and cluster wise)
                print("Beginning Ep")
                begin_Ep = time.time()
                theta = [u_k[q, q] / (n + 1) for q in range(n)]
                for p in range(n):
                    theta_k = [theta[q] + la_p_copy[k][p, q] for q in range(n)]
                    theta_bar_k = [u_k[p, q] + max(theta_k[q], 0) if q != p else theta_k[q] for q in range(n)]
                    mini = np.argmin(theta_bar_k)
                    # corresponds to x^k,p_pq * f^k_pq term for p fixed and all q
                    aux_p = [(q == mini) * features_list[k][p, q, :] for q in range(n) if p != q]
                    w[:idx_update] += s_t * sum(aux_p)
                    if make_graph:
                        w[-1] += s_t * sum([(q == mini) * paths_dist[p, q] for q in range(n) if p != q])
                        # corresponds to x^p_qq term used to complete previous sum of x^k,p_pq*f^k_pq and sum over X^q_q*f^k_qq
                    for q in range(n):
                        compo = s_t * ((theta_k[q] < 0) if q != p else (q == np.argmin(theta_bar_k)))
                        aux = compo / (n + 1)
                        if (aux * features_list[k][q, q, :] < 0).any():
                            print(p, q, compo, features_list[k][q, q, :])
                        w[:idx_update] += aux * features_list[k][q, q, :]
                        if make_graph:
                            #paths dist normally = 0 for a node to itself but might enable more general definitions of paths
                            w[-1] += aux * paths_dist[q, q]
                        la_p[k][p, q] += compo
                        la_p[k][:, q] = np.around(la_p[k][:, q] - aux, decimals=12)
                        la_C[k][q] = np.around(la_C[k][q] - aux, decimals=12)
                print("end Ep", time.time() - begin_Ep)
                # computation of the minimizers per cluster wise slaves
                count = 1
                print("Beginning EC")
                theta_k = [theta[q] + la_C_copy[k][q] for q in range(n)]
                begin_EC = time.time()
                for i in np.unique(clusters):
                    cluster_i = [idx for idx in range(n) if clusters[idx] == i]
                    n_i = len(cluster_i)
                    for q in range(n_i):
                        compo = s_t * (theta_k[cluster_i[q]] < alpha if (2 * alpha + sum(
                            [min(theta_k[cluster_i[q_2]] - alpha, 0) for q_2 in range(n_i)])) < 0 else 0)
                        aux = np.float(compo / (n + 1))
                        la_C[k][cluster_i[q]] = np.around(la_C[k][cluster_i[q]] - (aux - compo), decimals=12)
                        la_p[k][:, cluster_i[q]] = np.around(la_p[k][:, cluster_i[q]] - aux, decimals=12)
                        w[:idx_update] += aux * features_list[k][cluster_i[q], cluster_i[q], :]
                        if make_graph:
                            w[-1] += aux * paths_dist[cluster_i[q], cluster_i[q]]
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
    centers_aux = [[q for q in range(len(clustering_list[k])) if x[k][q, q] > 0] for k in range(K)]
    print(centers_aux)
    print(clustering_list)
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
    T = 5
    max_it = 50
    C = 0.1
    weighted_list = ["unweighted", "weighted"]

    dists = [lambda x, y: (x - y) ** 2 for _ in range(d)]

    mean = np.array([0 for i in range(d)])
    cov = np.diag([10 if i < d - r else 1000 for i in range(d)])
    seed = 50
    np.random.seed(seed)

    data = np.concatenate(
        [np.random.multivariate_normal(i + np.array([np.random.randint(-50, 50) for j in range(d)]),
                                       cov + np.diag([np.random.randint(0, 100) * i if j < d - r else  np.random.randint(0, 200) for j in range(d)]),
                                       n +  i) for i in range(c)])
    features_list = [np.array(data)]
    clustering_list = [
        [k for k in range(c) for _ in range(n +  k)]]

    aux_mean = np.mean(np.concatenate(features_list, axis=0), axis=0)
    aux_std = np.std(np.concatenate(features_list, axis=0), axis=0)
    data = np.copy(features_list)
    features_list[0] = (features_list[0] - aux_mean) / aux_std
    features_list[0] = np.array(
        [[[dists[i](features_list[0][p, i], features_list[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list[0])[0])] for q in
         range(np.shape(features_list[0])[0])])

    alpha = 1
    beta = 40
    speed = 1
    tau = 0.5
    update_name = "pos"
    nn_numbers = 10
    cpu_nb_slaves = 10
    regu = "lasso"
    make_graph = False
    weighted = True

    w, la_p, la_C, counter, conv_count, centers = learn(clustering_list, features_list, T, C, max_it,
                                                             alpha=alpha, beta=beta, speed=speed, tau=tau, weighted=weighted,
                                                            update_name=update_name, make_graph=make_graph, nn_numbers=nn_numbers,
                                                            regu=regu)
    print("counter", counter, "conv count:", conv_count)
    print(w, centers)

    import classify

    mean_val = np.array([10 for i in range(d)])
    cov_val = np.diag([30 if i < d - r else 2000 for i in range(d)])

    data_val = np.concatenate(
        [np.random.multivariate_normal(i + np.array([np.random.randint(-30, 30) for j in range(d)]),
                                       cov_val + np.diag([np.random.randint(0, 200) * i if j < d - r else  np.random.randint(0, 200) for j in range(d)]) ,
                                       n +  i) for i in range(c)])
    features_list_val = [np.array(data_val)]
    clustering_list_val = [
        [k for k in range(c) for _ in range(n +  k)]]

    data_val = np.copy(features_list_val)
    features_list_val[0] = (features_list_val[0] - aux_mean) / aux_std
    features_list_val[0] = np.array(
        [[[dists[i](features_list_val[0][p, i], features_list_val[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list_val[0])[0])] for q in
         range(np.shape(features_list_val[0])[0])])

    mean_test = np.array([50 for i in range(d)])
    cov_test = np.diag([30 if i < d - r else 10000 for i in range(d)])

    data_test = np.concatenate(
        [np.random.multivariate_normal(0.5*i + np.array([np.random.randint(-50, 50) for j in range(d)]),
                                       cov_test + np.diag([np.random.randint(0, 200) * i if j < d - r else np.random.randint(0, 200) for j in range(d)]),
                                       n +  i) for i in range(c)])
    features_list_test = [np.array(data_test)]
    clustering_list_test = [
        [k for k in range(c) for _ in range(n +  k)]]

    data_test = np.copy(features_list_test)
    features_list_test[0] = (features_list_test[0] - aux_mean) / aux_std
    features_list_test[0] = np.array(
        [[[dists[i](features_list_test[0][p, i], features_list_test[0][q, i]) for i in range(d)] for p in
          range(np.shape(features_list_test[0])[0])] for q in
         range(np.shape(features_list_test[0])[0])])

    algo = -1
    dist = -1
    weight_knn = 0
    k = 5

    for set_i in range(len(clustering_list)):
        distances_list = []
        distances_list.append(classify.to_distance_matrix(dist, data[set_i],
                                                                        np.concatenate(
                                                                            (data[
                                                                                 set_i],
                                                                             data_val[
                                                                                 set_i],
                                                                             data_test[
                                                                                 set_i]), axis=0),
                                                                        w, make_graph, nn_numbers,
                                                                        clustering_list[set_i],
                                                                        False,False))
        distances_list.append(classify.to_mean_distance_matrix(dist, data[set_i],
                                                                        np.concatenate(
                                                                            (data[
                                                                                 set_i],
                                                                             data_val[
                                                                                 set_i],
                                                                             data_test[
                                                                                 set_i]), axis=0),
                                                                        clustering_list[set_i],w, make_graph, nn_numbers,
                                                                        False,False))
        distances_list.append(classify.to_centers_distance_matrix(dist, features_list[set_i],
                                                                        np.concatenate(
                                                                            (data[
                                                                                 set_i],
                                                                             data_val[
                                                                                 set_i],
                                                                             data_test[
                                                                                 set_i]), axis=0),
                                                                        w, make_graph, nn_numbers, clustering_list[set_i],
                                                                        distances_list[0],
                                                                        False))
        distances = ["pearson", "spearman", "kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs",
                     "kendall_abs"]
        distances += ["learned_" + i for i in distances] + ["learned"]
        results_train, results_val, conf = classify.classify(
            data[set_i], data_val[set_i],
            clustering_list[set_i], clustering_list_val[set_i], algo, dist, w, k, weight_knn,
            make_graph, nn_numbers,
            centers[set_i], distances[dist],
            indexes_set =[0, len(clustering_list_val[set_i])], distances_list=distances_list)

        _, results_test, conf_test = classify.classify(features_list[set_i],
                                                                                  data_test[set_i],
                                                                                  clustering_list[set_i],
                                                                                  clustering_list_test[set_i], algo,
                                                                                  dist, w, k,
                                                                                  weight_knn, make_graph, nn_numbers,
                                                                                  centers[set_i], distances[dist],
                                                                                  indexes_set =[1, len(clustering_list_test[set_i])],
                                                                                  distances_list=distances_list)

        print(results_train,results_val, results_test)
