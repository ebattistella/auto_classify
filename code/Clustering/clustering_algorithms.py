import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, \
    OPTICS, Birch
from sklearn.pipeline import Pipeline
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import pickle
from lpstability import Lpstability
from dunn import dunn_index
import itertools
from sklearn.metrics import pairwise_distances

class AffinityPropagation_aux(AffinityPropagation):
    def __init__(self, dict_params):
        affinity =dict_params['affinity']
        damping = dict_params['damping']
        max_iter = dict_params['max_iter']
        convergence_iter = dict_params['convergence_iter']
        super().__init__(affinity=affinity, damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)

class KMeans_aux(KMeans):
    def __init__(self, dict_params):
        n_clusters =dict_params['n_clusters']
        init = dict_params['init']
        max_iter = dict_params['max_iter']
        n_init = dict_params['n_init']
        super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init)

class MeanShift_aux(MeanShift):
    def __init__(self, dict_params):
        n_jobs = dict_params['n_jobs']
        super().__init__(n_jobs=n_jobs)

class SpectralClustering_aux(SpectralClustering):
    def __init__(self, dict_params):
        n_clusters =dict_params['n_clusters']
        random_state = dict_params['random_state']
        n_init = dict_params['n_init']
        gamma = dict_params['gamma']
        n_jobs = dict_params['n_jobs']
        super().__init__(n_clusters=n_clusters, random_state=random_state, n_init=n_init, gamma=gamma,n_jobs=n_jobs)

class AgglomerativeClustering_aux(AgglomerativeClustering):
    def __init__(self, dict_params):
        n_clusters =dict_params['n_clusters']
        linkage = dict_params['linkage']
        super().__init__(n_clusters=n_clusters, linkage=linkage)

class DBSCAN_aux(DBSCAN):
    def __init__(self, dict_params):
        eps =dict_params['eps']
        min_samples = dict_params['min_samples']
        n_jobs = dict_params['n_jobs']
        super().__init__(eps=eps, min_samples=min_samples, n_jobs=n_jobs)

class OPTICS_aux(OPTICS):
    def __init__(self, dict_params):
        min_samples = dict_params['min_samples']
        n_jobs = dict_params['n_jobs']
        super().__init__(min_samples=min_samples, n_jobs=n_jobs)

class Birch_aux(Birch):
    def __init__(self, dict_params):
        n_clusters =dict_params['n_clusters']
        threshold = dict_params['threshold']
        branching_factor = dict_params['branching_factor']
        super().__init__(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)

def davies_bouldin(X, labels, metric='euclidean'):
    return davies_bouldin_score(X, labels)


def calinski_harabasz(X, labels, metric='euclidean'):
    return calinski_harabasz_score(X, labels)


def tuning(x_train, n_clusters_max=20, metric_idx=0, name="test"):
    print("Tuning")
    n_clusters_max = min(np.shape(x_train)[0], n_clusters_max)
    # Assessment metrics
    score_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Dunn Index']
    scores = [silhouette_score, calinski_harabasz, davies_bouldin, dunn_index]
    # Distance metrics
    possible_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'correlation',
                          'chebyshev']
    constrained_metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    possible_affinities = ['nearest_neighbors', 'rbf']
    # Define the list of the different possible metrics and affinities
    metrics = [[('metric', 'euclidean')], [("metric", metric) for metric in possible_metrics],
               [("affinity", 'euclidean')], [('metric', 'euclidean')],
               [("affinity", metric) for metric in possible_affinities],
               [("affinity", metric) for metric in constrained_metrics],
               [("metric", metric) for metric in possible_metrics],
               [("metric", metric) for metric in possible_metrics],
               [('metric', 'euclidean')]
               ]
    # Names of the clustering algorithms and their parameters
    names = ['K-Means', 'LP-Stability', 'Affinity Propagation', 'Mean Shift', 'Spectral Clustering',
             'Agglomerative Clustering', 'DBSCAN', 'OPTICS', 'Birch']
    clfs = [KMeans_aux, Lpstability, AffinityPropagation_aux, MeanShift_aux, SpectralClustering_aux, AgglomerativeClustering_aux,
            DBSCAN_aux, OPTICS_aux, Birch_aux]
    params = [
        [
            list(range(2, n_clusters_max)),
            ['k-means++'], [300], [10]
        ],
        [
            list(np.arange(0.1, 0.2, 0.1)), [name]
        ],
        [
            [0.5], [200], [15]
        ],
        [
            [10]
        ],
        [
            list(range(2, n_clusters_max)),
            [82], [10], [1.0], [10]
        ],
        [
            list(range(2, n_clusters_max)),
            ['complete', 'average', 'single']
        ],
        [
            list(np.arange(0.1, 1, 0.1)),
            list(range(2, 10)), [10]
        ],
        [
            list(range(2, 10)), [10]
        ],
        [
            list(np.arange(0.1, 1, 0.1)),
            list(range(30, 100, 10)),
            list(range(2, n_clusters_max))
        ]
    ]
    param_names = [
        ['n_clusters', 'init', 'max_iter', 'n_init'],
        ["penalty", "name"],
        ["damping", "max_iter", "convergence_iter"],
        ["n_jobs"],
        ['n_clusters', 'random_state', 'n_init', 'gamma', "n_jobs"],
        ['n_clusters', 'linkage'],
        ['eps', 'min_samples', 'n_jobs'],
        ['min_samples', 'n_jobs'],
        ['threshold', 'branching_factor', 'n_clusters']
    ]
    results = []
    best_params = []
    clusters = []
    if isinstance(x_train, pd.Series):
        x_train = x_train.values.reshape((-1, 1))
    for i in range(len(names)):
        print(i, names[i])
        if not os.path.exists(name + "_" + names[i] + ".sav"):
            aux_results = []
            aux_labels = []
            for metric_name, metric in metrics[i]:
                for param in list(itertools.product(*params[i])):
                    pip = Pipeline([('model', clfs[i](dict([(metric_name, metric)] +
                                                       [(param_names[i][k], param[k]) for k in
                                                        range(len(param))])))])
                    try:
                        pip.fit(x_train)
                    except Exception as e:
                        print(e)
                        continue
                    labels = pip['model'].labels_
                    if (np.unique(labels)<0).any():
                        labels = [list(np.unique(labels)).index(label_i) for label_i in labels]
                    if len(np.unique(labels)) == 1:
                        labels[0] = labels[0] + 1
                    metric_aux = metric
                    if metric_aux not in possible_metrics:
                        metric_aux = 'euclidean'
                    # Compute the performance
                    aux_results.append([names[i]] + [
                        [(metric_name, metric)] + [(param_names[i][param_i], param[param_i]) for param_i in range(len(param))]] + [
                                           (score_names[score_idx], scores[score_idx](x_train, labels, metric=metric_aux)) for
                                           score_idx in range(len(scores))])
                    aux_labels.append(labels)
            # Save the clusters
            pickle.dump(aux_results, open(name + "_" + names[i] + ".sav", "wb"))
            np.save(name + "_labels_" + names[i], aux_labels)
        else:
            aux_results = pickle.load(open(name + "_" + names[i] + ".sav", 'rb'))
            aux_labels = np.load(name + "_labels_" + names[i] + ".npy", allow_pickle=True)
        if aux_results:
            clusters.append(aux_labels)
            results.append(aux_results)
            best_params.append([aux_results[0][0], aux_results[np.argmax([res[metric_idx + 2][1] for res in aux_results])][1]])
    return results, best_params, clusters

# Test function to compute the clusters on a test dataset, or can be used on a training dataset if one want to manually
# select the algorithms parameters
def test(x_test, params, name):
    print("Testing")
    possible_metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'correlation',
                        'chebyshev']
    names = ['K-Means', 'LP-Stability', 'Affinity Propagation', 'Mean Shift', 'Spectral Clustering',
             'Agglomerative Clustering', 'DBSCAN', 'OPTICS', 'Birch']
    clfs = [KMeans_aux, Lpstability, AffinityPropagation_aux, MeanShift_aux, SpectralClustering_aux, AgglomerativeClustering_aux,
            DBSCAN_aux, OPTICS_aux, Birch_aux]
    score_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Dunn Index']
    scores = [silhouette_score, calinski_harabasz, davies_bouldin, dunn_index]
    if isinstance(x_test, pd.Series):
        x_test = x_test.values.reshape((-1, 1))
    clusters = []
    results = []
    counter = 0
    for i in range(len(params)):
        if names[i+counter] != params[i][0]:
            counter += 1
        print(i, names[i+counter])
        if not os.path.exists(name + "_" + names[i+counter] + "_test.sav"):
            # Define the piepline with the clustering algorithms and their parameters
            pip = Pipeline([('model', clfs[i+counter](dict([(params[i][1][k][0], params[i][1][k][1]) for k in
                                                        range(len(params[i][1]))])))])
            try:
                pip.fit(x_test)
            except Exception as e:
                print(e)
                continue
            labels = pip['model'].labels_
            if names[i+counter] == 'LP-Stability':
                exemplars = pip['model'].exemplars_
                np.save(name + "_exemplars_test.npy", exemplars)
            aux_results = params[i]
            if (np.unique(labels) < 0).any():
                labels = [list(np.unique(labels)).index(label_i) for label_i in labels]
            if len(np.unique(labels)) == 1:
                labels[0] = labels[0] + 1
            metric_aux = params[i][1][0][1]
            if metric_aux not in possible_metrics:
                metric_aux = 'euclidean'
            aux_results += [
                (score_names[score_idx], scores[score_idx](x_test, labels, metric=metric_aux)) for score_idx
                in range(len(scores))]
            pickle.dump(aux_results, open(name + "_" + names[i+counter] + "_test.sav", "wb"))
            np.save(name + "_" + names[i+counter] + "_clusters_test.npy", labels)
        else:
            aux_results = pickle.load(open(name + "_" + names[i+counter] + "_test.sav", 'rb'))
            labels = np.load(name + "_" + names[i+counter] + "_clusters_test.npy", allow_pickle=True)
        results.append(aux_results)
        clusters.append(labels)
    return results, clusters

# Compute the medoids of the clusters, the data point the closest to the center (average of coordinates)
def medoid(cluster, X, metric):
    labels = list(np.unique(cluster))
    cluster_i = [[i for i in range(len(cluster)) if cluster[i]==label] for label in labels]
    if metric == "nearest_neighbors":
        metric = "euclidean"
    distances = [pairwise_distances(X[cluster_i[i], :], metric=metric) for i in range(len(labels))]
    return [cluster_i[i][np.argmin(distances[i].sum(axis=0))] for i in range(len(labels))]
