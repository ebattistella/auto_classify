# -*- coding: utf-8 -*-

### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Graph; Distance properties
### Description: <Define a graph class using Networkx library and implement the needed graph properties>
### Input: data, set of nodes
### Ouput: graph structure, graph properties
###################################

import numpy as np
from scipy.sparse import csr_matrix
from networkx.algorithms.shortest_paths.dense import floyd_warshall
from networkx.convert_matrix import from_scipy_sparse_matrix
from networkx.algorithms.clique import graph_clique_number
from networkx.algorithms.connectivity.connectivity import node_connectivity
import networkx.algorithms.distance_measures
import networkx.algorithms.centrality


class Graph:
    def __init__(self, matrix):
        sparse_matrix = csr_matrix(matrix)
        self.graph = from_scipy_sparse_matrix(sparse_matrix)
        self.max_degree = max([self.graph.degree[i] for i in list(self.graph)])
        self.paths = []

    #Define a graph from a matrix of distance using a K-nearest neighbors approach
    @classmethod
    def from_data(cls, distance, k):
        closest_neigbors = np.zeros((distance.shape[0], distance.shape[0]))
        for i in range(distance.shape[0]):
            idx = np.argsort(distance[i, :])[:k]
            closest_neigbors[i, idx] = closest_neigbors[idx, i] = distance[i, idx]
        sparse_matrix = csr_matrix(closest_neigbors)
        return cls(sparse_matrix)

    #Define a graph from an adjacency matrix
    @classmethod
    def from_adjacency(cls, adjacency):
        sparse_matrix = csr_matrix(adjacency)
        return cls(sparse_matrix)

    #Add a sample to an adjacency matrix depending on its ground truth with noise in proportion proba
    #It will be used for the construction of a graph for classification in the provided example
    @classmethod
    def add_node(cls, adjacency, ground, sizes, proba):
        edges = np.array([int((np.random.rand() - proba >= 0) == (ground == k)) for k in sizes])
        aux = np.concatenate((np.concatenate((adjacency, edges.reshape(1, -1)), axis=0),
                              np.array(list(edges) + [0]).reshape(-1, 1)), axis=1)
        sparse_matrix = csr_matrix(aux)
        return cls(sparse_matrix)

    #Compute the path length between any pair of nodes
    #In case of a disconnected graph set the infinite values to max_path (needed for the metrics computation)
    def path_finding(self, max_path):
        self.paths = floyd_warshall(self.graph)
        for i in list(self.paths):
            self.paths[i] = dict(self.paths[i])
            for j in list(self.paths[i]):
                self.paths[i][j] = max_path if self.paths[i][j] == np.inf else self.paths[i][j]

    #Compute the clique order metric defined as max_degree - clique_order
    #It is computed for the nodes in indexes + center
    def clique_order(self, indexes, center):
        indexes.append(center)
        aux = self.graph.subgraph(indexes)
        return self.max_degree - graph_clique_number(aux)

    #Compute the eccentricity with center center in the subgraph defined by the nodes in indexes + center
    def eccentricity(self, indexes, center):
        indexes.append(center)
        subgraph = self.graph.subgraph(indexes)
        subpaths = dict({i: dict() for i in indexes})
        for i in indexes:
            for j in indexes:
                subpaths[i][j] = self.paths[i][j]
        return networkx.algorithms.distance_measures.eccentricity(subgraph, v=center, sp=subpaths)

    #Compute the connectivity in the subgraph defined by the nodes in indexes + center
    #It is defined as the number of nodes - their connectivity
    def connectivity(self, indexes, center):
        indexes.append(center)
        subgraph = self.graph.subgraph(indexes)
        return len(indexes) - node_connectivity(subgraph)
