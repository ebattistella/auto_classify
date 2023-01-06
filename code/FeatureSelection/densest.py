### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Densest Subgraph;
### Description: <Uses a flow algorithm to find the densest subgraph of a weighted graph>
### Input: adjacency matrix of a weighted graph
### Ouput: densest subgraph, maximal density
###################################

from random import seed
import numpy as np
import networkx as nx
from networkx.algorithms.flow import minimum_cut

def define_graph(M, c):
    print("graph definition")
    G = nx.Graph(incoming_graph_data=M)
    for i in G.nodes:
        for j in G.nodes:
            if G.has_edge(i, j):
                G[i][j]['capacity'] = M[i][j]
    G.add_node(-1)
    G.add_node(-2)
    G.add_edges_from([(-1, i, {"capacity": G.degree(i, weight="capacity")})
                      for i in G.nodes if i not in [-1, -2]])
    G.add_edges_from([(i, -2, {"capacity": c}) for i in G.nodes if i not in [-1, -2]])
    print("graph defined")
    return G


def update_graph(G, c):
    for i in G.nodes:
        if i not in [-1, -2]:
            G[i][-2]['capacity'] = c
    print("graph updated")
    return G


def Fractionnal_programming(M):
    n = len(M)
    S = []
    c = n / 2
    G = define_graph(M, c)

    stop_criterion = 1.0/(n*(n-1))
    print(stop_criterion)
    lower_bound = 0
    higher_bound = n
    while higher_bound - lower_bound >= stop_criterion:
        aux_dens, aux_S = minimum_cut(flowG=G, _s=-1, _t=-2, capacity='capacity')
        print("minimum_cut", aux_dens)
        if len(aux_S[0]) > 1:
            lower_bound = c
            S = list(aux_S[0])
        else:
            higher_bound = c
        c = (higher_bound + lower_bound) / 2
        G = update_graph(G, c)
    print(S)
    if -1 in S:
        S.remove(-1)
    dens = 0
    if S:
        dens = sum([G[u][v]['weight'] for u in S for v in S if G.has_edge(u, v)])/len(S)
    return S, dens


def compute_lam(M, x):
    return x.dot(M.dot(x)) / sum(x)


def densest_subgraph(cooc):
    seed(82)
    incidence = cooc - np.diag(np.diag(cooc))
    print("incidence")
    S, dens = Fractionnal_programming(incidence)
    return S, dens
