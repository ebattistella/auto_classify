### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: Feature Selection; Ensemble;
### Description: <Implement the ensemble feature selection techniques relying on prevalence and co-selection graph>
### Input: diag: features prevalence, cooc: co-selection matrix, max_it: maximal prevalence, k_S: decomposition level,
###        k_feat: number of features, th: selection threshold, n_jobs: number of cpus to use
### Ouput: list of selected features
###################################

from . import density_decomposition_prepro
import uuid, os
from itertools import chain
from . import hks_interface
import numpy as np
from .densest import densest_subgraph

# Select the features with prevalence 100%
def consensus_selector(diag, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    return (list((diag == max_it).nonzero()[0]))

# Select the features with prevalence above th% of the maximal prevalence
def majority_selector(diag, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    return (list((diag >= th * max_it).nonzero()[0]))


# Select the k_feat features with highest prevalence
def threshold_selector(diag, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    idx = np.argsort(diag)[::-1]
    return [idx[i] for i in range(min(k_feat,len(idx)))]

# Rely on the density friendly algorithm
def cooc_selector(cooc, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    filename = path + str(uuid.uuid4())
    density_decomposition_prepro.write_file(cooc, filename)
    density_decomposition_prepro.launch_decompo(filename, iter=100000, ncpu=n_jobs)
    S = density_decomposition_prepro.read_decompo(filename)
    if S == []:
        return []
    files = [".txt", "rates.txt", "pavafit.txt", "cuts.txt", "exact.txt"]
    for file in files:
        os.remove(filename + file)
    return chain(*S[:k_S])

# Rely on the heaviest k-subgraph algorithm
def k_density_selector(cooc, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    # Define a unique filename fof the heaviest k-density output, files will be removed after use
    filename = path + str(uuid.uuid4())
    hks_interface.write_file(cooc, filename)
    hks_interface.launch_hks(filename, k_feat, filename + '_out')
    density, size, nodes = hks_interface.read_result(filename + '_out')
    if size == 0:
        return []
    # Remove the generated files
    files = [".txt", "_out.txt"]
    for file in files:
        os.remove(filename + file)
    print(density, size, nodes)
    return nodes

# Rely on the density friendly algorithm, returns the density to score the selected features and compare different selections
def k_density_selector_repeat(cooc, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    # Define a unique filename fof the heaviest k-density output, files will be removed after use
    filename = path + str(uuid.uuid4())
    hks_interface.write_file(cooc, filename)
    hks_interface.launch_hks(filename, k_feat, filename + '_out')
    density, size, nodes = hks_interface.read_result(filename + '_out')
    # Remove the generated files
    files = [".txt", "_out.txt"]
    for file in files:
        os.remove(filename + file)
    return nodes, density

# Versions using the densest subgraph, less efficient and performs worse than the density decomposition
def densest_selector(cooc, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    S = densest_subgraph(cooc)
    return S

def densest_selector_robust(cooc, max_it, k_S=1, k_feat=5, th=0.25, n_jobs=1, path="./"):
    features = []
    for i in cooc:
        features.append(densest.densest_subgraph(i)[0])
    return list(set().union(*features))
