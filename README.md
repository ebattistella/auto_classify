# Machine Learning Tools

## Python, C++ and Bash code

- **Version**: V1
- **Creator**: Enzo Battistella [en.battistella\@gmail.com](mailto:en.battistella@gmail.com) 
- **Last code update**: 3/15/2022 
- **Last data update**: 3/15/2022
- **Keywords**: [Machine Learning; Classification; Feature Selection; Assessment; AutoML]
- **Rights Statement**: Creative Common's Attribution-NonCommercial 4.0 International License.

### Description
This code provides different functions essential when performing feature selection and classification.
In addition, it provides snippets to use the autoML technique TPOT and the distance learning method GHOST, 
compute and plot shap values and perform ensemble feature selection.

Bash code to run the code with a slurm scheduler is provided.

### Methodology
This pipeline leverages the notions of:
- Prevalence highlighted in: [Chassagnon 2021](https://pubmed.ncbi.nlm.nih.gov/33171345/)
- Co-selection graph described in: Battistella (to be published)
- Higher-order metric learning for classification from: [Battistella 2022](https://hal.archives-ouvertes.fr/hal-03563705/)

### Summary
Here an index of the different section presented in this document:
- [Feature Selection](###feature selection)
- [Classification](###classification)
- [Full classification pipeline](###full classification pipeline)
- [Shap Values](###shap values)
- [AutoML](###tpot)
- [Higher-order metric learning](###ghost)
- [Clustering](###clustering)
- [LP-Stability](###lp-stability)

### Feature Selection
#### Description
This part implements diverse feature selection techniques.
It has been designed to leverage a cross-validation framework, several feature selectors and ensemble feature selection techniques.
An automatic tuning of the classifiers used for feature selection is also implemented.
Though, it is modular and adaptable, the different adaptations will be describe later on.

#### Methodology
This code proposes different ensemble selection approaches:
- Prevalence-based methods: in a selection framework including a cross-validation and/or several feature selection techniques,
the prevalence of a feature is defined as the number of selections of a feature by every selection technique over
the different cross-validation folds. Then, a threshold over the number of features to select or the minimal prevalence is
used to determine the final selection. This method has been defined in [Chassagnon 2021](https://dx.doi.org/10.1016%2Fj.media.2020.101860). 
- Co-selection-based techniques: in a selection framework including a cross-validation and/or several feature selection techniques,
the co-selection graph's nodes are the selected features and the edges between two node are weighted by 
the number of co-selections of these features on a same cross-validation fold by a same feature selector.
Then, the final selection is determined through the heaviest k-subgraph algorithm ([Letsios 2016](https://doi.org/10.1109/ICDMW.2016.0024))
This method aims at better accounting for the complementarity of the features selected by a same feature selector call.
This method is defined in Battistella (to be published)
- Co-importance-based techniques: this method is a variant of the previous one. Instead of weighting the edges between features
i and j according to the number of co-selections, we rely on the total co-importance which is defined as the sum of the minimal importance weight
a feature selector grant to i or j over a same cross-validation fold. This method is defined in Battistella (to be published)

#### Use
- 'main_selection.py': Provides a typical framework leveraging all the different functions implemented in this folder.
Given some data with ground truth, this script first performs a splitting of the data into training and test.
Then, it uses a cross-validation to tune the parameters of diverse classifiers-based feature selectors.
The tuned selectors are then used to select features on every training set of the cross-validation.
Finally, the ensemble techniques are used to identify a final signature.
Variables:
  - data: data considered, pandas DataFrame
  - target: labels corresponding to data, pandas Series
  - n_iter_selec: number of iterations for the tuning of selectors
  - k_feat: number of feature to select
  - path: where to save the intermediary and result files
  - pre_filter: if True, keep only the most correlated features by an anova test befor feature selection
  - seed: random seed
  - clf_only: if True, do not use feature selectors based on statistical metrics
  - splitted: if True, whole data and target sets are to be used for training
In addition, several variables defined at the beginning of the script can be modified to adapt the code.
The lists ensemble and ensemble_names include the different ensemble feature selection techniques, methods
can be removed or added if required. 
cv_num is the number of fold of the different cross-validations. It can be increased for a finer selection or
set to 1 if one need to use directly the whole training set for the feature selection and, for instance,
take the feature selected by a single selectors. 
If one wants to use the methods based on prevalence, co-selection or the features selected by a selector directly,
the variables threshold_select_classif and threshold_stats which are thresholds for the selectors respectively 
based on classifiers and on statistical tests.
- 'feature_selection.py': This script contains all the functions to performs the feature selection with the 
different selectors, a pre_filtering of the data based on an anova test and a merge function to combine the
co-selection matrices of the different feature selectors. The most important function of this script is
'feature_selector'. This function performs the tuning of the selectors and the selection of the features.
It can be adapted by changing the classifiers used for the selection variables clfs and names.
Also, if one want to manually tune the parameters of the different classifiers or change the ranges of values 
used in the random search, this can be done by modifying the params variable with the required values.
The variant of this method 'feature_selection_bootstrap' leverages a boostrap framework, i.e. the selection
is performed on features subsets sampled uniformly with replacement.
- 'ensemble_feature_selection.py': is an interface for the available ensemble feature selection approaches.
In particular:
  - consensus_selector: implements consensus voting: select the features with prevalence 100%
  - majority_selector: implements majority voting: select the features with prevalence above th%
  - threshold_selector: select the k_feat features with highest prevalence
  - cooc_selector: select the features in the k_S subset of the density friendly decomposition in the co-selection graph
  - k_density_selector: select features in the heaviest k_feat-subgraph of the input graph, can be used with
either co-selection or co-importance graphs
  - k_density_selector_repeat: is a variant of the previous method which return the density of the heaviest k_feat-subgraph
in order to perform an assessment / a comparison of the selected features
  - densest_selector: select the feature from the densest subgraph algorithm, less efficient than cooc_selector
  - densest_selector_robust: select the features from the densest graph of the co-selection graph of each selector, can 
be used to perform another ensemble approach or to compare the different selectors
- 'hks_interface.py': is an interface for the heaviest k-subgraph algorithm leveraging the c++ code from the folder hks
- 'density_decomposition_prepro.py': is an interface for the density friendly decomposition code contained in
density_decomposition.cpp and using the boost library from the folder boost_1_63_0
- 'densest.py': is an implementation of the densest subgraph retrieval algorithm relying on the flow algorithm

### Classification

#### Description
This  part implements all the different functions usefull for a complete classification task 
assuming that the features to consider are already known or have been selected with the
previous [feature selection code](###feature selection).

Specifically, it enables to perform a model tuning, define ensemble models from the train models,
assess the results on cross-validation using divers assessment metrics, select the best model
and, finally, test the models and save their performance in a .csv file.

#### Methodology
The classification pipeline implemented here is a generalization of the one proposed
in [Chassagnon 2021](https://dx.doi.org/10.1016%2Fj.media.2020.101860). 
It consists in leveraging diverse models through an ensemble framework leveraging the best performing ones.
The  good performance is assessed on the validation results for a given metric, in the article the balanced accuracy.
But, to avoid selecting method overfitting the training set, the discrepancy of the results
between training and testing is limited to a chosen threshold.

The different ensemble strategy considered are:
- Majority Voting: The final prediction is the class which was the most predicted by the different estimators.
- Weighted Majority Voting: Follows the same principle as majority voting except that the votes are weighted by the validation
performance of the estimator.
- Stacking Model: A simple consensus estimator is used to make the final prediction from the predictions of the different estimators.
Here, the default consensus estimator considered is a decision tree classifier.

#### Use
- 'main_classification.py': Provides a typical framework leveraging all the different functions implemented in this folder.
Given some data with ground truth, this script first performs a splitting of the data into training and test.
Then, it uses a cross-validation to tune the parameters of diverse classifiers.
The tuned classifiers' performance is assessed on cross-validation.
The best performing ones are selected to be used for building ensemble models.
Finally, the ensemble techniques are used to identify a final signature.
Variables:
  - data: data considered, pandas DataFrame
  - target: labels corresponding to data, pandas Series
  - features: the set of features to use for the classification
  - feature_names: the name used to identify the signature (e.g. the output name of Feature_Selection\main_selection.py) 
  - n_iter_classif: number of iterations for the tuning of selectors
  - k_feat: number of feature to select
  - criterion: metric to be used  for the models tuning, it has to be an authorized string from sklearn metrics
  - seed: random seed
  - clf_only: if True, do not use feature selectors based on statistical metrics
  - splitted: if True, whole data and target sets are to be used for training
In addition, several variables defined at the beginning of the script can be modified to adapt the code.
The select_th is the maximal allowed discrepancy between training and testing.
select_k is the number of classifiers to select for the ensemble methods.
cv_num is the number of fold of the different cross-validations. It can be increased for a finer selection or
set to 1 if one need to use directly the whole training set for the feature selection and, for instance,
take the feature selected by a single selectors. 
- 'ensemble.py': defines the different ensemble methods and the model rankin strategy define in methodology
- 'classification': implements the different classificaiton functions, in particular:
  - models_tuning: relies on a randomized search framework to define the best parameters for the different classification models
It returns the best models for each classifier and the corresponding average cross-validation results and their standard deviation.
  - cross_val: Assess the performance on cross-validation of the models provided in input, in particular, in the main function
it is used for the ensemble models. It returns the average results and their standard deviation on the cross-validation.
  - test_results: Perform the prediction on the testing set and return the results and confusion matrices
  - assessment: allows to compute the performance and confusion matrix of a given prediction (not used in the main file)
  - performance_saving: saves the results provided in input in the input .csv file.
  - confusion_matrix_saving: saves the list of confusion matrices provided in input in the input .csv file.

### Full Classification Pipeline
#### Use
- 'main.py': Combine the feature selection and the classification pipelines for an end-to-end classification.
- 'external_prediction.py': load the model and the features saved by the main file to perform a prediction on an external dataset.
- 'main.sh': example of a bash script to launch the main function on a SLURM scheduler, this job can be submitted using
sbatch --export=n_iter_selec=10,k_feat=5,pre_filter=0,seed=0,clf_only=0,n_iter_classif=10, main.sh.
- 'launcher_main_seed.sh': example of bash code to submit several jobs of main.sh with different parameters. 
### Shap Values

#### Description
Shap values are a convenient method to assess the importance of each feature in a model predictions.
This code leverages the [library shap](https://shap.readthedocs.io/en/latest/index.html).

#### Use
- 'shap_values.py': compute the shap values for a given model on the provided dataset and plot the 
importance of each feature using three different representations. The figures are saved at the provided path.

### TPOT

#### Description
TPOT is an automl algorithm aiming at finding the best regression or classification pipeline for a given task.
An example on a classification settings is provided in this repository.
TPOT is defined in [Le 2020](https://doi.org/10.1093/bioinformatics/btz470).
It is based on a genetic algorithm.

#### Use
- 'tpot_train.py': allows to train the autoML pipeline on a given dataset, it outputs a python file implementing the 
pipeline and returns the trained model. It can be then used in combination with the functions provided in the 
[assessment folder](###assessment) to obtain the performance of the model.

### GHOST

#### Description
GHOST is a distance learning technique defined in [Battistella 2022](https://hal.archives-ouvertes.fr/hal-03563705/).
It relies on conditional random fields and graph properties to estimate the best distance metric for a given classification task.
This distance can then be leveraged through an adapted K-Nearest Neighbors (K-NN) approach to predict a
sample class. This algorithm also allows to determine the features importance and can be used for 
higher-order feature selection.

The basic second-order implementation provides a usual distance defined from the provided features.
This method has been published in [Komodakis 2011](https://doi.org/10.1109/ICCV.2011.6126227).

The higher-order implementation provides an extended notion of distance as defined in the article.
It offers a usual distance defined from the provided features and either a provided natural 
graph structure for the dataset or a graph generated using K-NN. 
The second order graph property leveraged is the shortest path.
The higher-order graph properties leveraged are the shortest path, the clique order, the eccentricity and
the connectivity.

Those distance learning approaches were originally defined for clustering applications.
In this case, we need training sets with some relevant clusters labels. Then the algorithm
will learn what is the best distance to recover the same kind of meaningful patterns in the test.

#### Use
The structure of both the second-order and the higher-order implementations are similar, they both include 
the files 'classify.py' and 'distance_learning.py'. The higher-order method also uses a 'graph.py' file to define the
graph notion to be used and the diverse functions to compute the needed graph properties and graph operations.
We describe next the files and functions for the higher-order case, the second-order case being built on the 
same template less the higher-order specific parameters.

- 'distance_learning.py': it provides in the main function an example on how to use the learn function to obtain the
higher-order metric and how to use it for classification. Only the 'learn' function is meant to be called, its 
parameters are:
  - clustering_list: ground-truth, list of labels for the samples for the K training sets
  - feature_list: tensor containing for each pair of samples their distance according to each metric we are learning on
  - T: number of projected subgradient rounds
  - C: coefficient for the convergence criterion, cf convergence function
  - max_it: maximum number of iterations in case convergence is not reached
  - alpha, beta, tau: coefficients of the constraints and of the regularization, to be tuned
  - speed: coefficient influencing the speed of convergence, weight the updates
  - update_name: name of the projection to be update at each iteration, in the higher-order case, the weights need to be positive
  - nn_numbers: size of the neighborhood to consider when buildind graphs based on the neighborhood
  - order: order to consider for the higher- order
  - cpu_nb_slaves: number of cpus for available for parallelization
  - regu: name of the regularization method, implemented methods are "lasso", "ridge" and "elasticnet"
  - init: possible pre-initialization of the weights for the second- order metrics (e.g. by first applying the second- order framework), empty list for no initialization
  - graphs: list of the graphs to consider for each of the K training sets, if empty the K- nearest neighbor method is used
  - cluster_metric : 0: no higher order, 1: higher-order of order "order" only, -1: cluster metric only, 2: higher-order and cluster metric
- 'classify.py': it performs the actual prediction using the higher-order distance previously learned.
The prediction function leverages a K-NN frameworks and can be used with various linkages (see the article
for more details). first the function 'to_centers_distance_matrix' has to be called with
  - distance: an int characterizing the type of distance we want to compute from the list "pearson", "spearman", 
"kendall", "euclidean", "cosine", "pearson_abs", "spearman_abs", "kendall_abs"
  - data_train: the training dataset
  - data_test: the testing dataset
  - w: the learned weights
  - nn_numbers: the number of nearest neighbors to consider
  - centers: the center points of the classes
  - clusters: the classes
  - order: the order
  - has_vectors: boolean to characterize if supplemental weighted distances between the feature vectors have to be computed
  - path: path to save the distance to
  - init: initial vector of weights
  - graph_train: graph of hte data defined on the training dataset
  - cluster_metric: distance to be used in the K-NN

  Then, the function classify can be called to perform the actual prediction and assess the results with the parameters:
  - clustering_train: labels of the training set
  - clustering_text: labels of the testing set
  - indexes_set: indexes of the distance matrix on which the classification has ot be performed
  - distances_list: previously learned distances
- 'graph.py': it defines the graph class based on networkx functions. 

### Clustering
#### Description
This part implements a complete clustering pipeline.
All the clustering algorithms from scikit-learn are leveraged.
All possible unsupervised assessment metrics from scikit-learn are implemented, in addition an implementation of the Dunn's 
Index is leveraged.
An automatic model tuning is provided.
A pipeline for assessment and writing the results in a csv file is proposed.
This pipeline can also be used as a feature selection framework, the main function saves the centers of the clusters
which can be used as a signature of features. This technique has been used in [Battistella 2021](https://doi.org/10.1109/tcbb.2021.3123910).
#### Use
- 'main_clustering.py': main function implementing an end-to-end pipeline to leverage all the clustering functions.
- 'clustering_algorithms.py': Implement an interface to all scikit-learn clustering functions and assessment metric.
The tuning function search for the best parameters of each function, the best distance and the best number of clusters.
The test function take as input the data, the parameters of the functions and the template name to save the results. It
is meant to be used to cluster the test set once the best parameters have been found using the tuning function. However,
it can be used on the training set, if one wants to manually choose the parameters of the algorithms.
- 'dunn.py': Implements the Dunn's metric, common to assess clusters relevance but not implemented in scikit-learn.
Call its dunn function with the clustering and the distance matrix as parameters to get the metric on this clustering.

### LP-Stability
#### Description
Lp-Stability is a center-based clustering approach defined in [Komodakis 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.143.9220).
It relies on linear programming and dual theory to find stable cluster centers minimizing the distance to other points and
resilient to noise.
The only hyperparameter of this algorithm is a notion of penalty which can be specified point-wise and will influence the
number of centers, and so of clusters, discovered by the method.
#### Use
- 'lp_stability.py': given a distance matrix between the points, a vector penalty and the input_name where to save the files,
lp_stab will call the lp_stability_wrapper.py to launch the c++ code of the LP-Stability algorithm.
- 'lp_stability_wrapper.py': wrapper to perform the writting of the input files and the reading of the output files
- 'clustering.cpp', 'clustering.h': c++ code and header of the LP-Stability algorithm.