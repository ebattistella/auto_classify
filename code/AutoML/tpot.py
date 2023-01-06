####################################
### Author: Enzo Battistella
### Date: 3/15/2022
####################################
### Keywords: AutoML; TPOT; Training;
### Description: <Train the autoML technique TPOT: epistasislab.github.io/tpot/>
### Input: training data and ground truth, optional: path for saving the results, a seed, a number of cross_validation
###        repeats, the size of the population
### Ouput: a python file with the pipeline is saved, return the trained pipeline
###################################

import preprocessing
from tpot import TPOTClassifier
import pandas as pd
import os
import numpy as np
from math import floor

warnings.warn = warn


def training(x_train, y_train, path="test", seed=0, cv_num=10, pop=50):
    np.random.seed(seed)

    if os.path.exists(path + "_tpot.csv"):
        return
    tpot = TPOTClassifier(generations=cv_num, population_size=pop, verbosity=2,
                          random_state=seed, scoring='balanced_accuracy')
    tpot.fit(x_train, y_train)
    print(tpot.score(x_train, y_train))
    tpot.export(tpot_directory + name + '.py')
    return tpot


# Example of main function to train TPOT
# A thorough evaluation can be performed using the function test_results in classification.py
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    x = pd.DataFrame(iris.data[:, :2], columns=["Feature 0", "Feature 1"])
    y = pd.DataFrame(iris.target)

    path = "./iris"
    cv_num = 10
    pop = 50
    seed = 0

    training(x, y, path, seed, cv_num, pop)
