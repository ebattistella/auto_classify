import lp_stability_wrapper
import time
import uuid,os
from sklearn.metrics import pairwise_distances

class Lpstability:
    def __init__(self, dict_params):
        self.metric_ = dict_params['metric']
        self.penalty_ = dict_params['penalty']
        self.name_ = dict_params['name']
    def fit(self, X, Y=0):
        x_trans = pairwise_distances(X, metric=self.metric_)
        self.labels_, self.exemplars_ = lp_stab(x_trans, self.penalty_, self.name_ + str(uuid.uuid4()))

# Interface code calling the wrapper for using the c++ code of LP-Stability
# distance: square np.array with the distances between data points to cluster
# penalty:
def lp_stab(distance, penalty, input_name):
    npoints = distance.shape[0]

    #solver = file_location + 'clustering.exe'
    solver = input_name + 'clustering.exe'
    input_f = input_name + '.txt'
    output_f = input_name + 'output.txt'

    fichier = open(output_f, "w+")
    fichier.close()

    X = None
    max_iter = 30

    #start = time.time()
    lp_stability_wrapper.create_big_inputfile(X,npoints,input_f,
                                               dist = distance,
                                               penalty = penalty,
                                               max_iter = max_iter) 

    lp_stability_wrapper.execute_cpp(solver,input_f,output_f)
    flabel, best_exemplars = [],[]
    (best_num_exemplars,best_exemplars,num_iter,
    primal_function,dual_function,flabel) = lp_stability_wrapper.read_outputfile(output_f,npoints)
    os.remove(input_f)
    os.remove(output_f)
    return flabel, best_exemplars