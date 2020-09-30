import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, extra_to_name='plot')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    for round_num in range(NUM_OF_RUNNING):
        print("Running round {0} with train set in size {1}".format(round_num, TRAIN_SET_SIZE))
        all_combinations = get_all_combinations()
        X = np.array(all_combinations, dtype=TYPE)
        #X = get_random_init_uniform_samples(TRAIN_SET_SIZE, D)
        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        #positive_X = np.array([x for x, y in zip(X, Y) if y == POSITIVE], dtype=TYPE)
        #B_init = np.array([calc_bais_threshold(x, readonce, 0, D) for x in positive_X], dtype=TYPE)
        #W_init = positive_X + SIGMA*get_random_init_uniform_samples(positive_X.shape[0], D)
        #B_init = B_init + SIGMA*get_random_init_uniform_samples(1, D).T[0]
        train_set = (X, Y)
        #network = FixLayerTwoNetwork(False, LR, 0, W_init=W_init , B_init=B_init)
        network = FixLayerTwoNetwork(False, LR, R)
        network.run(train_set, train_set)
        import IPython; IPython.embed()
        result_object.cluster_graph(network, str(round_num))
main() 
