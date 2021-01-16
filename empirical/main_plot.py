import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *
from NTK_svn import *
from NTK_network import *
from mariano import *

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, extra_to_name='plot')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    #noise_size = D - sum(DNF)

    for round_num in range(NUM_OF_RUNNING):
        print("Running round {0} with train set in size {1}".format(round_num, TRAIN_SIZE))
        X = get_random_init_uniform_samples(TRAIN_SIZE, D)
        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        #all_combinations = get_all_combinations()
        #X = np.array(all_combinations, dtype=TYPE)
        #Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
        train_set = (X, Y)
        test_set = (X_test, Y_test)

        #network = FixLayerTwoNetwork(False, LR, R)
        #network.run(train_set, test_set)
        #network = NTKNetwork(False, LR, R)
        #network.run(train_set, test_set)
        #result_object.cluster_graph(network, str(round_num))
        #network = FixLayerTwoNetwork(False, LR, R)
        #network.run(train_set, test_set)

        network = mariano()
        network.run(train_set, test_set)

        import IPython; IPython.embed()     
main() 
