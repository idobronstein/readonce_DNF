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
    #readonce = ReadOnceDNF(specifiec_DNF=[[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
    for round_num in range(NUM_OF_RUNNING):
        print("Running round {0} with train set in size {1}".format(round_num, TRAIN_SIZE))
        X = get_random_init_uniform_samples(TRAIN_SIZE, D)
        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        all_combinations = get_all_combinations()
        #X = np.array(random.sample(all_combinations, len(all_combinations)), dtype=TYPE)
        #Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)

        #X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        X_test = np.array(all_combinations, dtype=TYPE)
        Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
        train_set = (X, Y)
        test_set = (X_test, Y_test)

        
        #network = TwoLayerNetwork(R, LR_STA, use_crossentropy=True)
        #network = NTKNetwork(False, LR, R)
        #network = mariano()
        #network = NTKsvn(R)
        #network = TwoLayerNetwork(R, LR, use_batch=True, use_crossentropy=True, sigma_1=SIGMA_1, sigma_2=SIGMA_2)

        recovery_network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, xavier_init=False)
        recovery_network.run(train_set, test_set)
        result_object.cluster_graph(recovery_network, "recovery - ")

        X_positive = np.array([x for x in X if readonce.get_label(x) == POSITIVE], dtype=TYPE)
        W_memo = np.zeros([R, D], dtype=TYPE)
        B_memo = np.ones([R], dtype=TYPE) * (- D + 2 - 0.1)
        num_of_positive = X_positive.shape[0]
        for i in range(R):
            W_memo[i] = X_positive[i % num_of_positive]

        memo_network = FixLayerTwoNetwork(False, LR, 0, W_init=W_memo, B_init=B_memo, B0_init=-1 * np.ones([1], dtype=TYPE), use_crossentropy=True, xavier_init=False)
        memo_network.run(train_set, test_set)

        result_object.cluster_graph(memo_network, "memorization - ")
    
main() 
