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

def create_dnf_big_overlap(bais_dnf, number_of_terms, term_size, overlap_size):
    dnf = copy.deepcopy(bais_dnf.DNF)
    for i in range(1, number_of_terms, 2):
        term = np.zeros([D], dtype=TYPE)
        term[term_size * i - overlap_size : term_size * (i+1) - overlap_size] = [1] * term_size
        dnf[i] = term
    return ReadOnceDNF(specifiec_DNF=dnf)

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
    dnf_first = ReadOnceDNF(DNF)
    #dnf_second = create_dnf_big_overlap(dnf_first, len(DNF), DNF[0], OVERLAP_SIZE)
    for round_num in range(NUM_OF_RUNNING):
        for i in [6000, 15000, 40000]:
            print("Running round {0} with train set in size {1}".format(round_num, i))
            X = get_random_init_uniform_samples(i, D)
            X_test = get_random_init_uniform_samples(TEST_SIZE, D)
            #X_test = np.array(all_combinations, dtype=TYPE)
            #all_combinations = get_all_combinations()
            #X = np.array(random.sample(all_combinations, len(all_combinations)), dtype=TYPE)
            #Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
            
            Y_test = np.array([dnf_first.get_label(x) for x in X_test], dtype=TYPE)
            Y = np.array([dnf_first.get_label(x) for x in X], dtype=TYPE)
            train_set = (X, Y)
            test_set = (X_test, Y_test)
            network_first = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, use_batch=True)
            train_result, algorithem_result = network_first.run(train_set, test_set) 
            result_object.cluster_graph(network_first, "cluster_{0}_{1} - ".format(round_num, i))
            result_object.save_result_to_pickle("W_{0}_{1}.pkl".format(round_num, i), network_first.W)
            result_object.save_result_to_pickle("result_{0}_{1}.pkl".format(round_num,  i), (train_result, algorithem_result))
        
        large_init = 40000
        print("Running round {0} with train set in size {1} - large init".format(round_num, large_init))
        #X = get_random_init_uniform_samples(large_init, D)
        #X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        #X_test = np.array(all_combinations, dtype=TYPE)
        #all_combinations = get_all_combinations()
        #X = np.array(random.sample(all_combinations, len(all_combinations)), dtype=TYPE)
        #Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        Y_test = np.array([dnf_first.get_label(x) for x in X_test], dtype=TYPE)
        Y = np.array([dnf_first.get_label(x) for x in X], dtype=TYPE)
        train_set = (X, Y)
        test_set = (X_test, Y_test)
        network_first = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, use_batch=True, sigma=SIGMA_LARGE)
        train_result, algorithem_result = network_first.run(train_set, test_set) 
        #result_object.cluster_graph(network_first, "cluster_large_init_{0} - ".format(round_num))
        result_object.save_result_to_pickle("W_large_init_{0}.pkl".format(round_num), network_first.W)
        result_object.save_result_to_pickle("result_large_init_{0}.pkl".format(round_num), (train_result, algorithem_result))

        #Y_test = np.array([dnf_second.get_label(x) for x in X_test], dtype=TYPE)
        #Y = np.array([dnf_second.get_label(x) for x in X], dtype=TYPE)
        #train_set = (X, Y)
        #test_set = (X_test, Y_test)
        #network_second = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, use_batch=True)
        #train_result, algorithem_result = network_second.run(train_set, test_set) 
        #result_object.cluster_graph(network_second, "overlap_{0} - ".format(round_num))
        #result_object.save_result_to_pickle("W_overlap_{0}.pkl".format(round_num), network_second.W)
        #result_object.save_result_to_pickle("result_overlap_{0}.pkl".format(round_num), (train_result, algorithem_result))

        #network = TwoLayerNetwork(R, LR_STA, use_crossentropy=True)
        #network = NTKNetwork(False, LR, R)
        #network = mariano()
        #network = NTKsvn(R)
        #network = TwoLayerNetwork(R, LR, use_batch=True, use_crossentropy=True, sigma_1=SIGMA_1, sigma_2=SIGMA_2)

        #recovery_network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, xavier_init=False)
        #recovery_network.run(train_set, test_set)
        #result_object.cluster_graph(recovery_network, "recovery - ")
        #result_object.save_result_to_pickle("W_reco.pkl", recovery_network.W)
#
        #X_positive = np.array([x for x in X if readonce.get_label(x) == POSITIVE], dtype=TYPE)
        #W_memo = np.zeros([R, D], dtype=TYPE)
        #B_memo = np.ones([R], dtype=TYPE) * (- D + 2 - 0.1)
        #num_of_positive = X_positive.shape[0]
        #for i in range(R):
        #    W_memo[i] = X_positive[i % num_of_positive]
#
        #memo_network = FixLayerTwoNetwork(False, LR, 0, W_init=W_memo, B_init=B_memo, B0_init=-1 * np.ones([1], dtype=TYPE), use_crossentropy=True, xavier_init=False)
        #memo_network.run(train_set, test_set)
#
        #result_object.cluster_graph(memo_network, "memorization - ")
        #result_object.save_result_to_pickle("W_memo.pkl", memo_network.W)
main() 
