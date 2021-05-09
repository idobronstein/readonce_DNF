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
    result_object = Result(result_path, IS_TEMP, extra_to_name='parity_comp', const_dir=True)

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)

    all_readonce = [["", "DNF", 'r', "^"], ["", "parity",'b', "o"]]
    readonce = ReadOnceDNF(DNF)
    
    result_vec_test, round_num, train_list_location = result_object.load_state(all_readonce, TRAIN_SIZE_LIST)
    result_vec_train, _, _ = result_object.load_state(all_readonce, TRAIN_SIZE_LIST, "_train")

    result_object.save_const_file()

    for k in range(round_num, NUM_OF_RUNNING):
        for i in range(train_list_location, len(TRAIN_SIZE_LIST)):
            set_size = TRAIN_SIZE_LIST[i]
            #X = np.array(get_all_combinations(), dtype=TYPE)
            X = get_random_init_uniform_samples(set_size, D)
            #X_test = get_random_init_uniform_samples(TEST_SIZE, D)
            X_test = np.array(get_all_combinations(), dtype=TYPE)
            for j in range(2):
                if j == 0:
                    print('Running DNF with train set in size: {0}'.format(set_size))
                    Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
                    Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
                else:
                    Y = np.array([get_parity_label(x[:PARITY_SIZE]) for x in X], dtype=TYPE)
                    Y_test = np.array([get_parity_label(x[:PARITY_SIZE]) for x in X_test], dtype=TYPE)
                train_set = (X, Y)
                test_set = (X_test, Y_test)
                network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True)
                train_result, dnf_test_result = network.run(train_set, test_set)
                result_object.cluster_graph(network, "{0}_{1}_{2}- ".format(k, set_size,j))
                result_vec_test[k][j][i] = dnf_test_result 
                result_vec_train[k][j][i] = train_result 
                result_object.save_state(result_vec_test, k, i)
                result_object.save_state(result_vec_train, k, i, "_train")
        train_list_location = 0 
    result_object.comp_save_graph(result_vec_test, all_readonce, TRAIN_SIZE_LIST)
    result_object.comp_save_graph(result_vec_train, all_readonce, TRAIN_SIZE_LIST, "_train")
    
main() 
