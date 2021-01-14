import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *
from NTK_svn import *

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, const_dir=True)

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    all_algorithems = [
        (lambda: FixLayerTwoNetwork(False, LR, R), "Fix - Gaussion", 'bo'),
        #(lambda: TwoLayerNetwork(R, LR), "Regular - Gaussion", 'r.')
        (lambda:  NTKsvn(R), "NTK", 'r.')
    ]
    result_vec, round_num, train_list_location = result_object.load_state(all_algorithems)


    if FULL:
        print("Running with full config so generate all combinations for D={}".format(D))
        all_combinations = get_all_combinations()
        X_full = np.array(all_combinations, dtype=TYPE)
        Y_full = np.array([readonce.get_label(x) for x in X_full], dtype=TYPE)
        test_set = (X_full, Y_full)
    else:
        X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
        test_set = (X_test, Y_test)
    for k in range(round_num, NUM_OF_RUNNING):
        for i in range(train_list_location, len(TRAIN_SIZE_LIST)):
            flag = True
            set_size = TRAIN_SIZE_LIST[i]
            while flag:
                X_train = get_random_init_uniform_samples(set_size, D)
                Y_train = np.array([readonce.get_label(x) for x in X_train], dtype=TYPE)
                train_set = (X_train, Y_train)
                for j in range(len(all_algorithems)):
                    flag = True
                    algorithem = all_algorithems[j]
                    print('Running algorithem: "{0}" with train set in size: {1}'.format(algorithem[1], set_size))
                    result_object.save_state(result_vec, k, i)
                    l = 0
                    flag_2 = True
                    while l < ATTEMPT_NUM and flag_2:
                        l += 1
                        network = algorithem[0]()
                        train_result, algorithem_result = network.run(train_set, test_set) 
                        if train_result == 0:
                            flag = False
                            flag_2 = False
                    if flag:
                        print("Couldn't reach global minimum. Try again.")
                        break
                    result_vec[k][j][i] = algorithem_result 
        train_list_location = 0 
    result_object.comp_save_graph(result_vec)
    

main() 
