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
    result_object = Result(result_path, IS_TEMP, const_dir=True, extra_to_name='the_best_comp_' + EXTRA_TO_NANE)

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    all_algorithems = [
        #(lambda: FixLayerTwoNetwork(False, LR_FIX, R, use_batch=True), "Convex NN", 'b', "o"),
        (lambda: FixLayerTwoNetwork(False, LR_FIX, R, use_batch=True, use_crossentropy=True, xavier_init=False), "Convex, Standard init", 'b', "o", True),
        (lambda: FixLayerTwoNetwork(False, LR_FIX, R, sigma=SIGMA_LARGE, use_batch=True, use_crossentropy=True, xavier_init=False), "Convex, Large init", 'g', "s", True),
        (lambda: TwoLayerNetwork(R, LR_STA, use_batch=True, use_crossentropy=True, xavier_init=True), "Standard, Standard  init", 'r', "+", False),
        #(lambda: NTKsvn(R), "NTK svn", 'g', "s"),
        (lambda: mariano(), "mariano", 'm', "H", False)
    ]
    

    result_object.save_const_file()

    if FULL:
        print("Running with full config so generate all combinations for D={}".format(D))
        all_combinations = get_all_combinations()
        X_full = np.array(all_combinations, dtype=TYPE)
        Y_full = np.array([readonce.get_label(x) for x in X_full], dtype=TYPE)
        test_set = (X_full, Y_full)
        #train_list_size = [len(all_combinations) - i for i in REMOVE_SAMLPE_RANGE]
    else:
        X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
        test_set = (X_test, Y_test)
    train_list_size = TRAIN_SIZE_LIST

    result_vec, round_num, train_list_location = result_object.load_state(all_algorithems, train_list_size)

    for k in range(round_num, NUM_OF_RUNNING):
        for i in range(train_list_location, len(train_list_size)):
            set_size = train_list_size[i]
            X_train = get_random_init_uniform_samples(set_size, D)
            Y_train = np.array([readonce.get_label(x) for x in X_train], dtype=TYPE)
            train_set = (X_train, Y_train)
            for j in range(len(all_algorithems)):
                algorithem = all_algorithems[j]
                print('Running algorithem: "{0}" with train set in size: {1}'.format(algorithem[1], set_size))
                network = algorithem[0]()
                train_result, algorithem_result = network.run(train_set, test_set) 
                if algorithem[4]:
                    result_object.cluster_graph(network, "{0}_{1}_{2} - ".format(k, set_size, algorithem[1]))
                result_vec[k][j][i] = algorithem_result 
                result_object.save_state(result_vec, k, i)
        train_list_location = 0 
    result_object.comp_save_graph(result_vec, all_algorithems, train_list_size)
    

main() 
