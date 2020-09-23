import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *

def load_state(all_algorithems):
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, 'rb') as f:
           all_state = pickle.load(f) 
           result_vec, round_num, train_list_location = all_state
        print("Restore state: round_num - {0}, train list location - {1}".format(round_num, train_list_location))
    else:
        result_vec = np.zeros([NUM_OF_RUNNING, len(all_algorithems), len(TRAIN_SIZE_LIST)])
        round_num = 0
        train_list_location = 0
    return result_vec, round_num, train_list_location

def save_state(result_vec, round_num, train_list_location):
    print("Saving state for: round_num - {0}, train list location - {1}".format(round_num, train_list_location, ))
    with open(STATE_PATH, 'wb+') as f:
        pickle.dump((result_vec, round_num, train_list_location), f) 

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP)

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    if FULL:
        print("Running with full config so generate all combinations for D={}".format(D))
        all_combinations = get_all_combinations()
        X_full = np.array(all_combinations, dtype=TYPE)
        Y_full = np.array([readonce.get_label(x) for x in X_full], dtype=TYPE)
    
    # Compere algorthims    
    all_algorithems = [
        (lambda: FixLayerTwoNetwork(False, LR, R), "Fix - Gaussion", 'bo'),
        (lambda: TwoLayerNetwork(R, LR), "Regular - Gaussion", 'r.')
    ]
    result_vec, round_num, train_list_location = load_state(all_algorithems)

    '''
    if FULL:
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
                    save_state(result_vec, k, i)
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
    result_vec = np.mean(result_vec, axis=0)
    result_object.save_graph(all_algorithems, result_vec)
    
    '''
    # Plot graphs
    if FULL:
        X, Y = X_full, Y_full
    else:
        X = get_random_init_uniform_samples(TRAIN_SIZE, D)
        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
    train_set = (X, Y)
    #network = FixLayerTwoNetwork(False, LR, R)
    network = FixLayerTwoNetwork(True, LR, R)
    network.run(train_set, train_set)
    result_object.angle_vs_norm_graph(network, readonce, X, noise_size)
    result_object.bias_graph(network, readonce, X, noise_size)
    result_object.cluster_graph(network)
    above_mean_indexes = find_indexes_above_half_of_max(network)
    W_prone = network.W[above_mean_indexes]
    B_prone = network.B[above_mean_indexes]
    prone_network = FixLayerTwoNetwork(False, LR, W_init=W_prone, B_init=B_prone)    
    result_object.cluster_graph(prone_network, add_to_name='prone_')
    import IPython; IPython.embed()
    assert check_reconstraction(prone_network, readonce, noise_size)
    
main() 
