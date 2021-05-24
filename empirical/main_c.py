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
    result_object = Result(result_path, IS_TEMP, extra_to_name='angle')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)
    result_vec = np.zeros([NUM_OF_RUNNING, len(TRAIN_SIZE_LIST)])

    result_object.save_const_file()

    result_vec = np.zeros([2, len(TRAIN_SIZE_LIST)])

    for i, train_set_size in enumerate(TRAIN_SIZE_LIST):
        print("Running  with train set in size {0}".format(train_set_size))
        X = get_random_init_uniform_samples(train_set_size, D)
        #all_combinations = get_all_combinations()
        #X = np.array(random.sample(all_combinations, len(all_combinations)), dtype=TYPE)
        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
        train_set = (X, Y)
        W_init = np.array(SIGMA * np.random.randn(R, D), dtype=TYPE)
        B_init = np.zeros([R], dtype=TYPE) 
        C_init = np.zeros([1], dtype=TYPE)
        network = FixLayerTwoNetwork(False, LR, 0, W_init=W_init, B_init=B_init, B0_init=C_init, use_crossentropy=True, use_batch=True)
        network.run(train_set, train_set)
        network_fix_c = FixLayerTwoNetwork(False, LR, 0, W_init=W_init, B_init=B_init, B0_init=C_init, use_crossentropy=True, use_batch=True, without_B0=True)
        network_fix_c.run(train_set, train_set)
        res = np.array([angle_between(network.W[i], network_fix_c.W[i]) for i in range(R)], dtype=TYPE)
        result_vec[0][i] = np.mean(res)
        result_vec[1][i] = np.std(res)
        result_object.save_result_to_pickle('result.pkl', result_vec)

    result_object.angle_save_graph(result_vec, TRAIN_SIZE_LIST)
    
main() 
