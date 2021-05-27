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

    X = get_random_init_uniform_samples(TRAIN_SIZE, D)
    #all_combinations = get_all_combinations()
    #X = np.array(random.sample(all_combinations, len(all_combinations)), dtype=TYPE)
    Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
    train_set = (X, Y)
    W_init = np.array(SIGMA * np.random.randn(R, D), dtype=TYPE)
    B_init = np.zeros([R], dtype=TYPE) 
    C_init = np.zeros([1], dtype=TYPE)
    network = FixLayerTwoNetwork(False, LR, 0, W_init=W_init, B_init=B_init, B0_init=C_init, use_crossentropy=True, use_batch=True)
    train_result_normal, algorithem_result_normal = network.run(train_set, train_set)
    network_fix_c = FixLayerTwoNetwork(False, LR, 0, W_init=W_init, B_init=B_init, B0_init=C_init, use_crossentropy=True, use_batch=True, without_B0=True)
    train_result_fix, algorithem_result_fix = network_fix_c.run(train_set, train_set)

    leaves_index = cluster_network(network)
    result_object.cluster_graph(network, "cluster_normal - ", leaves_index=leaves_index)
    result_object.save_result_to_pickle("W_normal.pkl", network.W)
    result_object.save_result_to_pickle("result_normal.pkl", (train_result_normal, algorithem_result_normal))
    result_object.cluster_graph(network_fix_c, "cluster_fix_c - ", leaves_index=leaves_index)
    result_object.save_result_to_pickle("W_fix_c.pkl", network_fix_c.W)
    result_object.save_result_to_pickle("result_fix_c.pkl", (train_result_normal, algorithem_result_normal))


main() 
