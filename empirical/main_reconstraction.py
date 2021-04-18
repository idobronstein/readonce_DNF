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
    result_object = Result(result_path, IS_TEMP, extra_to_name='reconstraction')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)
    result_vec = np.zeros([NUM_OF_RUNNING, len(TRAIN_SIZE_LIST)])

    result_object.save_const_file()

    for round_num in range(NUM_OF_RUNNING):
        for i, train_set_size in enumerate(TRAIN_SIZE_LIST):
            print("Running round {0} with train set in size {1}".format(round_num, train_set_size))
            X = get_random_init_uniform_samples(train_set_size, D)
            Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
            train_set = (X, Y)
            network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True)
            network.run(train_set, train_set)
            result_object.cluster_graph(network, "{0}_{1}- ".format(round_num, train_set_size))
            for prune_factor in PRUNE_FACTOR_RANGE:
                flag = False
                above_mean_indexes = find_indexes_above_half_of_max(network, 1, prune_factor)
                if len(above_mean_indexes) > 0:
                    W_prone = network.W[above_mean_indexes]
                    B_prone = network.B[above_mean_indexes]
                    prone_network = FixLayerTwoNetwork(False, LR, W_init=W_prone, B_init=B_prone)    
                    for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
                        if check_reconstraction(prone_network, readonce, noise_size, 1, reconstraction_factor):
                            print("Reconstraction seecced with prune_factor: {0} and reconstraction_factor: {1}".format(prune_factor, reconstraction_factor))
                            result_vec[round_num][i] = 1
                            flag = True
                            break
                    if flag:
                        break
            result_object.save_result_to_pickle('result.pkl', result_vec)
    #result_vec_mean = np.mean(result_vec, axis=0)
    #result_object.save_reconstraction_graph(result_vec_mean)
    
    
main() 
