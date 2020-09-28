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
    result_object = Result(result_path, IS_TEMP, extra_to_name='two_layer_reconstraction')

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)
    result_vec = np.zeros([NUM_OF_RUNNING, len(TRAIN_SIZE_LIST)])

    for round_num in range(NUM_OF_RUNNING):
        for i, train_set_size in enumerate(TRAIN_SIZE_LIST):
            print("Running round {0} with train set in size {1}".format(round_num, train_set_size))
            X = get_random_init_uniform_samples(train_set_size, D)
            Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
            train_set = (X, Y)
            network = TwoLayerNetwork(R, LR)
            network.run(train_set, train_set)
            W_fold = np.zeros(network.W.shape)
            B_W_fold = np.zeros(network.B_W.shape)
            U_fold = np.ones(network.U.shape)
            for j in range(network.r):
                W_fold[j] = network.W[j] * np.abs(network.U[j])
                B_W_fold[j] = network.B_W[j] * np.abs(network.U[j])
                U_fold[j] = U_fold[j] * np.sign(network.U[j])
            fold_network = TwoLayerNetwork(R, LR, W_init=W_fold, U_init=U_fold, B_W_init=B_W_fold, B_U_init=network.B_U)
            for prune_factor in PRUNE_FACTOR_RANGE:
                import IPython; IPython.embed()
                flag = False
                above_mean_indexes = find_indexes_above_half_of_max(fold_network, 1, prune_factor)
                if len(above_mean_indexes) > 0:
                    W_prone = fold_network.W[above_mean_indexes]
                    B_W_prone = fold_network.B_W[above_mean_indexes]
                    U_prone = fold_network.U[above_mean_indexes]
                    prone_network = TwoLayerNetwork(R, LR, W_init=W_prone, U_init=U_prone, B_W_init=B_W_prone, B_U_init=network.B_U)
                    for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
                        if np.sum(U_prone) == U_prone.shape[0]:
                            if check_reconstraction(prone_network, readonce, noise_size, 1, reconstraction_factor):
                                #import IPython; IPython.embed()
                                print("Reconstraction seecced with prune_factor: {0} and reconstraction_factor: {1}".format(prune_factor, reconstraction_factor))
                                result_vec[round_num][i] = 1
                                flag = True
                                break
                    if flag:
                        break
            result_object.save_result_to_pickle('result.pkl', result_vec)
    result_vec_mean = np.mean(result_vec, axis=0)
    result_object.save_reconstraction_graph(result_vec_mean)
    
    
main() 
