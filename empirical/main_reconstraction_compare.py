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

def reconstract_convex_nn(network, readonce, noise_size):
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
                    return 1
    return 0

def reconstract_standard_nn(network ,readonce, noise_size):
    W_fold = np.zeros(network.W.shape)
    B_W_fold = np.zeros(network.B_W.shape)
    U_fold = np.ones(network.U.shape)
    for j in range(network.r):
        W_fold[j] = network.W[j] * np.abs(network.U[j])
        B_W_fold[j] = network.B_W[j] * np.abs(network.U[j])
        U_fold[j] = U_fold[j] * np.sign(network.U[j])
    fold_network = TwoLayerNetwork(R, LR, W_init=W_fold, U_init=U_fold, B_W_init=B_W_fold, B_U_init=network.B_U)
    for prune_factor in PRUNE_FACTOR_RANGE:
        flag = False
        above_mean_indexes = find_indexes_above_half_of_max(fold_network, 1, prune_factor)
        print(above_mean_indexes)
        if len(above_mean_indexes) > 0:
            W_prone = fold_network.W[above_mean_indexes]
            B_W_prone = fold_network.B_W[above_mean_indexes]
            U_prone = fold_network.U[above_mean_indexes]
            prone_network = TwoLayerNetwork(R, LR, W_init=W_prone, U_init=U_prone, B_W_init=B_W_prone, B_U_init=network.B_U)
            for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
                if np.sum(U_prone) == U_prone.shape[0]:
                    if check_reconstraction(prone_network, readonce, noise_size, 1, reconstraction_factor):
                        print("Reconstraction seecced with prune_factor: {0} and reconstraction_factor: {1}".format(prune_factor, reconstraction_factor))
                        return 1
    return 0

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, const_dir=True, extra_to_name='rec_' + EXTRA_TO_NANE)

    print("Start a run for: {0}".format(DNF))        
    run_name = '_'.join([str(i) for i in DNF]) 
    result_object.create_dir(run_name)
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    all_algorithems = [
        (lambda: FixLayerTwoNetwork(False, LR, R, use_crossentropy=True), "Convex NN CE", 'b', "o", reconstract_convex_nn),
        (lambda: TwoLayerNetwork(R, LR, use_crossentropy=True), "Standard NN CE", 'r', "+", reconstract_standard_nn)
    ]
    
    result_object.save_const_file()

    if FULL:
        print("Running with full config so generate all combinations for D={}".format(D))
        all_combinations = get_all_combinations()
        X_full = np.array(all_combinations, dtype=TYPE)
        Y_full = np.array([readonce.get_label(x) for x in X_full], dtype=TYPE)
        test_set = (X_full, Y_full)
        train_list_size = [len(all_combinations) - i for i in REMOVE_SAMLPE_RANGE]
    else:
        X_test = get_random_init_uniform_samples(TEST_SIZE, D)
        Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
        test_set = (X_test, Y_test)
        train_list_size = TRAIN_SIZE_LIST

    result_vec, round_num, train_list_location = result_object.load_state(all_algorithems, train_list_size)

    for k in range(round_num, NUM_OF_RUNNING):
        for i in range(train_list_location, len(train_list_size)):
            set_size = train_list_size[i]
            if FULL:
                X_train = np.array(random.sample(all_combinations, set_size), dtype=TYPE)
            else:
                X_train = get_random_init_uniform_samples(set_size, D)
            Y_train = np.array([readonce.get_label(x) for x in X_train], dtype=TYPE)
            train_set = (X_train, Y_train)
            for j in range(len(all_algorithems)):
                algorithem = all_algorithems[j]
                print('Running algorithem: "{0}" with train set in size: {1}'.format(algorithem[1], set_size))
                network = algorithem[0]()
                train_result, algorithem_result = network.run(train_set, test_set) 
                result_vec[k][j][i] = algorithem[4](network, readonce, noise_size)
                result_object.save_state(result_vec, k, i)
        train_list_location = 0 
    
main() 
