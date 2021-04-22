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
    result_object = Result(result_path, IS_TEMP, extra_to_name='overlap')

    dnf_1 = ReadOnceDNF(specifiec_DNF=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    dnf_2 = ReadOnceDNF(specifiec_DNF=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0]])
    dnf_3 = ReadOnceDNF(specifiec_DNF=[[1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0]])
    all_DNF = [dnf_1]
    result_vec = result_vec = np.zeros([len(all_DNF), NUM_OF_RUNNING, len(TRAIN_SIZE_LIST)])

    for j, dnf in enumerate(all_DNF):
        print("Running for {0}".format(dnf.DNF))
        #noise_size = D - np.sum(dnf.DNF)
        noise_size = 0
        for round_num in range(NUM_OF_RUNNING):
            for i, train_set_size in enumerate(TRAIN_SIZE_LIST):
                print("Running round {0} with train set in size {1}".format(round_num, train_set_size))
                X = get_random_init_non_uniform_samples(train_set_size, 0.9, D)
                Y = np.array([dnf.get_label(x) for x in X], dtype=TYPE)
                train_set = (X, Y)
                network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True)
                network.run(train_set, train_set)
                for prune_factor in PRUNE_FACTOR_RANGE:
                    flag = False
                    above_mean_indexes = find_indexes_above_half_of_max(network, 1, prune_factor)
                    if len(above_mean_indexes) > 0:
                        W_prone = network.W[above_mean_indexes]
                        B_prone = network.B[above_mean_indexes]
                        prone_network = FixLayerTwoNetwork(False, LR, W_init=W_prone, B_init=B_prone)    
                        for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
                            if check_reconstraction(prone_network, dnf, noise_size, 1, reconstraction_factor):
                                print("Reconstraction seecced with prune_factor: {0} and reconstraction_factor: {1}".format(prune_factor, reconstraction_factor))
                                result_vec[j][round_num][i] = 1
                                flag = True
                                break
                        if flag:
                            break
                result_object.save_result_to_pickle('result.pkl', result_vec)
    import ipdb; ipdb.set_trace()
    result_vec_mean = np.mean(result_vec, axis=0)
    result_object.save_reconstraction_graph(result_vec_mean)
    
    
main() 
