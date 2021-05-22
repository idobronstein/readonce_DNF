import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from result import *
from fix_layer_2_netowrk import *
from two_layer_network import *


def create_all_dnfs(bais_dnf, number_of_terms, term_size):
    all_dnfs = [bais_dnf]
    diffrent_term_size = term_size - 1
    for i in range(1, number_of_terms):
        dnf = copy.deepcopy(all_dnfs[-1].DNF)
        term = np.zeros([D], dtype=TYPE)
        term[0] = 1
        term[1 + diffrent_term_size * (i-1) : 1 + diffrent_term_size *i] = [0] * diffrent_term_size
        term[1 + diffrent_term_size * i : 1 + diffrent_term_size * (i+1)  ] = [1] * diffrent_term_size
        dnf[i] = term
        all_dnfs.append(ReadOnceDNF(specifiec_DNF=dnf))
    return all_dnfs

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP, extra_to_name='overlap')

    all_DNF = create_all_dnfs(ReadOnceDNF(DNF), len(DNF), DNF[0])
    result_vec, round_num, train_list_location = result_object.load_state(all_DNF, [TRAIN_SIZE])

    X_test = get_random_init_uniform_samples(TEST_SIZE, D)

    for k in range(round_num, NUM_OF_RUNNING):
        X = get_random_init_uniform_samples(TRAIN_SIZE, D)
        for j, dnf in enumerate(all_DNF):
            print("Running for {0}".format(dnf.DNF))
            Y = np.array([dnf.get_label(x) for x in X], dtype=TYPE)
            Y_test = np.array([dnf.get_label(x) for x in X_test], dtype=TYPE)
            train_set = (X, Y)
            test_set = (X_test, Y_test)
            network = FixLayerTwoNetwork(False, LR, R, use_crossentropy=True, use_batch=True)
            train_result, algorithem_result = network.run(train_set, test_set)
            result_vec[k][j][0] = algorithem_result 
            result_object.save_state(result_vec, k, 0)
    result_object.dnfs_save_graph(result_vec)
    
    
main() 
