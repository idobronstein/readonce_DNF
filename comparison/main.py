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
    result_object = Result(result_path, IS_TEMP)

    print("Generate all partitions")
    all_partitions = get_all_balanced_partitions()
    all_partitions.remove([D])

    print("Generate all combinations")
    all_combinations = get_all_combinations()
    X = np.array(all_combinations, dtype=FLOAT_TYPE)

    all_algorithems = [
        (FixLayerTwoNetwork(False, LR_TWO_LAYER_REGULAR, R_GAUSS_FIX_LAYER), "Fix Layer Two - Gaussion Init", '.'),
        (FixLayerTwoNetwork(True, LR_FIX_LAYER), "Fix Layer Two - Epsilon Init", 'o'),
        (TwoLayerNetwork(R_GAUSS_TWO_LAYER_REGULAR, LR_TWO_LAYER_REGULAR), "Regular Two Layer - Gaussion Init", 'x')
    ]

    for partition in all_partitions:
        if 1 in partition:
            continue

        print("Start a run for: {0}".format(partition))        
        run_name = '_'.join([str(i) for i in partition])  
        result_vec = np.zeros([len(all_algorithems), len(SAMPLE_PROB_LIST)])

        readonce = ReadOnceDNF(partition)
        Y = np.array([readonce.get_label(x) for x in X], dtype=FLOAT_TYPE)

        for i, prob in enumerate(SAMPLE_PROB_LIST):
            X_downsample, Y_downsample = downsampling(X, Y, prob)
            train_set = (X_downsample, Y_downsample)
            test_set = (X, Y)

            for j, algorithem in enumerate(all_algorithems):
                print('Running algorithem: "{0}" with prob: {1}'.format(algorithem[1], prob))

                algorithem_result = algorithem[0].run(train_set, test_set)
                result_vec[j][i] = algorithem_result 

        result_object.save_graph(run_name, all_algorithems, result_vec)


main() 
