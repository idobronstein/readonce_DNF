import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from network import *
from result import * 


def run_network(network, X, Y, run_name, result_object, readonce):
    step = 0
    global_minimum_point, local_minimum_point = False, False
    while not global_minimum_point and not local_minimum_point and step < MAX_STEPS:
        display = False
        if step % PRINT_STEP_JUMP == 0 and step > 0:
            result_name = os.path.join(run_name, str(step))
            save_results(global_minimum_point, result_name, result_object, network, readonce, X, Y)
            print("Step number: {0}, ".format(step), end ="")
            display = True
        global_minimum_point, local_minimum_point = network.update_network(X, Y, LR, display)
        step += 1
    if local_minimum_point and not global_minimum_point:
        print("Got to local minimums")
    return global_minimum_point

def save_results(minimum_point, name, result_object, network, readonce, X, Y):
    result_object.save_run(name, network, readonce, minimum_point, is_perfect_classification(network, X, Y))
    result_object.generate_cluster_graph(name, network)
    result_object.summarize_alined_terms(name, network, readonce, X)

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    result_object = Result(result_path, IS_TEMP)

    print("Making result object in the path: {0}".format(result_path))
    print("Generate all partitions")
    all_partitions = get_all_balanced_partitions()
    all_partitions.remove([D])
    print("Generate all combinations")
    all_combinations = get_all_combinations()
    r = len(all_combinations)

    X = np.array(all_combinations, dtype=FLOAT_TYPE)

    for partition in all_partitions:
        if 1 in partition:
            continue
        run_name = '_'.join([str(i) for i in partition])  
        backup_name = os.path.join(run_name, BACKUP_DIR)
        result_object.create_dir(run_name)
        result_object.create_dir(backup_name)

        readonce = ReadOnceDNF(partition)
        Y = np.array([readonce.get_label(x) for x in X], dtype=FLOAT_TYPE)
        
        W_init = np.array(all_combinations, dtype=FLOAT_TYPE) * SIGMA
        B_init = np.zeros([r], dtype=FLOAT_TYPE)
        network = Network(W_init, B_init)

        result_name = os.path.join(run_name, BACKUP_DIR)
        minimum_point = run_network(network, X, Y, backup_name, result_object, readonce)

        save_results(minimum_point, os.path.join(run_name, ORIGINAL_FINAL_DIR), result_object, network, readonce, X, Y)

        if minimum_point:
            print("We got to minimum point - Prone the network")

            aligend_indexes = get_all_algined_indexes(network, readonce, X)
            W_prone = network.W[aligend_indexes]
            B_prone = network.B[aligend_indexes]
            prone_network = Network(W_prone, B_prone)

            save_results(minimum_point, os.path.join(run_name, PRONE_DIR), result_object, prone_network, readonce, X, Y)

            if is_perfect_classification(network, X, Y):
                print("The prone network has a perfect classification")
            else:
                print("The prone network has wornd! it doesn't classify good")


main()