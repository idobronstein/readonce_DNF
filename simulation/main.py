import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from network import *
from result import * 


def run_network(network, X, Y, run_name, result_object, readonce, noise_size):
    step = 0
    global_minimum_point, local_minimum_point = False, False
    while not global_minimum_point and not local_minimum_point and step < MAX_STEPS:
        display = False
        if step % PRINT_STEP_JUMP == 0 and step > 0:
            result_name = os.path.join(run_name, str(step))
            save_results(global_minimum_point, result_name, result_object, network, readonce, X, Y, noise_size)
            print("Step number: {0}, ".format(step), end ="")
            display = True
        global_minimum_point, local_minimum_point = network.update_network(X, Y, LR, display)
        step += 1
    if local_minimum_point and not global_minimum_point:
        print("Got to local minimums")
    return global_minimum_point

def save_results(minimum_point, name, result_object, network, readonce, X, Y, noise_size):
    result_object.save_run(name, network, readonce, minimum_point, is_perfect_classification(network, X, Y))
    result_object.generate_cluster_graph(name, network)
    result_object.summarize_alined_terms(name, network, readonce, X, noise_size)

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    print("Making result object in the path: {0}".format(result_path))
    result_object = Result(result_path, IS_TEMP)
    
    print("Generate all combinations for D={}".format(D))
    all_combinations = get_all_combinations()
    r = len(all_combinations)
    X = np.array(all_combinations, dtype=TYPE)

    for dnf_size in range(2, D+1):
        noise_size = D - dnf_size

        print("Generate all balanced partitions in size {}".format(dnf_size))
        all_partitions = get_all_balanced_partitions(dnf_size)
        # remove the DNF of all 1 and the DNF with one term
        all_partitions = all_partitions[1:-1]

        if len(all_partitions) == 0:
            print("No relevant phrases. Skippinig..")
        else:
            for epsilon in range(UNIT, LR + UNIT, UNIT):
                print("Running for all dnf in size: {0} and initialization: {1}".format(dnf_size, epsilon))
                            
                result_object.set_result_path(dnf_size, epsilon)

                for partition in all_partitions:
            
                    print("Start a run for: {0}".format(partition))
                    
                    run_name = '_'.join([str(i) for i in partition])  
                    backup_name = os.path.join(run_name, BACKUP_DIR)
                    result_object.create_dir(run_name)
                    result_object.create_dir(backup_name)
            
                    readonce = ReadOnceDNF(partition)
                    Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
                    
                    W_init = np.array(all_combinations, dtype=TYPE) * epsilon
                    B_init = np.zeros([r], dtype=TYPE)
                    network = Network(W_init, B_init)
            
                    result_name = os.path.join(run_name, BACKUP_DIR)
                    minimum_point = run_network(network, X, Y, backup_name, result_object, readonce, noise_size)
            
                    save_results(minimum_point, os.path.join(run_name, ORIGINAL_FINAL_DIR), result_object, network, readonce, X, Y, noise_size)
            
                    if minimum_point:
                        print("We got to minimum point")
            
                        aligend_indexes = get_all_algined_indexes(network, readonce, X, noise_size)
                        if len(aligend_indexes) > 0:
                            print("Prone the network by aligment")
                            W_prone = network.W[aligend_indexes]
                            B_prone = network.B[aligend_indexes]
                            prone_network = Network(W_prone, B_prone)
                            save_results(minimum_point, os.path.join(run_name, PRONE_DIR), result_object, prone_network, readonce, X, Y, noise_size)
                            if is_perfect_classification(network, X, Y):
                                print("The prone network has a perfect classification")
                            else:
                                print("The prone network has wornd! it doesn't classify good")
                        else:
                            print("Skipping pruning the network by aligned, because there isn't aligned weights")
            
                        print("Prone the network by mean")
                        above_mean_indexes = find_indexes_above_half_of_max(network)
                        W_prone = network.W[above_mean_indexes]
                        B_prone = network.B[above_mean_indexes]
                        prone_network = Network(W_prone, B_prone)
                        save_results(minimum_point, os.path.join(run_name, PRONE_BY_MEAN_DIR), result_object, prone_network, readonce, X, Y, noise_size)
                        if set(above_mean_indexes).issubset(aligend_indexes):
                            print(colorama.Fore.GREEN + "All above mean indexes are aligned with some term" + colorama.Style.RESET_ALL)
                            result_object.save_all_weights_aligned_with_term(os.path.join(run_name, PRONE_BY_MEAN_DIR))
                        else:
                            print(colorama.Fore.RED + "There is above mean index that is no aligned with some term" + colorama.Style.RESET_ALL)

main()