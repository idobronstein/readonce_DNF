import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))

from consts import *
from data import *
from network import *
from result import * 


def run_network(network, X, Y, run_name, result_object, readonce, noise_size, sess):
    step = 0
    global_minimum_point, local_minimum_point = False, False
    while not global_minimum_point and not local_minimum_point and step < MAX_STEPS:
        global_minimum_point, local_minimum_point, non_zero_loss_sample_counter = network.update_network(sess, LR)
        step += 1
        if step % PRINT_STEP_JUMP == 0 and step > 0:
            result_name = os.path.join(run_name, str(step))
            result_object.logger.info("Step number: {0}, Accuracy: {1} / {2}".format(step, X.shape[0] - non_zero_loss_sample_counter, X.shape[0]))
            #save_results(global_minimum_point, result_name, result_object, network, readonce, X, Y, noise_size)
    if local_minimum_point and not global_minimum_point:
        result_object.logger.error("Got to local minimums")
    return global_minimum_point

def save_results(minimum_point, name, result_object, network, readonce, X, sess, noise_size):
    result_object.save_run(name, network, readonce, minimum_point, is_perfect_classification(network, X, sess))
    result_object.generate_cluster_graph(name, network)

def main():
    result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
    result_object = Result(result_path, IS_TEMP)
    result_object.logger.info("Making result object in the path: {0}".format(result_path))
    
    result_object.logger.info("Generate all combinations for D={}".format(D))
    all_combinations = get_all_combinations()
    r = len(all_combinations)
    X = np.array(all_combinations, dtype=TYPE)

    for dnf_size in range(2, D + 1):
        noise_size = D - dnf_size

        result_object.logger.info("Generate all balanced partitions in size {}".format(dnf_size))
        all_partitions = get_all_balanced_partitions(dnf_size)
        # remove the DNF of all 1 and the DNF with one term
        all_partitions = all_partitions[1:-1]

        if len(all_partitions) == 0:
            result_object.logger.info("No relevant phrases. Skippinig..")
        else:
            for epsilon in range(MIN_EPSILON, MAX_EPSILON, STEP_EPSILON):
                result_object.logger.info("Running for all dnf in size: {0} and initialization: {1}".format(dnf_size, epsilon))
                            
                result_object.set_result_path(dnf_size, epsilon)

                for partition in all_partitions:
            
                    result_object.logger.info("Start a run for: {0}".format(partition))
                    
                    run_name = '_'.join([str(i) for i in partition])  
                    backup_name = os.path.join(run_name, BACKUP_DIR)
                    result_object.create_dir(run_name)
                    result_object.create_dir(backup_name)
            
                    readonce = ReadOnceDNF(partition)
                    Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
                    
                    tf.reset_default_graph()
                    with tf.Graph().as_default():
                        W_init = np.array(all_combinations, dtype=TYPE) * epsilon
                        B_init = np.zeros([r], dtype=TYPE)
                        network = Network(W_init, B_init)
                        network.prepere_update_network(X, Y)

                        result_name = os.path.join(run_name, BACKUP_DIR)

                        with tf.Session() as sess:
                            # init params
                            init = tf.initialize_all_variables()
                            sess.run(init)
                            minimum_point = run_network(network, X, Y, backup_name, result_object, readonce, noise_size, sess)
                    
                            save_results(minimum_point, os.path.join(run_name, ORIGINAL_FINAL_DIR), result_object, network, readonce, X, sess, noise_size)

                    tf.reset_default_graph()
                    with tf.Graph().as_default():
                        if minimum_point:
                            result_object.logger.info("We got to minimum point")
                            aligend_indexes = get_all_algined_indexes(network, readonce, X, noise_size)
                            result_object.logger.info("Prone the network by inf norm")
                            above_mean_indexes = find_indexes_above_half_of_max(network)
                            W_prone = network.W[above_mean_indexes]
                            B_prone = network.B[above_mean_indexes]
                            prone_network = Network(W_prone, B_prone)
                            prone_network.prepere_update_network(X, Y)
                            with tf.Session() as sess:
                                init = tf.initialize_all_variables()
                                sess.run(init)
                                save_results(minimum_point, os.path.join(run_name, PRONE_BY_MEAN_DIR), result_object, prone_network, readonce, X, sess, noise_size)
                                if not is_perfect_classification(prone_network, X, sess):
                                    result_object.logger.error("After pruning the network doesn't classify perfectly")
                                elif not set(above_mean_indexes).issubset(aligend_indexes):
                                    result_object.logger.error("After pruning there is terms which doesn't aligned with some term")
                                elif not check_reconstraction(prone_network, readonce, noise_size):
                                    result_object.logger.error("After pruning we can't succeed to reconstract the DNF")
                                else:
                                    result_object.logger.critical("After pruning the network classify perfectly, aligned with the terms and reconstract the DNF")
                        else:
                            result_object.logger.error("Got to local minimums")

main()