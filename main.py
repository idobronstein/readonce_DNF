from consts import *
from data import *
from network import *
from result import * 


def run_network(network, X, Y):
    step = 0
    global_minimum_point, local_minimum_point = False, False
    while not global_minimum_point and not local_minimum_point and step < MAX_STEPS:
        display = False
        if step % PRINT_STEP_JUMP == 0 and step > 0:
            print("Step number: {0}, ".format(step), end ="")
            display = True
        global_minimum_point, local_minimum_point = network.update_network(X, Y, LR, display)
        step += 1
    if local_minimum_point and not global_minimum_point:
        print("Got to local minimums")
    return global_minimum_point

def save_results(minimum_point, name, result_object, network, read_once_DNF, X):
    print("Saving the results in: {0}".format(name))
    result_object.save_run(name, network, read_once_DNF, minimum_point)
    result_object.generate_cluster_graph(name, network)
    #result_object.generate_positive_samples_to_values(run_name, network, X, Y)
    if minimum_point:
        result_object.bar_graph_UOT(name, network, read_once_DNF, X)

def main():
    print("Making result object in the path: {0}".format(GENERAL_RESULT_PATH))
    result_object = Result(True)
    print("Generate all partitions")
    all_partitions = get_all_partitions()
    all_partitions.remove([D])
    print("Generate all combinations")
    all_combinations = get_all_combinations()
    r = len(all_combinations)

    W_init = np.array(3 * SIGMA * np.random.randn(5 * r, D), dtype=np.float32)   
    #W_init = np.array(all_combinations, dtype=FLOAT_TYPE) * SIGMA
    B_init = np.zeros([5 * r], dtype=FLOAT_TYPE)

    X = np.array(all_combinations, dtype=FLOAT_TYPE)

    all_partitions = [[2, 6]]
    for partition in all_partitions:
        #if 1 in partition:
        if partition != [2, 6] and partition != [3, 3, 3, 3] and partition != [4, 4, 4] and partition != [6, 6]:
            print("Skipping: {0}".format(partition))
            continue
        print("Start a run for: {0}".format(partition))
        
        read_once_DNF = ReadOnceDNF(partition)
        Y = np.array([read_once_DNF.get_label(x) for x in X], dtype=FLOAT_TYPE)
        
        #X, Y = upsampling(X, Y, MOUNT_OF_UPSAMPELING, POSITIVE)
        #X, Y = downsampling(X, Y, PROB_OF_DOWNSAMPLING, 0)
        
        network = Network(W_init, B_init)
        minimum_point = run_network(network, X, Y)

        run_name = '_'.join([str(i) for i in partition])
        save_results(minimum_point, run_name, result_object, network, read_once_DNF, X)

        if minimum_point:
            print("Find all the weights which align with specific term")
            weights_to_terms = split_weights_to_UOT_2(network, read_once_DNF, X, -1, len(read_once_DNF.DNF), True)
            lottery_ticket = []
            for weights_to_term in weights_to_terms[:-1]:
                lottery_ticket += weights_to_term[1]
            lottery_ticket.sort()
            if lottery_ticket == list(range(2 ** D)):
                print("Every nueron belog to one of the union of terms")
        else:
            print("Doesn't get to global minimum and run out pf steps")

main()