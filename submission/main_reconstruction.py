from consts import *
from data import *
from utilities import *
from statistical_query import *
from standard_network import *
from convex_network import *

def main():
    print("Start reconstruction run for: {0}".format(DNF))        
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    X_test = get_random_init_uniform_samples(TEST_SIZE, D)
    Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
    test_set = (X_test, Y_test)

    result_vec = np.zeros([NUM_OF_RUNNING, len(TRAIN_SIZE_LIST)])

    for k in range(NUM_OF_RUNNING):
        for i in range(len(TRAIN_SIZE_LIST)):
            set_size = TRAIN_SIZE_LIST[i]
            print('Running with train set size: {0}'.format(set_size))

            X_train = get_random_init_uniform_samples(set_size, D)
            Y_train = np.array([readonce.get_label(x) for x in X_train], dtype=TYPE)
            train_set = (X_train, Y_train)
                
            network = ConvexNetwork(LR, R)
            network.run(train_set, test_set) 

            flag = False
            for prune_factor in PRUNE_FACTOR_RANGE:
                above_mean_indexes = find_indexes_above_half_of_max(network.W, prune_factor)
                if len(above_mean_indexes) > 0:
                    W_prone = network.W[above_mean_indexes]
                    B_prone = network.B[above_mean_indexes]
                    if check_reconstruction_per_nueron(W_prone, readonce, noise_size):
                        print("Reconstruction succeed")
                        result_vec[k][i] = 1
                        flag = True
                if flag:
                    break
    
    save_reconstruction_graph(result_vec)

main() 
