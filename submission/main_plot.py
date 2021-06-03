from consts import *
from data import *
from utilities import *
from statistical_query import *
from standard_network import *
from convex_network import *


def main():
    print("Start plot cluster graph for: {0}".format(DNF))        

    dnf = ReadOnceDNF(DNF)

    X = get_random_init_uniform_samples(TRAIN_SIZE, D)
    Y = np.array([dnf.get_label(x) for x in X], dtype=TYPE)
    train_set = (X, Y)
    X_test = get_random_init_uniform_samples(TEST_SIZE, D)
    Y_test = np.array([dnf.get_label(x) for x in X_test], dtype=TYPE)        
    test_set = (X_test, Y_test)
    network = ConvexNetwork(LR, R)
    network.run(train_set, test_set) 
    save_cluster_graph(network)

main() 
