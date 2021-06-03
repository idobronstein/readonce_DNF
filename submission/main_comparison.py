from consts import *
from data import *
from utilities import *
from statistical_query import *
from standard_network import *
from convex_network import *

def main():
    print("Start compere preformance for: {0}".format(DNF))        
    readonce = ReadOnceDNF(DNF)
    noise_size = D - sum(DNF)

    all_algorithems = [
        (lambda: ConvexNetwork(LR, R), "Convex, Standard init", 'b', "o"),
        (lambda: ConvexNetwork(LR, R, sigma=SIGMA_LARGE), "Convex, Large init", 'g', "s"),
        (lambda: StandardNetwork(LR, R), "Standard, Standard  init", 'r', "+"),
        (lambda: StatisticalQuery(), "Statistical Query", 'm', "H")
    ]
    
    X_test = get_random_init_uniform_samples(TEST_SIZE, D)
    Y_test = np.array([readonce.get_label(x) for x in X_test], dtype=TYPE)
    test_set = (X_test, Y_test)

    result_vec = np.zeros([NUM_OF_RUNNING, len(all_algorithems), len(TRAIN_SIZE_LIST)])

    for k in range(NUM_OF_RUNNING):
        for i in range(len(TRAIN_SIZE_LIST)):
            set_size = TRAIN_SIZE_LIST[i]
            X_train = get_random_init_uniform_samples(set_size, D)
            Y_train = np.array([readonce.get_label(x) for x in X_train], dtype=TYPE)
            train_set = (X_train, Y_train)
            for j in range(len(all_algorithems)):
                algorithem = all_algorithems[j]
                print('Running algorithem: "{0}" with train set in size: {1}'.format(algorithem[1], set_size))
                network = algorithem[0]()
                _, algorithem_result = network.run(train_set, test_set) 
                result_vec[k][j][i] = algorithem_result 

    save_comparition_graph(result_vec, all_algorithems)

main() 
