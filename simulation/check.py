import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))
from consts import *
from data import *
from network import *
from result import *
def check_result_again(name, prune_factor_weight, prune_factor_total_norm, reconstraction_factor_weight, reconstraction_factor_norm):
    all_combinations = get_all_combinations()
    X = np.array(all_combinations, dtype=TYPE)
    base_path =os.path.join("backup","results_290620", name)
    for dnf_size_dir in os.listdir(base_path):
        dnf_size_dir_full = os.path.join(base_path, dnf_size_dir)
        if os.path.isdir(dnf_size_dir_full):
            for epsilon_size_dir in os.listdir(dnf_size_dir_full):   
                epsilon_size_dir_full = os.path.join(dnf_size_dir_full, epsilon_size_dir)
                for dnf_dir in os.listdir(epsilon_size_dir_full):
                    dnf_dir_full = os.path.join(epsilon_size_dir_full, dnf_dir)
                    if os.path.isdir(dnf_dir_full):
                        readonce = ReadOnceDNF([int(i) for i in dnf_dir.split('_')])
                        Y = np.array([readonce.get_label(x) for x in X], dtype=TYPE)
                        noise_size = D - np.sum(readonce.DNF)
                        result_dir = os.path.join(dnf_dir_full, "final")
                        with open(os.path.join(result_dir, "Network.pkl"), 'rb') as f:
                            W, B, _, _ = pickle.load(f)
                        network = Network(W, B)
                        above_mean_indexes = find_indexes_above_half_of_max(network, prune_factor_weight, prune_factor_total_norm)
                        W_prone = network.W[above_mean_indexes]
                        B_prone = network.B[above_mean_indexes]
                        prone_network = Network(W_prone, B_prone)
                        prone_network.prepere_update_network(X, Y)
                        if check_reconstraction(prone_network, readonce, noise_size, reconstraction_factor_weight, reconstraction_factor_norm):
                            print("Success - {}".format(result_dir))
                        else:
                            print("Fail - {}".format(result_dir))

print(sys.argv)
check_result_again(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),  int(sys.argv[5]))