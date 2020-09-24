import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))
from consts import *
from mnist_dataset import *
from result import *
from fix_layer_2_netowrk import *

def main():
	train_set, validation_set, test_set = get_binary_mnist_db(POSITIVE_NUMBERS, NEGATIVE_NUMBERS)
	network = FixLayerTwoNetwork(False, LR, R, use_batch=True)
	network.run(train_set, test_set)
	best_threshold = (0, 0)
	best_threshold_value = 0
	for prune_factor in PRUNE_FACTOR_RANGE:
		print("Prune with factor: {0}".format(prune_factor))
		above_mean_indexes = find_indexes_above_half_of_max(network, 1, prune_factor)
		if len(above_mean_indexes) > 0:
			W_prone = network.W[above_mean_indexes]
			B_prone = network.B[above_mean_indexes]
			prone_network = FixLayerTwoNetwork(False, LR, W_init=W_prone, B_init=B_prone)
			for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
				print("Reconstaction with factor: {0}".format(reconstraction_factor))
				W_reconstract = np.zeros(W_prone.shape, dtype=TYPE)
				for i in range(W_prone.shape[0]):
					W_reconstract[i] = reconstraction(prone_network, i, 1, reconstraction_factor)
				test_acc = validate_dataset_with_all_terms(validation_set, W_reconstract)
				print("Got {0} ".format(test_acc))
				if test_acc > best_threshold_value:
					best_threshold = (prune_factor, reconstraction_factor)
					best_threshold_value = test_acc
	import IPython; IPython.embed()
main()