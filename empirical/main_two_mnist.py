import sys
import os
sys.path.insert(-1, os.path.join(os.getcwd(), '..', 'common'))
from consts import *
from two_layer_network import *
from tabular_datasets import *
from result import *
from fix_layer_2_netowrk import *

def main():
	result_path = TEMP_RESULT_PATH if IS_TEMP else GENERAL_RESULT_PATH
	print("Making result object in the path: {0}".format(result_path))
	result_object = Result(result_path, IS_TEMP, extra_to_name='mnist')

	#train_set, validation_set, test_set = get_db(POSITIVE_NUMBERS, NEGATIVE_NUMBERS)
	train_set, test_set = get_kr_kp_db()
	network = TwoLayerNetwork(R, LR)
	network.run(train_set, train_set)
	W_fold = np.zeros(network.W.shape)
	B_W_fold = np.zeros(network.B_W.shape)
	U_fold = np.ones(network.U.shape)
	for j in range(network.r):
		W_fold[j] = network.W[j] * np.abs(network.U[j])
		B_W_fold[j] = network.B_W[j] * np.abs(network.U[j])
		U_fold[j] = U_fold[j] * np.sign(network.U[j])
	fold_network = TwoLayerNetwork(R, LR, W_init=W_fold, U_init=U_fold, B_W_init=B_W_fold, B_U_init=network.B_U)
	
	best_threshold = (0, 0)
	best_threshold_value = 0
	for prune_factor in PRUNE_FACTOR_RANGE:
		print("Prune with factor: {0}".format(prune_factor))
		above_mean_indexes = find_indexes_above_half_of_max(fold_network, 1, prune_factor)
		if len(above_mean_indexes) > 0:
			W_prone = fold_network.W[above_mean_indexes]
			B_W_prone = fold_network.B_W[above_mean_indexes]
			U_prone = fold_network.U[above_mean_indexes]
			prone_network = TwoLayerNetwork(R, LR, W_init=W_prone, U_init=U_prone, B_W_init=B_W_prone, B_U_init=network.B_U)
			for reconstraction_factor in RECONSTRACTION_FACTOR_RANGE:
				if np.sum(U_prone) == U_prone.shape[0]:
					print("Reconstaction with factor: {0}".format(reconstraction_factor))
					W_reconstract = np.zeros(W_prone.shape, dtype=TYPE)
					for i in range(W_prone.shape[0]):
						W_reconstract[i] = reconstraction(prone_network, i, 1, reconstraction_factor)
					test_acc = validate_dataset_with_all_terms(test_set, W_reconstract)
					print("Got {0} ".format(test_acc))
					if test_acc >= best_threshold_value:
						best_threshold = (prune_factor, reconstraction_factor)
						best_threshold_value = test_acc
	import IPython; IPython.embed()
main()