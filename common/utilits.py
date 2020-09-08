from defs import *
from consts import *

def shift_label(Y):
	res = []
	for y in Y:
	 	if y == POSITIVE:
	 		res.append(1)
	 	else:
	 		res.append(0)
	return res 

def is_algined(network, X, i, term):
	for x in X:
		if np.dot(x, term) == np.sum(term):
			if network.get_neuron_value(x, i) <= 0:
				return False
	return True

def get_all_algined_indexes(network, readonce, X, noize_size):
	all_algined_indexes = []
	for term in readonce.DNF:
		term = np.pad(term, [0, noize_size], 'constant', constant_values=(0))
		for i in range(network.r):
			if is_algined(network, X, i, term):
				all_algined_indexes.append(i)
	return all_algined_indexes

def cluster_network(network):
	B_T = np.array([network.B]).T
	weights = np.concatenate([network.W, B_T], 1)
	Y = sch.linkage(weights, method='centroid')
	Z = sch.dendrogram(Y, orientation='right')
	leaves_index = Z['leaves']
	return leaves_index

def is_perfect_classification(network, X, Y):
	return network.check_predriction_for_dataset(X, Y) == X.shape[0]

def find_max_norm_inf(network):
	return np.max([calc_norm_inf(network, i) for i in range(network.r)])

def calc_norm_inf(network, i):
	return np.max(network.W[i])

def find_indexes_above_mean(network):
	mean = find_mean_weight(network)
	indexes = [i for i in range(network.r) if calc_norm_inf(network, i) > mean]
	return indexes

def find_indexes_above_half_of_max(network):
	max_norm_inf = find_max_norm_inf(network)
	indexes = [i for i in range(network.r) if PRUNE_FACTOR_WEIGHT * calc_norm_inf(network, i) > PRUNE_FACTOR_TOTAL_NORM * max_norm_inf]
	return indexes

def reconstraction(network, i):
	reconstraction_nueron = np.zeros([D], dtype=TYPE)
	info_norm = calc_norm_inf(network, i)
	for j in range(D):
		if RECONSTRACTION_FACTOR_WEIGHT * network.W[i][j] >= RECONSTRACTION_FACTOR_NORM * info_norm:
			reconstraction_nueron[j] = 1
	return reconstraction_nueron

def check_reconstraction(network, readonce, noize_size):
	for i in range(network.r):
		reconstraction_nueron = reconstraction(network, i)
		flag = False
		for term in readonce.DNF:
			term = np.pad(term, [0, noize_size], 'constant', constant_values=(0))
			if np.dot(term, reconstraction_nueron) == np.sum(term):
				flag = True
		if not flag:
			return False
	return True