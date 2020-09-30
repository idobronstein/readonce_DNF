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

def is_all_algined(network, readonce, X, noize_size, indexes):
	padded_terms = [np.pad(term, [0, noize_size], 'constant', constant_values=(0)) for term in readonce.DNF]
	for i in indexes:
		for term in padded_terms:
			if not is_algined(network, X, i, term):
				return False
	return True

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    val = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return val

def find_angle_to_closest_term(network, readonce, X, noize_size, i):
	padded_terms = [np.pad(term, [0, noize_size], 'constant', constant_values=(0)) for term in readonce.DNF]
	max_angle = 0
	for term in padded_terms:
		max_angle = np.min([max_angle, angle_between(term, network.W[i])])
	return max_angle

def cluster_network(network):
	B_T = np.array([network.B]).T
	weights = np.concatenate([network.W, B_T], 1)
	Y = sch.linkage(weights, method='centroid')
	Z = sch.dendrogram(Y, orientation='right')
	leaves_index = Z['leaves']
	return leaves_index

def is_perfect_classification(network, X, sess):
	return network.check_predriction_for_dataset(sess) == X.shape[0]

def find_max_norm_inf(network):
	return np.max([calc_norm_inf(network, i) for i in range(network.r)])

def calc_norm_inf(network, i):
	return np.max(network.W[i])

def find_indexes_above_mean(network):
	mean = find_mean_weight(network)
	indexes = [i for i in range(network.r) if calc_norm_inf(network, i) > mean]
	return indexes

def find_indexes_above_half_of_max(network, prune_factor_weight, prune_factor_total_norm):
	max_norm_inf = find_max_norm_inf(network)
	indexes = [i for i in range(network.r) if prune_factor_weight * calc_norm_inf(network, i) > prune_factor_total_norm * max_norm_inf]
	return indexes

def reconstraction(network, i, reconstraction_factor_weight, reconstraction_factor_norm):
	reconstraction_nueron = np.zeros([D], dtype=TYPE)
	info_norm = calc_norm_inf(network, i)
	for j in range(D):
		if reconstraction_factor_weight * np.abs(network.W[i][j]) >= reconstraction_factor_norm * info_norm:
			if network.W[i][j] > 0:
				reconstraction_nueron[j] = 1
			else:
				reconstraction_nueron[j] = 1
	return reconstraction_nueron

def check_reconstraction(network, readonce, noize_size, reconstraction_factor_weight, reconstraction_factor_norm):
	terms_flag = [False] * len(readonce.DNF)
	for i in range(network.r):
		reconstraction_nueron = reconstraction(network, i, reconstraction_factor_weight, reconstraction_factor_norm)
		flag = False
		for j, term in enumerate(readonce.DNF):
			term = np.pad(term, [0, noize_size], 'constant', constant_values=(0))
			if np.array_equal(term, reconstraction_nueron):
				flag = True
				terms_flag[j] = True
		if not flag:
			return False
	if False in terms_flag:
		return False
	return True

def calc_bais_threshold(w, readonce, noize_size, D):
    value = 0
    padded_terms = [np.pad(term, [0, noize_size], 'constant', constant_values=(0)) for term in readonce.DNF]
    for term in padded_terms:
      term_in_weight = np.array([w[l] for l in range(D) if term[l] != 0])
      min_value = np.max([np.min(term_in_weight), 0])
      value += - np.sum(np.abs(term_in_weight)) + 2 * min_value 
    return value


def validate_figure_with_term(figure, term, need_to_reshape=False):
	if need_to_reshape:
		figure_reshape = figure.reshape([-1])
	return np.dot(figure, term)	== np.sum(np.abs(term))

def validate_dataset_with_all_terms(dataset, all_terms, need_to_reshape=False):
	success_num = 0
	for x, y in zip(dataset[0], dataset[1]):
		flag = False
		for term in all_terms:
			if validate_figure_with_term(x, term, need_to_reshape):
				if y == POSITIVE:
					flag = True
					success_num += 1
					break
				else:
					flag = True
					break
		if not flag and y == NEGATIVE:
			success_num += 1
	return success_num / float(dataset[0].shape[0])