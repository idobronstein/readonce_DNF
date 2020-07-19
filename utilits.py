from consts import *



def is_algined(network, X, i, term):
	for x in X:
		if np.dot(x, term) == np.sum(term):
			if network.get_neuron_value(x, i) <= 0:
				return False
	return True

def get_all_algined_indexes(network, readonce, X):
	all_algined_indexes = []
	for term in readonce.DNF:
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

##################################### OLD #####################################

def find_weights_of_specific_UOT(network, X, uot, step, weights_indexes):
	weights_indexes_copy = weights_indexes[:]
	for i in weights_indexes:
		for x in X:
			if np.dot(x, uot) == np.sum(uot):
				if network.get_past_neuron_value(x, i, step) <= 0:
					weights_indexes_copy.remove(i)
					break
	return weights_indexes_copy

def split_weights_to_UOT(network, ReadOnceDNF, X, step, max_union_size ,return_indexes):
	weights_to_uot = []
	r = network.W.shape[0]
	all_weights_indexes = list(range(r))
	for i in range(max_union_size + 1):
		for uot_indexes in combinations(range(len(ReadOnceDNF.DNF)), i):
			if len(uot_indexes) > 0:
				uot = np.sum([ReadOnceDNF.DNF[j] for j in uot_indexes], 0)
				weights_index = find_weights_of_specific_UOT(network, X, uot, step, all_weights_indexes[:])
				weights_to_uot.append((uot, weights_index))
				for j in weights_index:
					all_weights_indexes.remove(j)
	
	weights_leftover = []
	for i in range(r):
		flag = True
		for _, weights in weights_to_uot:
			if i in weights:
				flag = False
		if flag:
			weights_leftover.append(i)
	weights_to_uot.append((None, weights_leftover))
	if not return_indexes:
		weights_to_uot = [(a, network.all_W[step][b]) for a, b in weights_to_uot]
	return weights_to_uot

def find_weights_of_specific_UOT_2(network, X, uot, uot_size, step):
	weights_indexes = list(range(network.W.shape[0]))
	for i in range(network.W.shape[0]):
		for x in X:
			is_pass = network.get_past_neuron_value(x, i, step) > 0
			uot_pad = np.pad(uot, [0,x.shape[0] - uot.shape[0]], 'constant', constant_values=(0))
			is_align_sample = np.dot(x, uot_pad) >= np.sum(uot) - 2*(uot_size - 1)
			if not ((is_align_sample and is_pass) or (not is_align_sample and not is_pass)):
					weights_indexes.remove(i)
					break
	return weights_indexes

def split_weights_to_UOT_2(network, ReadOnceDNF, X, step, max_union_size ,return_indexes):
	weights_to_uot = []
	for i in range(max_union_size + 1):
		for uot_indexes in combinations(range(len(ReadOnceDNF.DNF)), i):
			if len(uot_indexes) > 0:
				uot = np.sum([ReadOnceDNF.DNF[j] for j in uot_indexes], 0)		
				uot_size = len(uot_indexes)
				weights_index = find_weights_of_specific_UOT_2(network, X, uot, uot_size, step)
				weights_to_uot.append((uot, weights_index))
	
	weights_leftover = []
	for i in range(network.W.shape[0]):
		flag = True
		for _, weights in weights_to_uot:
			if i in weights:
				flag = False
		if flag:
			weights_leftover.append(i)
	weights_to_uot.append((None, weights_leftover))
	if not return_indexes:
		weights_to_uot = [(a, network.all_W[step][b]) for a, b in weights_to_uot]
	return weights_to_uot					
