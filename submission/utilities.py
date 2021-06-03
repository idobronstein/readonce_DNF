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

def cluster_network(network):
	B_T = np.array([network.B]).T
	weights = np.concatenate([network.W, B_T], 1)
	Y = sch.linkage(weights, method='centroid')
	Z = sch.dendrogram(Y, orientation='right')
	leaves_index = Z['leaves']
	return leaves_index

def find_max_norm_inf(W):
	return np.max([calc_norm_inf(W, i) for i in range(W.shape[0])])

def calc_norm_inf(W, i):
	return np.max(W[i])

def find_indexes_above_half_of_max(W, prune_factor):
	max_norm_inf = find_max_norm_inf(W)
	indexes = [i for i in range(W.shape[0]) if  calc_norm_inf(W, i) > prune_factor * max_norm_inf]
	return indexes

def reconstruction(W, i, reconstruction_factor):
	reconstruction_nueron = np.zeros([D], dtype=TYPE)
	info_norm = calc_norm_inf(W, i)
	for j in range(D):
		if  np.abs(W[i][j]) >= reconstruction_factor * info_norm:
			if W[i][j] > 0:
				reconstruction_nueron[j] = 1
			else:
				reconstruction_nueron[j] = -1
	return reconstruction_nueron

def check_reconstruction_per_nueron(W, readonce, noize_size):
	terms_flag = [False] * len(readonce.DNF)
	for i in range(W.shape[0]):
		flag = False
		for reconstruction_factor in RECONSTRUCTION_FACTOR_RANGE:
			reconstruction_nueron = reconstruction(W, i, reconstruction_factor)
			for j, term in enumerate(readonce.DNF):
				term = np.pad(term, [0, noize_size], 'constant', constant_values=(0))
				if np.array_equal(term, reconstruction_nueron):
					flag = True
					terms_flag[j] = True
			if flag:
				break
		if not flag:
			return False
	if False in terms_flag:
		return False
	return True


def save_comparition_graph(result_vec, all_algorithems):
    plt.rcParams.update({'font.size': 20, 'figure.subplot.left': 0.25, 'figure.subplot.right': 0.95, 'figure.subplot.bottom': 0.20, 'figure.subplot.top': 0.97})
    plt.rcParams.update({'axes.labelsize':'large', 'xtick.labelsize':'large', 'ytick.labelsize':'large','legend.fontsize': 'medium'})
    result_vec = 100 * result_vec
    mean = np.mean(result_vec, axis=0)
    std = np.std(result_vec, axis=0)
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    for i in range(len(all_algorithems)):
        label = all_algorithems[i][1]
        color = all_algorithems[i][2]
        marker = all_algorithems[i][3]
        plt.plot(TRAIN_SIZE_LIST, mean[i], color=color, label=label, marker = marker, linewidth=4.0, markersize=10.0)
        plt.fill_between(TRAIN_SIZE_LIST , mean[i] + std[i], mean[i] - std[i], alpha=.2, label='_')
    ax.legend(loc=0, prop={'size': 17})
    ax.set_xlabel('Train Set Size')
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.set_ylabel('Classification Accuracy')
    fig.savefig(RESULT_PATH)
    plt.close(fig)

def save_reconstruction_graph(result_vec):
    plt.rcParams.update({'font.size': 20, 'figure.subplot.left': 0.29, 'figure.subplot.right': 0.98, 'figure.subplot.bottom': 0.20, 'figure.subplot.top': 0.97})
    plt.rcParams.update({'axes.labelsize':'large', 'xtick.labelsize':'large', 'ytick.labelsize':'large','legend.fontsize': 'medium'})
    result_vec = 100 * result_vec
    mean = np.mean(result_vec, axis=0)
    n = result_vec.shape[0]
    std = np.array([1.96 * np.sqrt(p * (100 - p) / n) for p in mean], dtype=TYPE)
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    plt.plot(TRAIN_SIZE_LIST, mean, color="blue", marker = "o", linewidth=4.0, markersize=10.0)
    plt.fill_between(TRAIN_SIZE_LIST , mean + std, mean- std, alpha=.2, label='_')
    ax.set_xlabel('Train Set Size')
    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.set_ylabel('DNF Recovery Accuracy')
    fig.savefig(RESULT_PATH)
    plt.close(fig)

def save_cluster_graph(network):
    leaves_index = cluster_network(network)
    fig = pylab.figure()
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    weights_by_leaves = network.W[leaves_index,:]
    im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig(RESULT_PATH)
    plt.close(fig)


