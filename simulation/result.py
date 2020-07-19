from consts import *
from utilits import *

class Result():

    PARAM_TEXT ="""
Time          -  {time}
D             -  {D}
learinig rate -  {lr}
sigma         -  {sigma}
"""
    NETWROK_NAME = "Network.pkl"
    READ_ONCE_DNF_NAME = "ReadOnceDNF.pkl"
    RESULT_SUMMERY_NAME = 'result_summery.txt'
    CLUSTER_GRAPH_W_NAME = "ClusterGraph_W.png"
    CLUSTER_GRAPH_B_NAME = "ClusterGraph_B.png"
    SUMMARIZE_ALINED_TERMS_NAME = 'summarize_alined_terms.txt'

    def __init__(self, result_path=TEMP_RESULT_PATH, is_tmp=True, new_D=None):
        assert os.path.isdir(result_path), "The result path: {0} doesn't exsits".format(result_path)
        dir_name = "D={0}".format(D)    
        self.result_dir = os.path.join(result_path, dir_name)
        if os.path.exists(self.result_dir):
            if is_tmp:
                self.enforce_delete_dir()
            else:
                assert False, "There is already permanet directory here: {0}".format(self.result_dir)
        os.mkdir(self.result_dir)
        param_text_file_path = os.path.join(self.result_dir, "param_file.txt")
        with open(param_text_file_path, "w") as f:
            f.write(self.PARAM_TEXT.format(time=datetime.now(), D=D, lr=LR, sigma=SIGMA))

    def enforce_delete_dir(self):
        try_count = 0
        while try_count < MAX_TRY_TO_DELETE_DIR:
            try_count += 1
            if not os.path.exists(self.result_dir):
                return
            try:
                shutil.rmtree(self.result_dir)
            except:
                pass
        assert True, "Can't delete the directory: {0}".format(self.result_dir)

    def create_dir(self, name):
        result_path = os.path.join(self.result_dir, name)
        os.mkdir(result_path)

    def save_run(self, name, network, readonce, global_minimum, perfect_classifocation):
        result_path = os.path.join(self.result_dir, name)
        os.mkdir(result_path)
        with open(os.path.join(result_path, self.NETWROK_NAME), "wb") as f:
            pickle.dump(network, f)
        with open(os.path.join(result_path, self.READ_ONCE_DNF_NAME), "wb") as f:
            pickle.dump(readonce, f)
        with open(os.path.join(result_path, self.RESULT_SUMMERY_NAME), "w") as f:
            f.write("{0}    -   Got perfect classification\n".format(perfect_classifocation))
            f.write("{0}    -   Got to global minimum\n".format(global_minimum))

    def load_run(self, name):
        result_path = os.path.join(self.result_dir, name)
        with open(os.path.join(result_path, self.NETWROK_NAME), "rb") as f:
            network = pickle.load(f)
        with open(os.path.join(result_path, self.READ_ONCE_DNF_NAME), "rb") as f:
            readonce = pickle.load(f) 
        return network, readonce

    def generate_cluster_graph(self, name, network):
        result_path = os.path.join(self.result_dir, name)
        leaves_index = cluster_network(network)

        # Plot W graph
        fig = pylab.figure()
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        weights_by_leaves = network.W[leaves_index,:]
        im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.savefig(os.path.join(result_path, self.CLUSTER_GRAPH_W_NAME), bbox_inches="tight")
        plt.clf()

        # Plot B graph
        fig = pylab.figure()
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        weights_by_leaves = np.array([network.B]).T[leaves_index,:]
        im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.savefig(os.path.join(result_path, self.CLUSTER_GRAPH_B_NAME), bbox_inches="tight")
        plt.clf()

    def summarize_alined_terms(self, name, network, readonce, X):
        result_path = os.path.join(self.result_dir, name)

        all_algined_indexes = get_all_algined_indexes(network, readonce, X)
        all_non_algined_indexes = [i for i in range(network.r) if i not in all_algined_indexes]

        get_mean_norm = lambda w, indexes: np.mean(np.mean(np.sum(np.abs(w[indexes]),1)))
        mean_algined_weights_norm = get_mean_norm(network.W, all_algined_indexes)
        mean_non_algined_weights_norm = get_mean_norm(network.W, all_non_algined_indexes)

        file_name = os.path.join(result_path, self.SUMMARIZE_ALINED_TERMS_NAME)
        with open(file_name, 'w') as f:
            f.write("{algined}/{total}    -   alined terms num\n".format(algined=len(all_algined_indexes), total=network.r))
            f.write("{nonalgined}/{total}    -   non alined terms num\n".format(nonalgined=len(all_non_algined_indexes), total=network.r))
            f.write("{mean_algined}    -   alined terms mean norm\n".format(mean_algined=mean_algined_weights_norm))
            f.write("{mean_non_algined}    -   non alined terms mean norm\n".format(mean_non_algined=mean_non_algined_weights_norm))


##################################### OLD #####################################
    UOT_TO_NUMBER_OF_NEURONS_GRAPH_NAME = 'number_of_neurons.png'
    UOT_TO_MEAN_NORM_OF_NEURONS_GRAPH_NAME = 'mean_norm_of_neurons.png'
    INDEX_TO_UOF_MAP = 'index_to_UOF.txt'

    def show_3D_graph(self, X, Y, Z, network, gradient_path):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(network.all_a, network.all_b, gradient_path, 'ro', alpha=0.5)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.clf()

    def bar_graph_terms(self, name, network, readonce, X):
        result_path = os.path.join(self.result_dir, name)

        # Calculate weights to union of terms to this step
        number_of_steps = len(network.all_W)
        weights_to_uof = split_weights_to_UOT_2(network, readonce, X, -1, len(readonce.DNF), False)

        # Save graph bar that show terms to number of neurons
        plt.bar(range(len(weights_to_uof)), [len(t[1]) for t in weights_to_uof], align='center', alpha=0.5)
        plt.xlabel('Index of term')
        plt.ylabel('Number of neurons')
        plt.title('terms to number of neurons')
        bar_graph_name = os.path.join(result_path, self.UOT_TO_NUMBER_OF_NEURONS_GRAPH_NAME)
        plt.savefig(bar_graph_name)
        plt.clf()

        # Save graph bar that show terms to mean norm of wights
        norm_weights_to_uof = []
        for t in weights_to_uof:
            norm_weights_to_uof.append([])
            if len(t[1]) == 0:
                norm_weights_to_uof[-1].append(0)
            for w in t[1]:
                norm_weights_to_uof[-1].append(np.sum(np.abs(w)))
        plt.bar(range(len(weights_to_uof)), [np.mean(n) for n in norm_weights_to_uof], align='center', alpha=0.5)
        plt.xlabel('Index of term')
        plt.ylabel('Mean norm')
        plt.title('terms to mean norm of neurons')
        bar_graph_name = os.path.join(result_path, self.UOT_TO_MEAN_NORM_OF_NEURONS_GRAPH_NAME)
        plt.savefig(bar_graph_name)
        plt.clf()

        # Create map from index to unuion of term:
        file_name = os.path.join(result_path, self.INDEX_TO_UOF_MAP)
        with open(file_name, 'w') as f:
            for i in range(len(weights_to_uof)):
                if weights_to_uof[i][0] is None:
                    f.write("{0}    -   leaftover\n".format(i))
                else:
                    f.write("{0}    -   {1}\n".format(i, weights_to_uof[i][0]))