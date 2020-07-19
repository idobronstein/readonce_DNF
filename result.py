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
    GLOBAL_MINIMUM_SIGN = "achive_global_minimum"
    SUBNETWORK_GLOBAL_MINIMUM_SIGN = "subnetwork_global_minimum"
    CLUSTER_GRAPH_W_NAME = "ClusterGraph_W.png"
    CLUSTER_GRAPH_B_NAME = "ClusterGraph_B.png"
    POSITIVE_SAMPLE_TO_NEURON = 'PositiveToSampleNeuron.csv'
    UOT_TO_NUMBER_OF_NEURONS_GRAPH_NAME = 'number_of_neurons.png'
    UOT_TO_MEAN_NORM_OF_NEURONS_GRAPH_NAME = 'mean_norm_of_neurons.png'
    INDEX_TO_UOF_MAP = 'index_to_UOF.txt'

    def __init__(self, create_dir=False, new_D=None):
        assert os.path.isdir(GENERAL_RESULT_PATH), "The result path: {0} doesn't exsits".format(GENERAL_RESULT_PATH)
        result_D = new_D if new_D is not None else D
        dir_name = "D={0}".format(result_D)    
        self.result_dir = os.path.join(GENERAL_RESULT_PATH, dir_name)
        if create_dir:
            if os.path.exists(self.result_dir):
                shutil.rmtree(self.result_dir)
            os.mkdir(self.result_dir)
            param_text_file_path = os.path.join(self.result_dir, "param_file.txt")
            with open(param_text_file_path, "w") as f:
                f.write(self.PARAM_TEXT.format(time=datetime.now(), D=result_D, lr=LR, sigma=SIGMA))
        self.leaves_index = None

    def save_run(self, name, Network, ReadOnceDNF, global_minimum):
        result_path = os.path.join(self.result_dir, name)
        os.mkdir(result_path)
        with open(os.path.join(result_path, self.NETWROK_NAME), "wb") as f:
            pickle.dump(Network, f)
        with open(os.path.join(result_path, self.READ_ONCE_DNF_NAME), "wb") as f:
            pickle.dump(ReadOnceDNF, f)
        if global_minimum:
            with open(os.path.join(result_path, self.GLOBAL_MINIMUM_SIGN), "wb") as f:
                pass 

    def save_subnetwork_succeed(self, name, subnetwork_global_minimum):
        result_path = os.path.join(self.result_dir, name)
        if subnetwork_global_minimum:
            with open(os.path.join(result_path, self.SUBNETWORK_GLOBAL_MINIMUM_SIGN), "wb") as f:
                pass 

    def load_run(self, name):
        result_path = os.path.join(self.result_dir, name)
        with open(os.path.join(result_path, self.NETWROK_NAME), "rb") as f:
            Network = pickle.load(f)
        with open(os.path.join(result_path, self.READ_ONCE_DNF_NAME), "rb") as f:
            ReadOnceDNF = pickle.load(f) 
        return Network, ReadOnceDNF


    def generate_cluster_graph(self, name, Network):
        result_path = os.path.join(self.result_dir, name)
        B_T = np.array([Network.B]).T
        weights = np.concatenate([Network.W, B_T], 1)
        
        # Plot W graph
        fig = pylab.figure()
        axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
        Y = sch.linkage(weights, method='centroid')
        Z = sch.dendrogram(Y, orientation='right')
        axdendro.set_xticks([])
        axdendro.set_yticks([])
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        leaves_index = Z['leaves']
        weights_by_leaves = Network.W[leaves_index,:]
        im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.savefig(os.path.join(result_path, self.CLUSTER_GRAPH_W_NAME), bbox_inches="tight")
        plt.clf()

        # Plot B graph
        fig = pylab.figure()
        axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
        Y = sch.linkage(weights, method='centroid')
        Z = sch.dendrogram(Y, orientation='right')
        axdendro.set_xticks([])
        axdendro.set_yticks([])
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        leaves_index = Z['leaves']
        weights_by_leaves = B_T[leaves_index,:]
        im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.savefig(os.path.join(result_path, self.CLUSTER_GRAPH_B_NAME), bbox_inches="tight")
        plt.clf()

        # Save for later use
        self.leaves_index = leaves_index

    def generate_positive_samples_to_values(self, name, Network, X, Y):
        result_path = os.path.join(self.result_dir, name)
        all_positive_x = [X[i] for i in range(X.shape[0]) if Y[i] == 1]
        all_positive_x_num = len(all_positive_x)
        result = []
        if self.leaves_index is None:
            index_list = range(Network.r)
        else:
            index_list = self.leaves_index[::-1]
        for i in index_list:
            count = 0
            for x in all_positive_x:
                if Network.get_neuron_value(x, i) > 0:
                    count += 1
            result.append((i, (count / all_positive_x_num) * 100))
        with open(os.path.join(result_path, self.POSITIVE_SAMPLE_TO_NEURON), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Nueron Index", "Precent Of Positive samples"])
            for r in result:
                writer.writerow([r[0], "{0} %".format(r[1])])

    def show_3D_graph(self, X, Y, Z, Network, gradient_path):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(Network.all_a, Network.all_b, gradient_path, 'ro', alpha=0.5)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.clf()

    def bar_graph_UOT(self, name, Network, ReadOnceDNF, X):
        result_path = os.path.join(self.result_dir, name)

        # Calculate weights to union of terms to this step
        number_of_steps = len(Network.all_W)
        weights_to_uof = split_weights_to_UOT_2(Network, ReadOnceDNF, X, -1, len(ReadOnceDNF.DNF), False)

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