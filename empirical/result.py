from consts import *
from utilits import *

class Result():

    PARAM_FILE_NAME = "param_file.txt"

    def __init__(self, result_path=TEMP_RESULT_PATH, is_tmp=True, extra_to_name=''):
        assert os.path.isdir(result_path), "The result path: {0} doesn't exsits".format(result_path)
        dir_name = "D={0}_{1}".format(D, extra_to_name) 
        self.result_dir = os.path.join(result_path, dir_name)
        if os.path.exists(self.result_dir):
            if is_tmp:
                self.enforce_delete_dir()
                os.mkdir(self.result_dir)
        else:
            os.mkdir(self.result_dir)
        if not is_tmp:
            now = datetime.now()
            self.result_dir = os.path.join(self.result_dir, now.strftime("%m_%d_%H_%M"))
            os.mkdir(self.result_dir)

        param_text_file_path = os.path.join(self.result_dir, self.PARAM_FILE_NAME)
        with open(param_text_file_path, "w") as f:
            f.write("D             -  {}\n".format(D))
            f.write("lr            -  {}\n".format(LR))
            f.write("sigma         -  {}\n".format(SIGMA))
            f.write("r             -  {}\n".format(R))

    def create_dir(self, name):
        self.result_dir = os.path.join(self.result_dir, name)
        os.mkdir(self.result_dir)

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

    def save_graph(self, all_algorithems, result_vec):
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)
        for i in range(len(all_algorithems)):
            ax.plot(TRAIN_SIZE_LIST, result_vec[i], all_algorithems[i][2], markersize=14)
        ax.set_title('Learning f1')
        ax.set_xlabel('Train set size')
        ax.set_ylabel('Accuracy')
        #ax.set_ylim(0.5, 1.05)
        #legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
        fig.savefig(os.path.join(self.result_dir, "compersion.png"))
        plt.close(fig)

    def save_reconstraction_graph(self, result_vec):
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 5)
        ax.plot(TRAIN_SIZE_LIST, result_vec, '.', markersize=14)
        ax.set_title('Reconstracion f1')
        ax.set_xlabel('Train set size')
        ax.set_ylabel('Success rate')
        fig.savefig(os.path.join(self.result_dir, "reconstraction.png"))
        plt.close(fig)

    def save_result_to_pickle(self, name, result):
        with open(os.path.join(self.result_dir, name), 'wb') as f:
            pickle.dump(result, f)

    def angle_vs_norm_graph(self, network, readonce, X, noize_size):
        fig, ax = plt.subplots()
        ax.set_title('Norm Vs. Angle with closest term')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Norm')
        angel_vec = [find_angle_to_closest_term(network, readonce, X, noize_size, i) for i in range(network.r)]
        norm_vec = [np.linalg.norm(network.W[i]) for i in range(network.r)]
        ax.plot(angel_vec, norm_vec, '.')
        fig.savefig(os.path.join(self.result_dir, "angle_vs_norm.png"))
        plt.close(fig)

    def bias_graph(self, network, readonce, X, noize_size):
        steps_range = range(len(network.all_W))
        fig, ax = plt.subplots()
        ax.set_title('Mean Bais Threshold \\ Network Bais Vs. Steps')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Mean Value')
        bais_threshold_vec = [np.mean([calc_bais_threshold(network.all_W[step].T[i], readonce, noize_size, D) for i in range(network.r)]) for step in steps_range]
        bais_vec = [np.mean(network.all_B[step]) for step in range(len(network.all_W))]
        ax.plot(steps_range, bais_threshold_vec, 'r.', label="Bias Threshold")
        ax.plot(steps_range, bais_vec, 'b.', label="Network Bias")
        legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
        fig.savefig(os.path.join(self.result_dir, "bias_threshold.png"))
        plt.close(fig)

    def cluster_graph(self, network, add_to_name=''):
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
        fig.savefig(os.path.join(self.result_dir, add_to_name + "cluster_w.png"), bbox_inches="tight")
        plt.close(fig)

        # Plot B graph
        fig = pylab.figure()
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        weights_by_leaves = np.array([network.B]).T[leaves_index,:]
        im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.savefig(os.path.join(self.result_dir, add_to_name+ "cluster_b.png"), bbox_inches="tight")
        plt.close(fig)
