from consts import *
from utilits import *

class Result():

    PARAM_FILE_NAME = "param_file.txt"

    def __init__(self, result_path=TEMP_RESULT_PATH, is_tmp=True, extra_to_name='', const_dir=False):
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
            if not const_dir: 
                now = datetime.now()
                self.result_dir = os.path.join(self.result_dir, now.strftime("%m_%d_%H_%M"))
                os.mkdir(self.result_dir)

        param_text_file_path = os.path.join(self.result_dir, self.PARAM_FILE_NAME)
        with open(param_text_file_path, "w") as f:
            f.write("D             -  {}\n".format(D))
            f.write("lr            -  {}\n".format(LR))
            f.write("sigma         -  {}\n".format(SIGMA))
            f.write("r             -  {}\n".format(R))

        self.const_dir = const_dir


    def save_const_file(self):
        shutil.copyfile(CONST_FILE_NAME, os.path.join(self.result_dir, CONST_FILE_NAME))


    def create_dir(self, name):
        self.result_dir = os.path.join(self.result_dir, name)
        if not os.path.exists(self.result_dir):
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

    def cluster_graph(self, network, add_to_name='', leaves_index=None):
        if leaves_index is None:
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

        ## Plot B graph
        #fig = pylab.figure()
        #axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
        #weights_by_leaves = np.array([network.B]).T[leaves_index,:]
        #im = axmatrix.matshow(weights_by_leaves, aspect='auto', origin='lower')
        #axmatrix.set_xticks([])
        #axmatrix.set_yticks([])
        #axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
        #pylab.colorbar(im, cax=axcolor)
        #fig.savefig(os.path.join(self.result_dir, add_to_name+ "cluster_b.png"), bbox_inches="tight")
        #plt.close(fig)


    def comp_save_graph(self, result_vec, all_algorithems, train_size_list, extra_to_name=''):
        plt.rcParams.update({'font.size': 26, 'figure.subplot.left': 0.25, 'figure.subplot.right': 0.95, 'figure.subplot.bottom': 0.20, 'figure.subplot.top': 0.97})
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
            plt.plot(train_size_list, mean[i], color=color, label=label, marker = marker, linewidth=4.0, markersize=10.0)
            plt.fill_between(train_size_list , mean[i] + std[i], mean[i] - std[i], alpha=.2, label='_')
        ax.legend(loc=0, prop={'size': 15})
        ax.set_xlabel('Train Set Size')
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(65, 101)
        fig.savefig(os.path.join(self.result_dir, "comparsion{0}.png".format(extra_to_name)))
        plt.close(fig)

    def dnfs_save_graph(self, result_vec, extra_to_name=''):
        plt.rcParams.update({'font.size': 24, 'figure.subplot.left': 0.25, 'figure.subplot.right': 0.95, 'figure.subplot.bottom': 0.20, 'figure.subplot.top': 0.97})
        plt.rcParams.update({'axes.labelsize':'large', 'xtick.labelsize':'large', 'ytick.labelsize':'large','legend.fontsize': 'medium'})
        result_vec = 100 * result_vec
        mean = np.mean(result_vec, axis=0).reshape(result_vec.shape[1])
        std = np.std(result_vec, axis=0).reshape(result_vec.shape[1])
        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 4.8)
        plt.plot(range(mean.shape[0]), mean, color='b', marker = "o", linewidth=4.0, markersize=10.0)
        plt.fill_between(range(mean.shape[0]) , mean + std, mean - std, alpha=.2, label='_')
        ax.set_xlabel('Number of overlap terms')
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(65, 101)
        fig.savefig(os.path.join(self.result_dir, "overlap{0}.png".format(extra_to_name)))
        plt.close(fig)

    def angle_save_graph(self, result_vec, train_size_list):
        plt.rcParams.update({'font.size': 24, 'figure.subplot.left': 0.25, 'figure.subplot.right': 0.95, 'figure.subplot.bottom': 0.20, 'figure.subplot.top': 0.97})
        plt.rcParams.update({'axes.labelsize':'large', 'xtick.labelsize':'large', 'ytick.labelsize':'large','legend.fontsize': 'medium'})
        mean = result_vec[0]
        std = result_vec[1]
        fig, ax = plt.subplots()
        fig.set_size_inches(6.4, 4.8)
        plt.plot(train_size_list, mean, color='b', marker = "o", linewidth=4.0, markersize=10.0)
        plt.fill_between(train_size_list , mean + std, mean - std, alpha=.2, label='_')
        ax.set_xlabel('Train Set Size')
        ax.xaxis.set_label_coords(0.5, -0.15)
        ax.set_ylabel('Average Angle')
        ax.set_ylim(0, 30)
        fig.savefig(os.path.join(self.result_dir, "angle.png"))
        plt.close(fig)

    def save_state(self, result_vec, round_num, train_list_location, extra_to_name=""):
        print("Saving state for: round_num - {0}, train list location - {1}".format(round_num, train_list_location))
        state_path = os.path.join(self.result_dir, STATE_PATH + extra_to_name)
        with open(state_path, 'wb+') as f:
            pickle.dump((result_vec, round_num, train_list_location), f) 

    def load_state(self, all_algorithems, train_size_list, extra_to_name=""):
        state_path = os.path.join(self.result_dir, STATE_PATH + extra_to_name)
        if os.path.isfile(state_path):
            with open(state_path, 'rb') as f:
                all_state = pickle.load(f) 
                result_vec, round_num, train_list_location = all_state
            print("Restore state: round_num - {0}, train list location - {1}".format(round_num, train_list_location))
        else:
            result_vec = np.zeros([NUM_OF_RUNNING, len(all_algorithems), len(train_size_list)])
            round_num = 0
            train_list_location = 0
        return result_vec, round_num, train_list_location