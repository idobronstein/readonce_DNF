from consts import *
from utilits import *

class Result():
    COLORS = {
        'DEBUG':    'cyan',
        'INFO':     'cyan',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'green',
    }

    PARAM_TEXT ="""
Time                 -  {time}
D                    -  {D}
learinig rate        -  {lr}
dnf size             -  {dnf_size}
epsilon              -  {epsilon}
second layer bais    -  {second_layer_bais}
"""
    NETWROK_NAME = "Network.pkl"
    READ_ONCE_DNF_NAME = "ReadOnceDNF.pkl"
    RESULT_SUMMERY_NAME = 'result_summery.txt'
    CLUSTER_GRAPH_W_NAME = "ClusterGraph_W.png"
    CLUSTER_GRAPH_B_NAME = "ClusterGraph_B.png"

    def __init__(self, result_path=TEMP_RESULT_PATH, is_tmp=True, load=False):
        assert os.path.isdir(result_path), "The result path: {0} doesn't exsits".format(result_path)
        if not load:
            dir_name = "D={0}".format(D)
            #dir_name = "new_D={0}".format(D)
            self.result_dir = os.path.join(result_path, dir_name)
            if os.path.exists(self.result_dir):
                if is_tmp:
                    self.enforce_delete_dir()
                else:
                    pass
                    assert False, "There is already permanet directory here: {0}".format(self.result_dir)
            os.mkdir(self.result_dir)
        else:
            self.result_dir = result_path
        self.base_result_dir = self.result_dir
        self.set_logging()

    def set_logging(self):
        logFormatter = colorlog.ColoredFormatter("%(log_color)s%(message)s", log_colors=self.COLORS)
        self.logger = logging.getLogger()
        
        fileHandler = logging.FileHandler(os.path.join(self.result_dir, "log.txt"))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)
        
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        self.logger.setLevel(logging.INFO)

    def set_result_path(self, dnf_size, epsilon):
        self.result_dir = os.path.join(self.base_result_dir, "DNF_size={0}".format(dnf_size))
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.result_dir = os.path.join(self.result_dir, "epsilon={0}".format(epsilon))
        os.mkdir(self.result_dir)
        param_text_file_path = os.path.join(self.result_dir, "param_file.txt")
        with open(param_text_file_path, "w") as f:
            f.write(self.PARAM_TEXT.format(time=datetime.now(), D=D, lr=LR, dnf_size=dnf_size, epsilon=epsilon, second_layer_bais=SECOND_LAYER_BAIS))

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
        fig.savefig(os.path.join(result_path, self.CLUSTER_GRAPH_B_NAME), bbox_inches="tight")
        plt.close(fig)