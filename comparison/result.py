from consts import *
from utilits import *

class Result():

    PARAM_FILE_NAME = "param_file.txt"

    def __init__(self, result_path=TEMP_RESULT_PATH, is_tmp=True, new_D=None):
        assert os.path.isdir(result_path), "The result path: {0} doesn't exsits".format(result_path)
        dir_name = "D={0}".format(D)    
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
            f.write("lr fix        -  {}\n".format(LR_FIX_LAYER))
            f.write("lr normal     -  {}\n".format(LR_TWO_LAYER_REGULAR))
            f.write("sigma epsilon -  {}\n".format(SIGAM_EPSILON))
            f.write("sigma gauss   -  {}\n".format(SIGMA_GAUSS))
            f.write("r epsilon fix -  {}\n".format(2 ** D))
            f.write("r gauss fix   -  {}\n".format(R_GAUSS_FIX_LAYER))
            f.write("r gauss normal-  {}\n".format(R_GAUSS_TWO_LAYER_REGULAR))

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

    def save_graph(self, run_name, all_algorithems, result_vec):
        result_path = os.path.join(self.result_dir, run_name)
        fig, ax = plt.subplots()
        for i in range(len(all_algorithems)):
            ax.plot(SAMPLE_PROB_LIST, result_vec[i], all_algorithems[i][2], label=all_algorithems[i][1])
        legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
        fig.savefig(result_path)
        plt.clf()