from defs import *

NEGATIVE = -1
POSITIVE = 1

D = 784
LR_FIX_LAYER = 1e-2
LR_TWO_LAYER_REGULAR = 1e-2

SAMPLE_PROB_LIST = [0.009, 0.01, 0.025, 0.05, 0.075, 0.1, 0.12, 0.15]

# epsilon intialization
SIGAM_EPSILON = 1e-5

# gaussion initialization
SIGMA_GAUSS = 1e-5
R_GAUSS_FIX_LAYER = 200
R_GAUSS_TWO_LAYER_REGULAR = 100

# results consts
IS_TEMP = False
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\results"
MAX_TRY_TO_DELETE_DIR = 4

PRINT_STEP_JUMP =10
MAX_STEPS = 70000
TYPE = np.float32


