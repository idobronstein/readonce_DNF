from defs import *

NEGATIVE = -1
POSITIVE = 1

D = 12
LR_FIX_LAYER = 1e-3
LR_TWO_LAYER_REGULAR = 5e-2

SAMPLE_PROB_LIST = [0.1, 0.2, 0.4, 0.6, 0.8, 1]

# epsilon intialization
SIGAM_EPSILON = 1e-5

# gaussion initialization
SIGMA_GAUSS = 3e-5
R_GAUSS_FIX_LAYER = 240
R_GAUSS_TWO_LAYER_REGULAR = 300

# results consts
IS_TEMP = True
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\results"
MAX_TRY_TO_DELETE_DIR = 4

PRINT_STEP_JUMP = 1000
MAX_STEPS = 1000000
FLOAT_TYPE = np.float32


