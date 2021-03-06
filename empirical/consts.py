from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

EXTRA_TO_NANE = ''

### Target params ###
D = 100
DNF = [3,3]
MAX_LITERAL_REPEAT = [1, 10, 25, 50, 80]
NUMBER_OF_TERMS = 500
TERM_SIZE = 4
PARITY_SIZE = 8
OVERLAP_SIZE = 1

### Hyper params ###
LR = 1e-3
R = 600
R_SVN = 5000
SIGMA = 1e-6
SIGMA_LARGE = 1
LR_FIX = 1e-3
LR_STA = 1e-2

### Hyper params to NTK### 
SIGMA_1 = 1
SIGMA_2 = 1

### Test params ###
TEST_SIZE = 500
TRAIN_SIZE = 250
TRAIN_SIZE_LIST = list(range(40, 100, 20)) 
FULL = True
REMOVE_SAMLPE_RANGE = range(0, 2100, 100) 
NUM_OF_RUNNING = 1

### Learning ###
PRINT_STEP_JUMP = 5000
MAX_STEPS = 100
CROSSENTROPY_THRESHOLD = 0.0002
HINGELOSS_THRESHOLD = 1e-3
ATTEMPT_NUM = 100

### After learning ###
PRUNE_FACTOR_RANGE = np.arange(0, 1, 0.1)
RECONSTRACTION_FACTOR_RANGE = np.arange(0, 1, 0.2)

### MNIST ###
POSITIVE_NUMBERS = [2,4]
NEGATIVE_NUMBERS = [1,3]
TRAIN_SET_PRECENT = 0.8
BINARY_THRESHOLD = 127
BATCH_SIZE = 32

### Result ###
IS_TEMP = True
TEMP_RESULT_PATH = "tmp"
GENERAL_RESULT_PATH = "results"
CONST_FILE_NAME = "consts.py"
MAX_TRY_TO_DELETE_DIR = 4

### State File ###
STATE_PATH = 'state.pkl'

### Dataset Path ##
KR_VS_KP_PATH = r"D:\checkouts\read_once_dnf\common\datasets\dataset_3_kr-vs-kp.csv"
KR_VS_KP_BIMARY_PATH = r"D:\checkouts\read_once_dnf\common\datasets\kr-vs-kp_binary.csv"
SLICE_PATH = r"D:\checkouts\read_once_dnf\common\datasets\dataset_46_splice.csv"
SLICE_BIMARY_PATH = r"D:\checkouts\read_once_dnf\common\datasets\splice_binary.csv"
DIABETES_PATH = r"D:\checkouts\read_once_dnf\common\datasets\diabetes_data_upload.csv"
DIABETES_BIMARY_PATH = r"D:\checkouts\read_once_dnf\common\datasets\diabetes_binary.csv"
PIMA_PATH = r"D:\checkouts\read_once_dnf\common\datasets\dataset_37_diabetes.csv"
PIMA_BIMARY_PATH = r"D:\checkouts\read_once_dnf\common\datasets\pima_diabetes_binary.csv"
BALANCE_PATH = r"D:\checkouts\read_once_dnf\common\datasets\balance-scale.csv"
BALANCE_BIMARY_PATH = r"D:\checkouts\read_once_dnf\common\datasets\balance-scale_binary.csv"


### Mariano ###
START_EPSILON = 0.01
END_EPSILON = 0.5
NUMBER_OF_EPSILONS = 50
VALIDATION_SIZE = 0.1

### Plot params ###
TRAIN_SET_SIZE = 0.9