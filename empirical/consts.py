from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

### Target params ###
D = 9
DNF = [3, 3, 3]
FULL = False

### Hyper params ###
LR = 1e-3
R = 700
SIGMA = 1e-6

### Test params ###
TEST_SIZE = 10
TRAIN_SIZE = 30
TRAIN_SIZE_LIST = range(100, 200, 50) 
NUM_OF_RUNNING = 4

### Learning ###

PRINT_STEP_JUMP = 100
MAX_STEPS = 1000000
ATTEMPT_NUM = 3

### After learning ###
PRUNE_FACTOR_RANGE = np.arange(0.1, 1, 0.1)
RECONSTRACTION_FACTOR_RANGE = np.arange(0.1, 1, 0.1)

### MNIST ###
POSITIVE_NUMBERS = [2,4]
NEGATIVE_NUMBERS = [1,3]
TRAIN_SET_PRECENT = 0.8
BINARY_THRESHOLD = 127
BATCH_SIZE = 5000

### Result ###
IS_TEMP = False
TEMP_RESULT_PATH = "tmp"
GENERAL_RESULT_PATH = "results"
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


### Plot params ###
#TRAIN_SET_SIZE = 400