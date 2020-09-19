from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

### Target params ###
D = 14
DNF = [4] * 3
FULL = True

### Hyper params ###
LR = 1e-3
R = 1000
SIGMA = 1e-5

### Compartion params ###
TEST_SIZE = 5000
TRAIN_SIZE_LIST = range(150, 750, 50)
NUM_OF_RUNNING = 5

### Graphs ###
TRAIN_SIZE = 5000

### Learning ###
PRINT_STEP_JUMP = 1000
MAX_STEPS = 500000

### After learning ###
PRUNE_FACTOR_WEIGHT = 10
PRUNE_FACTOR_TOTAL_NORM = 4
RECONSTRACTION_FACTOR_WEIGHT = 10
RECONSTRACTION_FACTOR_NORM = 9

### Result ###
IS_TEMP = False
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\comparison\results"
MAX_TRY_TO_DELETE_DIR = 4




