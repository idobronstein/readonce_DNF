from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

### Target params ###
D = 50
DNF = [5] * 9
FULL = False

### Hyper params ###
LR = 1e-3
R = 10000
SIGMA = 1e-5

### Compartion params ###
TEST_SIZE = 5000
TRAIN_SIZE_LIST = range(150, 750, 50)
NUM_OF_RUNNING = 2

### Graphs ###
TRAIN_SIZE = 5000

### Learning ###
PRINT_STEP_JUMP = 100
MAX_STEPS = 1000000
ATTEMPT_NUM = 3

### After learning ###
PRUNE_FACTOR_WEIGHT = 10
PRUNE_FACTOR_TOTAL_NORM = 4
RECONSTRACTION_FACTOR_WEIGHT = 10
RECONSTRACTION_FACTOR_NORM = 5

### Result ###
IS_TEMP = True
TEMP_RESULT_PATH = "tmp"
GENERAL_RESULT_PATH = "results"
MAX_TRY_TO_DELETE_DIR = 4

### State File ###
STATE_PATH = 'state.pkl'

