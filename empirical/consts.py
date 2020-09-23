from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

### Target params ###
D = 784
DNF = [4] * 3
FULL = False

### Hyper params ###
LR = 1e-4
R = 500
SIGMA = 1e-5

### Test params ###
TEST_SIZE = 5000
TRAIN_SIZE_LIST = range(100, 3100, 100)
NUM_OF_RUNNING = 10

### Learning ###
PRINT_STEP_JUMP = 100
MAX_STEPS = 100000
ATTEMPT_NUM = 3

### After learning ###
PRUNE_FACTOR_RANGE = np.arange(0.1, 1, 0.1)
RECONSTRACTION_FACTOR_RANGE = np.arange(0.3, 1, 0.1)

### MNIST ###
POSITIVE_NUMBERS = [2,4]
NEGATIVE_NUMBERS = [1,3]
TRAIN_SET_PRECENT = 0.8
BINARY_THRESHOLD = 127
BATCH_SIZE = 1000

### Result ###
IS_TEMP = True
TEMP_RESULT_PATH = "tmp"
GENERAL_RESULT_PATH = "results"
MAX_TRY_TO_DELETE_DIR = 4

### State File ###
STATE_PATH = 'state.pkl'

