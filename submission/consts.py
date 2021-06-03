from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.float32

### Target params ###
D = 6
DNF = [3,3]

### Hyper params ###
LR = 1e-3
R = 600
SIGMA = 1e-6
SIGMA_LARGE = 1
BATCH_SIZE = 32
CROSSENTROPY_THRESHOLD = 0.01
MAX_STEPS = 1000000

### Run params ###
TEST_SIZE = 500
TRAIN_SIZE = 100
TRAIN_SIZE_LIST = list(range(10, 100, 10)) 
NUM_OF_RUNNING = 2

### After learning ###
PRUNE_FACTOR_RANGE = np.arange(0, 1, 0.1)
RECONSTRUCTION_FACTOR_RANGE = np.arange(0, 1, 0.2)

### Result ###
RESULT_PATH = "results_4"
PRINT_STEP_JUMP = 5000

### StatisticalQuery ###
START_EPSILON = 0.01
END_EPSILON = 0.5
NUMBER_OF_EPSILONS = 50
VALIDATION_SIZE = 0.1
