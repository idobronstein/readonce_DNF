from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.int64

### Hyper params ###
D = 20
ALPHA = 1000 * 2**D
LR = 100

### Network  ###
HINGE_LOST_CONST = ALPHA
SECOND_LAYER_BAIS = - ALPHA
MIN_EPSILON = 1
MAX_EPSILON = 100 
STEP_EPSILON = 5
MAX_VALUE_FOR_POSITIVE_SAMPLE = 2 * abs(SECOND_LAYER_BAIS)
MIN_VALUE_FOR_NEGATIVE_SAMPLE = 0

### Learning ###
PRINT_STEP_JUMP = 500
MAX_STEPS = 500000

### After learning ###
PRUNE_FACTOR_WEIGHT = 2
PRUNE_FACTOR_TOTAL_NORM = 1
RECONSTRACTION_FACTOR_WEIGHT = 10
RECONSTRACTION_FACTOR_NORM = 8

### Result ###
IS_TEMP = True
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\results"
BACKUP_DIR = 'backup'
PRONE_DIR = 'prone'
PRONE_BY_MEAN_DIR = 'prone_by_mean'
ORIGINAL_FINAL_DIR = 'final'
MAX_TRY_TO_DELETE_DIR = 4