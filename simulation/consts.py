from defs import *

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.int64

### Hyper params ###
D = 9
ALPHA = 1000 * 2**D
LR = 100 * 2**D

### Network  ###
HINGE_LOST_CONST = ALPHA
SECOND_LAYER_BAIS = - ALPHA
MIN_EPSILON = 2**D
MAX_EPSILON = 100 *2**D
STEP_EPSILON = 20 * 2 **D
MAX_VALUE_FOR_POSITIVE_SAMPLE = 2 * abs(SECOND_LAYER_BAIS)
MIN_VALUE_FOR_NEGATIVE_SAMPLE = 0

### Learning ###
PRINT_STEP_JUMP = 500
MAX_STEPS = 500000

### After learning ###
PRUNE_FACTOR_WEIGHT = 2
PRUNE_FACTOR_TOTAL_NORM = 1
RECONSTRACTION_FACTOR_WEIGHT = 2
RECONSTRACTION_FACTOR_TOTAL_NORM = 1

### Result ###
IS_TEMP = False
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\results"
BACKUP_DIR = 'backup'
PRONE_DIR = 'prone'
PRONE_BY_MEAN_DIR = 'prone_by_mean'
ORIGINAL_FINAL_DIR = 'final'
MAX_TRY_TO_DELETE_DIR = 4

##################################### OLD #####################################

MOUNT_OF_UPSAMPELING = 25
PROB_OF_DOWNSAMPLING = 0.8
SIZE_OF_MAX_UNION = 1
PLOT_UOF_JUMP = 50
A_RANGE =  np.arange(-0.2, 1.2, 0.05)
B_RANGE = np.arange(-4.5, 0.5, 0.05)
A_INIT = 0.7
B_INIT = 0
OPISITE_VALUE = 1