from defs import *

### Hyper params ###
D = 12
UNIT = 2**D
LR = UNIT * 10

### General ###
NEGATIVE = -1
POSITIVE = 1
TYPE = np.int32

### Network  ###
SECOND_LAYER_BAIS = - UNIT * 10000
MAX_VALUE_FOR_POSITIVE_SAMPLE = 2 * abs(SECOND_LAYER_BAIS)
MIN_VALUE_FOR_NEGATIVE_SAMPLE = 0

### Result ###
IS_TEMP = False
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\results"
BACKUP_DIR = 'backup'
PRONE_DIR = 'prone'
PRONE_BY_MEAN_DIR = 'prone_by_mean'
ORIGINAL_FINAL_DIR = 'final'
MAX_TRY_TO_DELETE_DIR = 4

### Learning ###
PRINT_STEP_JUMP = 100
MAX_STEPS = 70000

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