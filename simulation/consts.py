from defs import *

D = 12
LR = 4e-3
SIGMA = 1e-6
ALPHA = 0
SECOND_LAYER_BAIS = -1

NEGATIVE = -1
POSITIVE = 1

ZERO_THRESHOLD = 1e-8
assert ZERO_THRESHOLD < SIGMA and ZERO_THRESHOLD < LR / 2 ** D

MAX_VALUE_FOR_POSITIVE_SAMPLE = 2
MIN_VALUE_FOR_NEGATIVE_SAMPLE = 0

IS_TEMP = False
TEMP_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\tmp"
GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\simulation\results"
BACKUP_DIR = 'backup'
PRONE_DIR = 'prone'
ORIGINAL_FINAL_DIR = 'final'
MAX_TRY_TO_DELETE_DIR = 4

FLOAT_TYPE = np.float32

PRINT_STEP_JUMP = 50
MAX_STEPS = 10000

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