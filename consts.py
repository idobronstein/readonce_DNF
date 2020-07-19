import copy
import os
from datetime import datetime
import pickle
import shutil
import numpy as np
import pylab
import csv
from itertools import combinations
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

D = 12
LR = 1e-2
SIGMA = 1e-5
ALPHA = 0

NEGATIVE = -1
POSITIVE = 1

ZERO_THRESHOLD = 1e-8
assert ZERO_THRESHOLD < SIGMA and ZERO_THRESHOLD < LR / 2 ** D

MAX_VALUE_FOR_POSITIVE_SAMPLE = 2
MIN_VALUE_FOR_NEGATIVE_SAMPLE = 0

#GENERAL_RESULT_PATH = r"D:\checkouts\read_once_dnf\results"
GENERAL_RESULT_PATH = r"C:\temp\2020-07-12\results7"

FLOAT_TYPE = np.float32

PRINT_STEP_JUMP = 50
MAX_STEPS = 300

MOUNT_OF_UPSAMPELING = 25
PROB_OF_DOWNSAMPLING = 0.8

SIZE_OF_MAX_UNION = 1
PLOT_UOF_JUMP = 50

A_RANGE =  np.arange(-0.2, 1.2, 0.05)
B_RANGE = np.arange(-4.5, 0.5, 0.05)
A_INIT = 0.7
B_INIT = 0
OPISITE_VALUE = 1