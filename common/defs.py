import copy
import os
import pickle
import shutil
import pylab
import warnings
import colorama
import random
import math
from datetime import datetime
from sklearn.svm import SVC
import numpy as np
from datetime import datetime
import tensorflow.compat.v1 as tf
from itertools import combinations
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import logging
import colorlog
import pandas as pd

#import mnist

tf.disable_v2_behavior() 

colorama.init(convert=True)

warnings.filterwarnings('ignore')
