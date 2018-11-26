
import os
import sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib
import h5py
import argparse
import random


#Numpy
import numpy as np
from numpy import unravel_index


import datetime
import time
import gzip

from sklearn import svm, datasets
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


import glob
import fnmatch
import scipy
from scipy.fftpack import fft
from scipy import ndimage
from scipy import misc

import csv
import matplotlib.patches as patches

from PIL import Image
from collections import Counter


# If LINUX OS
if sys.platform == "linux" or sys.platform == "linux2":
  # TODO: Add directories.
  print ("LINUX OS")
  import matplotlib.pyplot as plt

  
# If Mac OSX
elif sys.platform == "darwin":
  print ("MAC OSX")  
  matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt  


plt.ion()

