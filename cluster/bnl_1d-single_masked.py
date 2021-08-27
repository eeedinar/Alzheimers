
# Import module from parent directory
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d
import numpy as np
import pylab as plt
import os, sys, inspect, time
from essential_func import *
qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])

exp_folder = '/home/bashit.a/July-2021/'
samples    = '/home/bashit.a/July-2021/sample-5/'
file       = '2512_EC-roi0.h5'

args =  0, 62100, 36300   # manually can set args = (a_min.value, a_max.value, thr.value) = 0, 30000, 16000 or Get values from 3rd cell print output

os.chdir(samples)

tic = time.time()
print(f'Loading file at {tic}')

masked_file = circ_avg_from_patches(file, qgrid2, args=args, method = 'thresholding', exp_folder = exp_folder)

tac = time.time()
print(f'total processing time {tac-tic} seconds')
