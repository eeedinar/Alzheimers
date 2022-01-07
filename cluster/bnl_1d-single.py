#!/usr/bin/env python3

from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d
import numpy as np
import pylab as plt
import os, time

exp_folder = '/home/bashit.a/July-2021/'
samples   = '/home/bashit.a/July-2021/sample-6/'
file = "2029_Dentate_gyrus-roi0.h5"

os.chdir(samples)

de = h5exp(exp_folder+"exp.h5")
#de.qgrid
print(de.detectors[1].extension)
qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
#print(qgrid2)

#load one file in a directory
dt  = h5xs(file, [de.detectors, qgrid2])
tic = time.time()
print(f'Loading file at {tic}')
dt.load_data(N=8, debug=True)
tac = time.time()
print(f'total processing time {tac-tic} seconds')
