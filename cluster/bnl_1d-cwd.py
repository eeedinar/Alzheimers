
from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d
import numpy as np
import pylab as plt
import os, time

exp_file = '/scratch/bashit.a/BNL-Data/Mar-2023/Done/'
samples   = '/scratch/bashit.a/BNL-Data/Mar-2023/Misc/'

os.chdir(samples)

de = h5exp(exp_file+"exp.h5")
#de.qgrid
print(de.detectors[1].extension)
qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
#print(qgrid2)

### load all files in a directory
def load_all_cwd():
    for file in [each for each in os.listdir(os.getcwd()) if each.endswith('.h5')]:
        try:
            print(f'Loading file {file}')
            dt  = h5xs(file, [de.detectors, qgrid2])
            dt.load_data(N=8)
        except:
            print(f'Loading failed {file}')

tic = time.time()
print(f'Loading file at {tic}')            
load_all_cwd()
print(f'total processing time {time.time()-tic} seconds')
