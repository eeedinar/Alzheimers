#!/usr/bin/env python3

# Import module from parent directory for py4xs
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d
import numpy as np
import pylab as plt
import time
from essential_func import *
import multiprocessing

# specs
samples    = '/scratch/bashit.a/BNL-Data/Mar-2023/1971/'
exp_folder = '/scratch/bashit.a/BNL-Data/Mar-2023/'
json_file  = "config-mar-2023-1971-withthreshold.json"

# semi-spec
os.chdir(samples)
data = get_json_str_data(os.path.join(samples, json_file))
method = 'thr_rec_circ_patch'                                      # 'thresholding' or 'rec_circ_patch' or 'thr_rec_circ_patch' 
qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])


### do masking and 1-d averaging (Lin Yang's code by BNL used here for 1-d averaging)
tic = time.time()
#### multiprocessing starts -------------------------
processes = []   # for multiprocessing
files = []

for file in data['files']:
    source_file = file['name']
    try: 
        if method == 'rec_circ_patch':
            patches = file['patches']
            print(source_file, 'patches to -->', patches)
            masked_file = circ_avg_from_patches(source_file, qgrid2, patches, method, exp_folder) 
        elif method == 'thresholding': 
            patches = file['threshold']
            print(source_file, 'thresholded to -->', patches)
            masked_file = circ_avg_from_patches(source_file, qgrid2, args=tuple(patches), method = 'thresholding', exp_folder=exp_folder)  # args must be of tuple
        elif method == 'thr_rec_circ_patch':
            if file['patches'] or file['threshold']:   # making sure either or both pathes and thresholding exists otherwise skip 
                dset_waxs_sum = loading_dset_waxs_sum(source_file, load_from = 'npz')
                patches_arg = file['patches']
                thr_args    =  dset_waxs_sum, *file['threshold']
                args = (thr_args, patches_arg)
                masked_file = circ_avg_from_patches(source_file, qgrid2, args, method, exp_folder)

                # p = multiprocessing.Process(target = circ_avg_from_patches, args = [source_file, qgrid2, args, method, exp_folder])      # pixalated_sum_waxs(file, save_as_file=True, save_as_file_only = True)
                # p.start()
                # processes.append(p)
            else:
                continue
    except: 
        continue
    files.append((source_file, masked_file))
    print(files)

### join processes started in for loop
# for process in processes:
#     process.join()

print(f'Successfully created (source file, masked file) \n\n')
[print(idx+1, '. ', s, '--->' ,t) for idx, (s,t) in enumerate(files)]

print(f'Program finished in {time.time()-tic} seconds')
