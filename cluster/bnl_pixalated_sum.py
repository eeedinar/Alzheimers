#!/usr/bin/env python3

# Import module from parent directory
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

samples   = '/home/bashit.a/July-2021/sample-1/'

os.chdir(samples)
files_lists = multiprocessing_lists_of_files_list(seek_str = '.h5', category_max_size_GB = 50)
tic = time.time()      

#### multiprocessing starts -------------------------
processes = []
for files_list in files_lists:
    ### load lists inside the list
    for file in files_list:
        try:
            print(f'Loading file {file}')
            p = multiprocessing.Process(target = pixalated_sum_waxs, args = [file, True, True])      # pixalated_sum_waxs(file, save_as_file=True, save_as_file_only = True)
            p.start()
            processes.append(p)
        except:
            print(f'Loading failed {file}')

    ### join processes started in for loop
    for process in processes:
        process.join()
#### multiprocessing ends -------------------------

print(f'Program finished in {time.time()-tic} seconds')
