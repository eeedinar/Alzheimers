#!/usr/bin/env python3
"""
Author: Abdullah Al Bashit
Ph.D. Student in Electrical Engineering
Northeastern University, Boston, MA
Date: 01/01/2021
"""
########################### ---------- Essential functions ---------- ###########################

## import packages
from essential_func import *
import scipy.io as sio

def save_mat(output_filename, names, values):
    """
        save_mat('IqBS', ('IqBS', 'qgrid2'), (IqBS, qgrid2)) --> generates IqBS.mat file with IqBS and qgrid2 variables
    """
    mat_contents = {}
    for name,value in zip(names, values):
        mat_contents[name] = value
    sio.savemat(output_filename+'.mat', mat_contents)


def generate_excel_file(file, qgrid, scattering, method = 'all-frames', frame= None, folder='CSV'):
    """
        file = '1943_B1a_masked.h5'
        scattering = 'merged'
        
        generate_excel_file(file, qgrid2, scattering, method='one-frame', frame=0)
        generate_excel_file(file, qgrid2, scattering, method='all-frames')
    """
    
    if method == 'all-frames':
        with h5py.File(file,'r') as hdf:
            Iq = hdf.get(h5_top_group(file) + '/processed')      # Iq = hdf.get('2048_B16/processed')
            Iq = np.array(Iq.get(scattering))                          # Iq = np.array(2048_B16/processed/merged')

        idx_l, idx_u, _ = valid_idx_search(qgrid, Iq[:,0,:])
        print(f'{scattering} Q = , {qgrid[idx_l:idx_u]}')
        #Iq = Iq[:,0,idx_l:idx_u]                                  # Iq shape (3721, 130) skipping dI    
        Iq = Iq[:,0,:]                                            # print including nan values
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savetxt(f'{folder}/{file}_{scattering}.csv', Iq, delimiter=",")           # https://www.geeksforgeeks.org/convert-a-numpy-array-into-a-csv-file/

    elif method == 'one-frame':
        scatterings = ('_SAXS', '_WAXS2', 'merged')
        Iq_t = np.zeros((len(qgrid), 1 + len(scatterings)))
        Iq_t[:,0] = qgrid

        for idx, scattering in enumerate(scatterings):
            with h5py.File(file,'r') as hdf:
                Iq = hdf.get(h5_top_group(file) + '/processed')      # Iq = hdf.get('2048_B16/processed')
                Iq = np.array(Iq.get(scattering))                          # Iq = np.array(2048_B16/processed/merged')
                Iq = Iq[frame, 0]                                              # Iq values only
                Iq_t[:,idx+1] = Iq

        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savetxt(f'{folder}/{file}_{frame}.csv', Iq_t , delimiter=",")           # https://www.geeksforgeeks.org/convert-a-numpy-array-into-a-csv-file/
        
    else:
        return 'something wrong - please check'

