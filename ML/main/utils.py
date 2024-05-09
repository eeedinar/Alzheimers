import numpy as np

def h5File_h5Dir_csv_loc_by_h5file(file, BNL_dir, sub_dir):
    """ 
        h5File_h5Dir_csv_loc_by_h5file( file = "1948_HIPPO-roi1_0_0_masked_intp.h5",  
                                    BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023' ,
                                    sub_dir     = "CSV_Conv-8-point")     
    """
    for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
        if file_search.find(file) > -1 and file_search.endswith(".h5"):
            directory = os.path.dirname(file_search)
        if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
            csv_file = file_search
            break
    return file, directory, csv_file


### snaking function
def snaking(Width, Height, X=None,):
    """
        img_orig = snaking(Width, Height, diff_patterns)
    """
    X = copy.deepcopy(X)    # deep copy
    if X is None:
        X_coor = np.arange(Height*Width).reshape(Height,Width)
    else:
        X_coor = X.reshape(Height,Width)            # input id numpy reshaped as height * width

    img_orig = np.flipud(X_coor)                                     # flipud matrix
    idx_even = np.arange(Height-2,-1,-2)                             # Get odd indices to leave to as it is
    img_orig[idx_even,:] = np.fliplr(img_orig[idx_even])             # flipping left to right 
    return img_orig

### snaking width_height extraction
## attrs in h5py is recorded in json format

def width_height(file, directory=None):
    """
        Width, Height = width_height(file)
    """
    directory = os.getcwd() if directory == None else directory
    with h5py.File(os.path.join(directory,file),'r') as hdf:
        dset = hdf.get(h5_top_group(file))
        header = json.loads(dset.attrs['start'])
        Height, Width = header['shape']
    return Width, Height

def h5_top_group(file):
    """ 
        input argument string of a h5py file name
        returns seperator
        seperator = h5_top_group(file)
    """
    return file.split('_masked')[0] if file.find('_masked') >= 0 else file.split('.')[0]

# assign snaking heapmap A to labels
def from_clusterFr_ceffs_to_matrix(A, cluster, coeffs):
    """
        cluster = [2066, 2067]
        coeffs  = [0.98, 0.99]
        A = np.array([np.zeros((Height,Width)),sna])                  # zero values matrix (A[0]=0) with frame numbers depth (A[1]=frames)
        
        funciton call: A = from_clusterFr_ceffs_to_matrix(A, cluster, coeffs)
    """
    for i, frame in enumerate(cluster):
        A[0][np.where(A[1]==frame)]=coeffs[i]

    B = A
    return B

def idx_from_grid(qvalue_lower_bound, qvalue_upper_bound):
    """
        return idx from qgrid2 
        qvalue_lower_bound = 0.66
        qvalue_upper_bound = 1.45
        idx_from_grid(qvalue_lower_bound, qvalue_upper_bound) -> (182, 340, 158)
    """
    qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
    lidx = np.argmin(qgrid2 < qvalue_lower_bound)   # qvalue = 0.7,  idx = 190
    uidx = np.argmin(qgrid2 < qvalue_upper_bound)   # qvalue = 1.46, idx = 342
    input_dim = (uidx - lidx)                       # (342-190) = 152
    return lidx, uidx, input_dim