#!/usr/bin/env python3
"""
Author: Abdullah Al Bashit
Ph.D. Student in Electrical Engineering
Northeastern University, Boston, MA
Date: 01/01/2021
"""
########################### ---------- Essential functions ---------- ###########################

## import packages    
import os, shutil, h5py, time, json, cv2
import numpy as np
import pylab as plt
import matplotlib.backends.backend_pdf                        # pdf generation package
import ipywidgets 
from PyPDF2 import PdfFileMerger

# Lin Yang's BNL packages for 1-d averaging
from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import Data2d    

## create 1d data from lin Yang's code (py4xs packages are used here)
def azimuthal_averaging(file, qgrid, n_proc=8):
    """
        azimuthal_averaging(masked_file, qgrid, n_proc=8)
    """
    de = h5exp("exp.h5")
    qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])

    dt  = h5xs(file, [de.detectors, qgrid])
    tic = time.perf_counter()
    print(f'Circular averaging starts now ... ')
    dt.load_data(N=n_proc)
    print(f'{file} total 1-d averaging time {time.perf_counter() - tic} seconds')


### valid indices search function
def h5_top_group(file):
    """ 
        input argument string of a h5py file name
        returns seperator
        seperator = h5_top_group(file)
    """
    return file.split('_masked')[0] if file.find('_masked') >= 0 else file.split('.')[0]

def valid_idx_search(qgrid, Iq):
    """
        Iq shape (#frames, #1-d avg. values) ex. (3721,690)

        function call: 
        idx_l, idx_u, valid_diff_values = valid_idx_search(qgrid2, Iq)
        
        retrun:
            idx_l = first valid   indices    SAXS = 2,   WAXS = 109 , merged = 0
            idx_u = first invalid indices    SAXS = 125, WAXS = 579 , merged = 690
            valid_diff_values = (#frames, only valid values)  ex. 
            '_SAXS'  = (4941, 123) --> 125 - 2 = 123
            '_WAXS2' = (4941, 470) --> 579-109 = 470
    """
    list_ = ~np.isnan(Iq[0,:].flatten())                      # list_ = [False, False, True, True, ...] boolean list in the first frame 
    idx_l = np.argmax(list_==True)                            # find first occurance of a boolean True ; SAXS = 2,   WAXS = 109 , merged = 0
    idx_u  = len(list_) - np.argmax(list_[::-1]==True)        # find last occurance of a invalid number;   SAXS = 125 , WAXS = 579 , merged = 690 valid on 124, 578 , 689 indices
    print(f'low valid idx = {idx_l}, low valid Q = {qgrid[idx_l]:0.3f}, high valid idx = {idx_u-1} , high valid Q = {qgrid[idx_u-1]:0.3f}')
    
    valid_diff_values = Iq[:, idx_l:idx_u]                 # SAXS-Iq --> (1,2:125,:3721)
    
    return idx_l, idx_u, valid_diff_values

### find representative value for each frame
def find_rep_value(qgrid, Iq, args=None, method = 'polyfit'):
    """
    Iq (3721,690) may contain valid or invalid values
    for polynomial averaging:
        method = 'polynomial' 
    for circular averaging:
        method = 'circ'
    diff_patterns = find_rep_value(qgrid2, Iq, args, method = 'polyfit')
    diff_patterns = find_rep_value(qgrid2, Iq, method = 'circ')
    
    """    
    if method == 'polyfit':
        poly_ord, idx_l, idx_u, limit_l, limit_u = args
        n_patterns = len(Iq)
        diff_patterns = [];
        for frame in range(n_patterns):                      # range(100) --> look for first 100 frames

            X = qgrid[idx_l:idx_u]
            y = Iq[frame][idx_l:idx_u]

            # Find coefficients
            coefs = np.polyfit(X,y,poly_ord)
            #print(coefs)

            X_test = X[limit_l:limit_u]
            y_test = y[limit_l:limit_u]
            diff_patterns.append(np.max( y_test - np.polyval(coefs, X_test) ));        # (0,2:125,0)  reshape diff_patterns shape (0, 2:125, 3721)
        diff_patterns = np.array(diff_patterns)      
    
    elif method == 'circ':
        n_patterns = len(Iq)
        _ , _ , valid_diff_values = valid_idx_search(qgrid, Iq)
        diff_patterns = np.zeros(len(valid_diff_values));
        
        for frame in range(n_patterns):                      # range(100) --> look for first 100 frames
            diff_patterns[frame] = np.mean(valid_diff_values[frame])

    return diff_patterns

### snaking function
def snaking( Width, Height, X=None,):
    """
        img_orig = snaking(Width, Height, diff_patterns)
    """
    if X is None:
        X_coor = np.arange(Height*Width).reshape(Height,Width)
    else:
        X_coor = X.reshape(Height,Width)            # input id numpy reshaped as height * width
    img_orig = np.zeros_like(X_coor)                  # 
    temp = np.flipud(X_coor)                        # flipud matrix
    idx_odd = np.arange(1,Height,2)                 # Get odd indices to fliplr
    img_orig[idx_odd,:]  = np.fliplr(temp[idx_odd])   # flipping odd indices
    idx_even = np.arange(0,Height,2)                # Get odd indices to leave to as it is
    img_orig[idx_even,:] = temp[idx_even]             # leaving as it is
    return img_orig

### snaking width_height extraction
## attrs in h5py is recorded in json format

def width_height(file):
    """
        Width, Height = width_height(file)
    """    
    with h5py.File(file,'r') as hdf:
        dset = hdf.get(h5_top_group(file))
        header = json.loads(dset.attrs['start'])
        Height, Width = header['shape']  
    return Width, Height

### read Iq data from hdf file
def read_Iq(file, scattering):
    """
        read Iq data
        Iq = read_Iq(file, scattering)
    """
    with h5py.File(file,'r') as hdf:
        Iq = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')
        Iq = np.array(Iq.get(scattering))                            # Iq = np.array(2048_B16/processed/merged')
        Iq = Iq[:,0,:]                                               # Iq shape (3721, 690)

    return Iq

### generate heatmap for differnet scatterings
def plot_heat_map_from_data(img_orig, Width, Height, args, title= None):
    """
        plot_heat_map_from_data(img_orig, Width, Height, args = None, title= None)
    """
    ########## --------- matplotlib mouse hovering function for snaking --------- ########## 
    frame_cor = snaking(Width, Height)                  # snaking indices for heat map, 
    numrows, numcols = Height, Width                    # format_coord function requires this global variables
    def format_coord(x, y):
        col = int(x)                                  # truncate x values
        row = int(y)                                  # truncate y values
        if 0 <= col < numcols and 0 <= row < numrows:
            z = np.flipud(frame_cor)[row, col]        # flipping to get correct value of z     
            return 'x=%1.2f, y=%1.2f, FRAME=%d' % (x, y, z)
        else:
            return 'x=%1.2f, y=%1.2f' % (x, y)        # outside the plotting range, no need
    
    ### plotting
    f, ax = args
    ax.clear()
    ax.imshow(img_orig, cmap = "viridis", interpolation = 'none', origin='upper', extent=[0,Width,0,Height], aspect='equal')
    ax.format_coord = format_coord
    ax.set(title = title, xticks = (np.arange(0,Width,5)), yticks = (np.arange(0,Height,5)))



def plot_heat_map_from_file(file, qgrid, scatterings = None, args = None):
    """
        Input args:
            must be tupple scattering = ('_SAXS',)
            args = (f, axs)
        f = plot_heat_map_from_file(file, qgrid, scatterings = scattering)
        f = plot_heat_map_from_file(file, qgrid, scatterings = None, args = (f,axs))
    """
    ## No given scattering argument plot all SAXS, WAXS, Merged
    if scatterings == None:
        scatterings = ('_SAXS',          '_WAXS2',        'merged'    )
    
    f, axs = plt.subplots(1, len(scatterings), num=f'{file} Heat maps', figsize=(10,5))

    if len(scatterings) == 1:   
        axs = [axs]       # for one scatterings input making axs a list to use in the loop as axs[i]

    ### mouse hovering function call
    Width, Height = width_height(file)
    
    for i, scattering in enumerate(scatterings):
        
        Iq = read_Iq(file, scattering)
        diff_patterns = find_rep_value(qgrid, Iq, method = 'circ')
        img_orig = snaking(Width, Height, diff_patterns)
        
        plot_heat_map_from_data(img_orig, Width, Height, args = (f, axs[i]), title= f'{scattering} {file}')
    #plt.tight_layout()
    plt.show()
    return f

def cwd_files_type_search(file_type):
    """
        files_sorted = cwd_files_type_search('.h5')
    """
    files = []
    for file in [each for each in os.listdir(os.getcwd()) if each.endswith(file_type)]:
        files.append(file)
    files_sorted = sorted(files)
    return files_sorted

def plot_all_heat_maps_cwd(file, qgrid, scatterings):
    """
        scatterings = ('_SAXS', '_WAXS2')
        
        Function call:
            plot_all_heat_maps_cwd('output.pdf', qgrid2, scatterings)
    """
    pdf = matplotlib.backends.backend_pdf.PdfPages('output.pdf')
    files_sorted = cwd_files_type_search('.h5')
    for file in files_sorted:
        print(f'Loading file {file}')
        
        try:       
            f = plot_heat_map_from_file(file, qgrid, scatterings = scatterings ) # figure object
            print(f'{file} Task Finished. Figure Number = ' , f.number)
        except:    continue
          
        pdf.savefig(f);   #f.savefig("foo.pdf", ) 

    pdf.close();
    print('PDF creation Finished')

### load config.json file in python
def get_json_str_data(file):
    """
        data = get_json_str_data("config.json")
    """
    with open(file, 'r') as f:
        data = json.load(f)         # read data from file to python object
        data = json.dumps(data)     # convert python obj to json format data
        data = json.loads(data)     # convert json formatted string into python dictionary
    return data

### set patch attributes
def set_patch_attributes(file, patches):
    """
        set_patch_attributes(file, patches)
    """
    with h5py.File(file,'r+') as hdf:
        
        dset = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')
        dset.attrs['patches'] = json.dumps(patches)

    return "patching information written on the h5 file processed directory"

### get patch attributes
def get_patch_attributes(file):
    """
        patches = get_patch_attributes(file)
    """
    with h5py.File(file,'r') as hdf:
        
        dset = hdf.get(f'{h5_top_group(file)}/processed')              # Iq = hdf.get('2048_B16/processed')
        patches = json.loads(dset.attrs['patches'])    
    
    return patches




def file_polyfit_heatmap_plot(file, indices, qgrid2):
    """
        file = '2048_B8.h5'
        ### q_low, q_high, q_low - idx[2] and q_high + idx[2] for poly fit, number of points
        indices = ((0.115, 0.16, 13,   48, '_SAXS' , 'bumpy'),      # 72, 81
                   (1.48,  1.57, 10, 4.15, '_WAXS2', 'paraffin'),   # 345, 355
                   (1.30,  1.38, 10,  4.7, '_WAXS2', 'amyloid'),    # 309, 323
                   (1.89,  1.93,  7,  3.2, '_WAXS2', 'mica')        # 427, 433
                  )

        file_polyfit_heatmap_plot(file, indices, qgrid2)
    """

    # polyfit and q value function
    f, axs = plt.subplots(2,len(indices), figsize = (15,9), num = f'{file}')

    def polyfit_heatmap_plot(frame_polyfit, poly_ord = 1):

        for idx_indices,index in enumerate(indices):

            ### extracting values from the patches
            idx_l_t = np.max(np.where(qgrid2 <= index[0]))  # 73
            idx_u_t = np.max(np.where(qgrid2 <= index[1]))  # 81
            idx_l =  idx_l_t - index[2]                     # 73 - 13 = 60
            idx_u =  idx_u_t + index[2]                     # 81 + 13 = 90
            limit_l = index[2]
            limit_u = index[2] + idx_u_t - idx_l_t
            scattering = index[4]
            comment = index[5]
            #print(idx_l, idx_u, limit_l, limit_u)

            ### read Iq data
            Iq = read_Iq(file, scattering)    # read Iq data

            X = qgrid2                        # X is qgrid value
            y = Iq[frame_polyfit]             # One frame Iq values

            #print(idx_l, idx_u, X[idx_l], y[idx_l], X[idx_u-1] , y[idx_u-1])
            X = qgrid2[idx_l:idx_u]           # get valid Iq values q rannge
            y = y[idx_l:idx_u]                # get valid Iq rannge

            coefs = np.polyfit(X,y,poly_ord)  # fit polynomial

            X_test = X[limit_l:limit_u]       # extract the narrow region for X
            y_test = y[limit_l:limit_u]       # extract the narrow region for y
            
            max_ind = np.argmax( y_test - np.polyval(coefs, X_test))
            print(f'{comment} max-point to fitting-point difference:  {np.max( y_test - np.polyval(coefs, X_test))}') 
            
            ## plot polynomial and heat map
            axs[0, idx_indices].clear()
            axs[0, idx_indices].scatter(X,y, color='red' , label = 'exp data') 
            axs[0, idx_indices].scatter(X_test, y_test, color='purple' , label = 'fit data')
            axs[0, idx_indices].scatter(X, np.polyval(coefs, X), color = 'black' , label = 'polyfit')
            axs[0, idx_indices].scatter(X_test[max_ind], y_test[max_ind], color = 'blue', marker='^', label = 'max point')
            axs[0, idx_indices].scatter(X_test[max_ind], np.polyval(coefs, X_test[max_ind]), color = 'orange' , marker=r'$\clubsuit$' ,label = 'ref point')
            axs[0, idx_indices].annotate("", (X_test[max_ind], y_test[max_ind]) , (X_test[max_ind], np.polyval(coefs, X_test[max_ind])), arrowprops={'arrowstyle':'<-'})

            axs[0, idx_indices].set(title = f'Poly_ord {poly_ord} Frame - {frame_polyfit} {comment}' ,
                        xlabel = 'X' , ylabel = 'y', xscale='linear', yscale = 'linear' )
            axs[0, idx_indices].legend()

            ## gather heat map data
            args = poly_ord, idx_l, idx_u, limit_l, limit_u
            diff_patterns = find_rep_value(qgrid2, Iq, args, method = 'polyfit')

            # snaked diff. patterns and plot heat map
            img_orig = snaking(Width, Height, diff_patterns)   
            plot_heat_map_from_data(img_orig, Width, Height, args = (f,axs[1, idx_indices]), title = (f'Q~{index[0], index[1]} {scattering} {index[3]}$\AA$'), )

    #        f.colorbar(pos, ax= axs[1, idx_indices])

        plt.suptitle(f'{h5_top_group(file)} Polyfit and Heat maps')
        plt.tight_layout()
        plt.show()
    
    ### frame no. and polyfit Int slider
    Width, Height = width_height(file)
    n_patterns = Width*Height

    frame = ipywidgets.IntSlider(min=0, max=n_patterns-1, value=0, description="Frame", continuous_update=False)
    poly_ord = ipywidgets.IntSlider(min=1, max=10, value=1, description='Poly fit order', continuous_update=False)
    ipywidgets.interact(polyfit_heatmap_plot, frame_polyfit=frame, poly_ord = poly_ord)

    ### save pdf widgets
    ipywidgets.interact_manual.opts['manual_name'] = 'Save PDF'
    @ipywidgets.interact_manual()
    def foo():
        f.savefig(f"{file}.pdf")

def pdfs_merging(directory = '', output = 'result.pdf'):
    """
        merge all pdfs in the current directory :
            directory = '', output = 'result.pdf')
        merge all pdfs in the /PDF directory : 
            directory = '/PDF', output = 'result.pdf')
    """
    root_dir = os.getcwd()
    print(f'Current location {root_dir}')
    try:
        print(f'Changing to {root_dir}/{directory} directory ')
        os.chdir(f'{root_dir}/{directory}')
        pdfs = cwd_files_type_search('.pdf')
        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)
        
        merger.write(output)
        merger.close()
        print(f'{output} file written successfully')
        os.chdir(root_dir)
    except:
        print('Either wrong directory or merging operation failed')
        os.chdir(root_dir)

    print('Back to root directory ', os.getcwd())


def patch_one_frame(img, patches):
    """
        dset_waxs[0] = patch_one_frame(dset_waxs[0], patches)
    """
    for args in patches:   
        if type(args[1]) == int: 
            orig, radius = args
            ### Syntax: cv2.circle(image, center_coordinates, radius, color, thickness/(-1 = fill by the color black=0)
            img = cv2.circle(img, tuple(orig), radius, 0, -1)
        elif type(args[1]) == list:    
            starting_point, ending_point = args
            ### Syntax: cv2.rectangle(image, starting_point, ending_point, color, thickness)
            img = cv2.rectangle(img, tuple(starting_point), tuple(ending_point), 0, -1)
    return img

def mica_patching(file, frame, patches, qgrid ):
    """
        mica_patching(file, frame, patches, qgrid2)
    """
    ## semi-specs
    valid_range_min, valid_range_max = (0,10)
    scattering = '_WAXS2'
    masked_file = f'{h5_top_group(file)}_masked_{frame}.h5'

    ## computation
    with h5py.File(file,'r') as hdf:
        dset = hdf.get(f'{masked_file.split("_masked")[0]}/primary/data')
        dset_saxs = np.expand_dims(dset['pil1M_image'][frame], axis=0)
        dset_waxs = np.expand_dims(dset['pilW2_image'][frame], axis=0)
        print(f'frame information extraction completes with _SAXS shape {dset_saxs.shape} _WAXS shape {dset_waxs.shape}...')

    # create temporary file
    if os.path.isfile(masked_file):
        os.remove(masked_file)
    
    # overwriting patching information
    with h5py.File(masked_file,'w') as hdf:
        
        # patching for one frame
        dset_waxs[0] = patch_one_frame(dset_waxs[0], patches)
        
        # save patched image
        dset = hdf.create_group(f'/{masked_file.split("_masked")[0]}/primary/data')
        dset.create_dataset('pil1M_image', data = dset_saxs, compression="lzf")
        dset.create_dataset('pilW2_image', data = dset_waxs, compression="lzf")

    ## Lin Yang's Code for 1-D averaging        
    azimuthal_averaging(masked_file, qgrid, n_proc=1)

    ### Plot image after patches
    img = np.clip(dset_waxs[0], valid_range_min, valid_range_max, dtype = np.int8)

    ### plot WAXS data
    with h5py.File(masked_file,'r') as hdf:
        dset_merged = hdf.get(f'{masked_file.split("_masked")[0]}/processed')
        dset_merged = dset_merged[scattering][:]

    f, axs = plt.subplots(1,2, figsize = (10,5), num=f'{masked_file} {scattering} data')
    axs[0].imshow(img, cmap = 'rainbow')
    axs[1].set_xlabel("$q (\AA^{-1})$",); 
    axs[1].set_ylabel("$I$",);
    axs[1].set_xscale('linear')
    axs[1].set_yscale('linear')
    #axs[1].errorbar(qgrid2, dset_merged[0][0], dset_merged[0][1], label=f'{scattering} {frame}')   # here dataset has only one frame
    axs[1].plot(qgrid, dset_merged[0][0], )   # here dataset has only one frame
    plt.show()

    # delete temporary file
    os.remove(masked_file)


### circular averaging after patching
def circ_avg_from_patches(source_file, qgrid, patches):
    """
        masked_file = circ_avg_from_patches(source_file, qgrid, patches)
    """
    ## specs
    masked_file = f'{h5_top_group(source_file)}_masked.h5'

    ## computation
    if os.path.isfile(masked_file):
        print(f'DID not PATCH {masked_file} already exists - did not copy, ')
        print(f'1-D Averaging is NOT performed as {masked_file} already exist ')
        print(f'TO do patching on {masked_file} delete the file first manually and run this cell again')


    else:
        print(f'{masked_file} is being created in {os.getcwd()} ...')
        shutil.copy2(source_file, masked_file)       # create a new file
        print(f'{masked_file} copy is done')

        # read _WAXS2 source data, replace masked data and save masked out data on _WAXS2
        with h5py.File(masked_file,'r+') as hdf:
            try:
                dset = hdf.get(f'{h5_top_group(masked_file)}')
                del dset['processed']
                #del dset["_SAXS"], dset["_WAXS2"], dset["merged"]
            except:
                pass

            tic = time.perf_counter();     
            print(f'{masked_file} Loading data into a numpy array started ')
            dset = dset.get(f'/{h5_top_group(masked_file)}/primary/data')
            dset_waxs = np.array(dset.get('pilW2_image'))        # np.array(dset.get('pilW2_image')), dset.get('pilW2_image')[...] 
            del dset['pilW2_image']
            print(f'{masked_file} Loading data into a numpy array finished in {time.perf_counter()-tic} seconds')


            # Loop over the data to mask out the patches
            print(f'{masked_file} Patching Started ')
            for frame in range(len(dset_waxs)):
                dset_waxs[frame] = patch_one_frame(dset_waxs[frame], patches)
            print(f'{masked_file} Patching Finished')

            tic = time.perf_counter();     
            print(f'{masked_file} patched pilW2_image dataset creation staring ... ')
            # 'lzf' (no compression_opts), chunks=(1,1043,981), compression_opts=1
            dset.create_dataset('pilW2_image', data = dset_waxs, compression="gzip", compression_opts=1)  
            print(f'{masked_file} patched pilW2_image dataset creation finished in {time.perf_counter()-tic} seconds')


        ## create 1d data from lin Yang's code (py4xs packages are used here)
        azimuthal_averaging(masked_file, qgrid, n_proc=8)

        ## setting patch attributes on the processed folder
        set_patch_attributes(masked_file, patches)

        return masked_file