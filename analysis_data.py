#!/usr/bin/env python3
"""
Author: Abdullah Al Bashit
Ph.D. Student in Electrical Engineering
Northeastern University, Boston, MA
Date: 04/18/2021
"""
# ---------- Analysis Class ----------

## import packages
from essential_func import *
from scipy import signal
from sklearn.manifold import TSNE
from matplotlib.animation import FuncAnimation
from IPython.display import display
from IPython.display import HTML
import mplcursors
import cv2

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

def file_preprocess(file, window_size, qgrid, q_min=None, q_max=None, normalize=False, normalize_at_q=None, directory = None):
    """ N point moving average , Normalize with a range from q_min ~ q_max
        function call: Iq = file_preprocess(file = '2048_B8_masked.h5', window_size=4, qgrid=qgrid2, q_min=1, q_max=2)
    """
    if directory == None:
        directory = os.getcwd()
    # spec window_size = N-point moving average and Iq range for mean shift
    window      = np.ones(window_size)/window_size                                    # window_size = 4 --> 0.25,0.25,0.25,0.225

    Iq = read_Iq(file, 'merged', directory = directory)
    Iq = np.array([np.convolve(window, Iq[idx], mode='same') for idx in range(Iq.shape[0])  ]) if window_size>1 else Iq   # filter output using convolution
    
    Iq = Iq/np.max(Iq,axis=1).reshape(-1,1) if normalize and normalize_at_q==None else Iq      # test how normalizing affecing data points print(Iq[0][:10])
    Iq = Iq/Iq[:,qgrid_to_indices(qgrid, qvalue=normalize_at_q)].reshape(-1,1) if normalize_at_q!=None else Iq      # test how normalizing affecing data points print(Iq[0][:10])

    if q_min!=None and q_max!=None:
        q_min_idx, q_max_idx = qgrid_to_indices(qgrid, q_min), qgrid_to_indices(qgrid, q_max)    
        Iq = Iq[:,q_min_idx:q_max_idx]                                                           
    return Iq

# label to heatmap plot for visualizing data
def plot_labels(file, labels, title=None, args = None, cmap='viridis'):
    """
        plot_labels(file, labels)
    """
    
    labels_unique = np.unique(labels)       # total labels
    n_clusters_ = len(labels_unique)        # toal clusters
    print("number of estimated clusters : %d" % n_clusters_)
    #cluster_centers = ms.cluster_centers_   # get trained cluster centers
    
    Width, Height = width_height(file)                             # width and height of the file
    sna = snaking(Width, Height)                                   # create snaking patterns
    A = np.array([np.zeros((Height,Width)),sna])                   # zero values matrix (A[0]=0) with frame numbers depth (A[1]=frames)

    for label in labels_unique:                                    # loop over all labels
        cluster_ = np.where(labels==label)[0].tolist()             # get cluster for each label
        label_   = np.repeat(label, len(cluster_))                 # create same size label for each cluster
        A = from_clusterFr_ceffs_to_matrix(A, cluster = cluster_, coeffs = label_)   # return matrix A for each cluster label

    f,ax = plt.subplots(num=file) if args==None else args
    plot_heat_map_from_data(A[0], Width, Height, args = (f,ax), title= title, cmap=cmap)

### 3d-plot
def plot_3d(X, labels, args = None, cmap='viridis'):
    """
        plot_3d(tsne_data, labels, args = (f2,ax2), cmap = discrete_cmap( N= len(np.unique(labels)), base_cmap = 'brg'))
    """
    # plot data
    fig, _ = plt.subplots() if args ==None else args
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter3D(*zip(*X), c=labels, cmap=cmap)
    ax.set(xlabel = "Dim_1", ylabel = "Dim_2", zlabel = "Dim_3",)
    fig.colorbar(img, ax = ax, shrink=0.8, pad=0.1)
    #ax.view_init(elev=None, azim=None)

    return ax

#### Plot 3d animation
def plot_3d_animation(X, labels, elev=35, azim=60, anim_frames=200, anim_interval=10):
    
    fig, ax= plt.subplots()
    ax = plot_3d(X, labels, args = (fig, ax), cmap = discrete_cmap( N= len(np.unique(labels)), base_cmap = 'brg'))

    # things that are going to change in every frame
    def animate(frame):
        ax.view_init(elev=elev, azim=-azim-frame)  # reducing azim every frame

    anim = FuncAnimation(fig, func=animate, frames=np.arange(0,anim_frames,0.1), interval=anim_interval)   # smooth --> interval = 20, frames =100 big changes --> interval = 100
    video = anim.to_html5_video()   # visualize in jupyter-lab
    html = HTML(video)      # convert to html
    display(html)           # display HTML file
    plt.close()                     # close plot

### t-SNE
def manifold_tsne(X, perplexity, n_iter, dim=3):
    tsne_data = TSNE(n_components=dim, perplexity=perplexity, n_iter=n_iter, init='pca', learning_rate=200, random_state=0, n_jobs=-1).fit_transform(X)
    return tsne_data

### Sorting labels by the total lenth of each categories
def sort_labels(labels):
    """
    labels = np.array([1,0,2,1,1,2,3,3,3])
               total_labels  new_labels
        0             1           3
        2             2           2
        1             3           1
        3             3           0
        [1 0 2 1 1 2 3 3 3]
        [1 3 2 1 1 2 0 0 0]
    """
    idx = []
    total_labels = []
    for i in np.unique(labels):
        idx.append(i)
        total_labels.append(len(np.where(labels==i)[0]))

    df = pd.DataFrame(total_labels, columns=['total_labels'], index = idx)
    df.sort_values(by='total_labels', ascending=True, inplace=True)
    df['new_labels'] = np.arange(len(np.unique(labels))-1,-1,-1)

#     print(df)

    new_labels = np.zeros_like(labels)
    for i in np.unique(labels):
        if i in idx:
            new_labels[np.where(labels==i)] = df.loc[i].new_labels
#     print(labels)
#     print(new_labels)
    return new_labels

class Data_Analysis():

    def __init__(self, file, qgrid, window_size=4, q_min=None, q_max=None, normalize=False, normalize_at_q=None , directory = os.getcwd()):

        """constructor variables: 
            self.n_patterns, self.n_qgrid, self.Iq_trans, self.Iq, self.qgrid
        """
        self.qgrid = qgrid
        self.file = file
        # read Iq file
        self.Iq = file_preprocess(file, window_size, qgrid, q_min=None, q_max=None, normalize=False, normalize_at_q=None, directory = directory) # read_Iq(file, scattering='merged', directory = directory)
        self.n_patterns, self.n_qgrid = self.Iq.shape

        # read transvalue
        with h5py.File(os.path.join(directory, file),'r') as hdf:
            Iq_trans = hdf.get(f'{h5_top_group(file)}/primary/data')          # Iq = hdf.get('2048_B16/primary/data')
            self.Iq_trans = np.array(Iq_trans.get('em2_sum_all_mean_value'))       # Iq = np.array(2048_B16/em2_sum_all_mean_value')


    def bkg_sub(self, bkg_frame):

        """
            returns background subtracted intensity IqBS = bkg_sub(bkg_frame = [6000, 6001])
            IqBS shape ex. 7812,690)
        """
        if bkg_frame!=None:
            self.bkg_frame = np.array([bkg_frame])
            n_bkg_frame = len(self.bkg_frame.flatten())   # [6000,6001] - 2 frames

            Iq_bkgs = self.Iq[ self.bkg_frame.flatten() ]                       # data.Iq[ bkg_frame.flatten() ].shape --> (2, 690)

            Iq_trans_norm = np.concatenate([ (self.Iq_trans/self.Iq_trans[fr]).reshape(-1,1) for fr in self.bkg_frame.flatten() ],axis=1)  # (7812,2)
            Iq_bkgs = np.divide(np.dot(Iq_trans_norm,Iq_bkgs),n_bkg_frame)   # (7812,690)

            self.IqBS = self.Iq - Iq_bkgs
        else:
            self.IqBS = self.Iq
        return self.IqBS

    def tissue_sub(self, input_fr, tissue_fr, mf_Qindices, scale_method = 'MSE', return_alg = 'one_tissue-fr' , area_minQ= 1.0, area_maxQ = 1.45, seek_mf = (-12,12,0.1), mf_min=1.0, mf_max=2.5 , window_size =4, show_result=False):

        ### extracting method parameters
        QSearchStart, QSearchEnd = mf_Qindices  # change here/add here
        input_fr  = np.array([input_fr])
        tissue_fr = np.array([tissue_fr])       # in case tissue_fr is number not list
        
        ### will be used for plotting
        self.input_fr  = input_fr
        self.tissue_fr = np.array([tissue_fr])
        self.area_minQ = area_minQ
        self.area_maxQ = area_maxQ

        ### extract scaling region intensities
        self.scaling_for = self.IqBS[input_fr.flatten()]   
        self.scaling_by  = self.IqBS[tissue_fr.flatten()]  
        Iq_BSTS     = np.zeros_like(self.scaling_for)
        Iq_BSTF     = np.zeros_like(self.scaling_for)
        
        ### moving average
        if window_size!=None:
            window      = np.ones(window_size)/window_size                                    # window_size = 4 --> 0.25,0.25,0.25,0.225
            self.scaling_for = np.array([np.convolve(window, self.scaling_for[idx], 'same') for idx in range(self.scaling_for.shape[0])  ])             # outputs same length output and will be used for area and plotting
            self.scaling_by  = np.array([np.convolve(window, self.scaling_by[idx],  'same')  for idx in range(self.scaling_by.shape[0])   ])            # outputs same length output and will be used for area and plotting

        ### create multiplication factor search region (idx_start, idx_end)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, QSearchStart) , qgrid_to_indices(self.qgrid, QSearchEnd)      # mf_Qindices --> scaling qgrid regions indices
        ll, ul, inc = seek_mf
        mul_factor  = np.arange(ll, ul, inc)                                  # scaling factor lookup region


        ### mf calculations
        err = {};  ### declare MF variables

        for idx_fr_plaque, fr_plaque in enumerate(input_fr.flatten()):

            ### create variables
            err[fr_plaque] = {}
            err[fr_plaque]['MF']  = {}
            err[fr_plaque]['MSE'] = {}
            err[fr_plaque]['AREA'] = {}
            
            Iq_P = self.scaling_for[idx_fr_plaque]  # plque frame q~1.55 to q~1.8

            ### calculate mf
            Iq_T_temp = []  # holding mf*Iq[tissue] (2,690) for 2 tissue frames and resets for each plaques
            for idx_fr_tissue, fr_tissue in enumerate(tissue_fr.flatten()):

                # temporary mf and mse holder for each each background loop
                mf  = np.full(len(mul_factor), np.nan)         # temporary variable
                mse = np.full(len(mul_factor), np.nan)         # temporary variable

                Iq_T  = self.scaling_by[idx_fr_tissue]  # tissue frame q~1.55 to q~1.8
                
                for idx_mf, mf_temp in enumerate(mul_factor):
                    mf[idx_mf] = mf_temp
                    if scale_method == 'MSE':
                        mse[idx_mf] = np.mean(np.sum(np.square(Iq_P[idx_start:idx_end] - mf_temp*Iq_T[idx_start:idx_end])));   # mean square error for scaling

                minIdx = mse.argmin()                                         # mf is determined by minimum MSE
                err[fr_plaque]['MF'][fr_tissue]  = round( mf[minIdx],  6)
                err[fr_plaque]['MSE'][fr_tissue] = round(mse[minIdx], 10)

                ### find area - composite trapezoidal rule in ROI - (area_minQ, area_maxQ)
                idx_start_area, idx_end_area = qgrid_to_indices(self.qgrid, area_minQ) , qgrid_to_indices(self.qgrid, area_maxQ)        # area_minQ => (290,310)
                err[fr_plaque]['AREA'][fr_tissue] = np.trapz(y=Iq_P[idx_start_area:idx_end_area] - err[fr_plaque]['MF'][fr_tissue]*Iq_T[idx_start_area:idx_end_area], x=self.qgrid[idx_start_area:idx_end_area])
                err[fr_plaque]['AREA'][fr_tissue] = round(err[fr_plaque]['AREA'][fr_tissue],10)  # rounding

                if show_result:
                    print('plaque-fr :', fr_plaque, ' background-fr :', fr_tissue, \
                          '| MF = ', err[fr_plaque]['MF'][fr_tissue], \
                          f'AREA: {round(self.qgrid[idx_start_area],3)}~{round(self.qgrid[idx_end_area],3)} = ', err[fr_plaque]['AREA'][fr_tissue])  # show only if plot
                
                ### get Iq_BSTF and Iq_BSTS given that mf values are limited    
                if err[fr_plaque]['MF'][fr_tissue] < mf_max and err[fr_plaque]['MF'][fr_tissue] > mf_min:
                    Iq_T_temp.append(err[fr_plaque]['MF'][fr_tissue]*Iq_T)
                    if show_result:
                        print('>> plaque-fr :', fr_plaque, ' << Tissue Subtraction only counts >>',  ' background-fr :' , fr_tissue,  'MF = ' , err[fr_plaque]['MF'][fr_tissue])
            
            Iq_BSTF[idx_fr_plaque] = np.mean(np.array(Iq_T_temp), axis=0)     # background subtracted tissue frames intensity average
            Iq_BSTS[idx_fr_plaque] = Iq_P - Iq_BSTF[idx_fr_plaque]            # background subtracted intensity minus background subtracted tissue subtracted intensities

        if return_alg == 'one_tissue-fr':
            return err[fr_plaque]['MF'][fr_tissue], err[fr_plaque]['AREA'][fr_tissue], Iq_BSTF, Iq_BSTS
        elif return_alg == 'multi_tissue-fr':
            return Iq_BSTF, Iq_BSTS

    def scaling_frame(self, input_fr, tissue_fr, mf_Qindices, method = 'NEG', area_minQ= 1.3, area_maxQ = 1.42, seek_mf = (-8,8,0.1), window_size =4, show_result=False):

        ### will be used for plotting
        self.input_fr  = input_fr
        self.tissue_fr = np.array([tissue_fr]) # in case tissue_fr is number not list
        self.area_minQ = area_minQ
        self.area_maxQ = area_maxQ

        ### extract scaling region intensities
        self.scaling_for = self.IqBS[self.input_fr]                                  # background subtracted amyloid
        self.scaling_by  = np.mean(self.IqBS[self.tissue_fr,:], axis=1).flatten() if len(self.tissue_fr.shape) > 1 else  self.IqBS[self.tissue_fr].flatten()   # self.IqBS[self.tissue_fr,:].shape = (1,5,690)
        #print(self.scaling_for.shape, self.scaling_by.shape)
        
        ### moving average
        window  = np.ones(window_size)/window_size                                # window_size = 4 --> 0.25,0.25,0.25,0.225
        self.scaling_for = np.convolve(window, self.scaling_for, 'same')                   # outputs same length output and will be used for area and plotting
        self.scaling_by  = np.convolve(window, self.scaling_by,  'same')                   # outputs same length output and will be used for area and plotting

        ### create multiplication factor search region (idx_start, idx_end)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, mf_Qindices[0]) , qgrid_to_indices(self.qgrid, mf_Qindices[1])       # mf_Qindices --> scaling qgrid regions indices
        ll, ul, inc =seek_mf
        mul_factor = np.arange(ll, ul, inc)                                  # scaling factor lookup region

        err = {}; err['MF'] = np.zeros(len(mul_factor)); err['MSE'] = np.zeros(len(mul_factor)); err['NEG'] = np.zeros((len(mul_factor), idx_end-idx_start));
        for idx, mf in enumerate(mul_factor):
            err['MF'][idx] = mf
            if method == 'NEG':
                err['NEG'][idx] = self.scaling_for[idx_start:idx_end] - mf*self.scaling_by[idx_start:idx_end]
            elif method == 'MSE':
                err['MSE'][idx] = np.mean(np.sum(np.square(self.scaling_for[idx_start:idx_end] - mf*self.scaling_by[idx_start:idx_end])));   # mean square error for scaling
        
        if method == 'NEG':
            temp = np.zeros(len(mul_factor))
            for idx, mf in enumerate(mul_factor):
                temp[idx] = (err['NEG'][idx]>0).all()                                         # mf is determined by minimum MSE
            self.mf = round(err['MF'][temp.argmin()],4)
        elif method == 'MSE':
            minIdx = err['MSE'].argmin()                                         # mf is determined by minimum MSE
            self.mf = round(err['MF'][minIdx],4)


        ### find area - composite trapezoidal rule in ROI - (area_minQ, area_maxQ)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, self.area_minQ) , qgrid_to_indices(self.qgrid, self.area_maxQ)        # area_minQ => (290,310) 
        self.area = np.trapz(y=self.scaling_for[idx_start:idx_end] - self.mf*self.scaling_by[idx_start:idx_end], x=self.qgrid[idx_start:idx_end])

        if show_result:
            print('MF = ', self.mf, f'AREA: {round(self.qgrid[idx_start],3)}~{round(self.qgrid[idx_end],3)} = ', self.area)  # show only if plot

        self.IqBsTs = self.scaling_for - self.mf*self.scaling_by            # background subtracted and tissue subtracted data

        return self.mf, self.area

    def plot(self, ax=None, plot_minQ=1.20, plot_maxQ=1.6):

        idx_start, idx_end = qgrid_to_indices(self.qgrid, plot_minQ) , qgrid_to_indices(self.qgrid, plot_maxQ)        # plot_Q => (290,310)


        ax.plot(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end] - self.mf*self.scaling_by[idx_start:idx_end], \
            label='Fr(' + str(self.input_fr) + ') - Fr(' + str(self.bkg_frame) + ") - " + str(round(self.mf,2)) + "*(" + 'Fr(' +str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.plot(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end],    label=            'Fr(' + str(self.input_fr)  + ') - Fr(' + str(self.bkg_frame) + ")")
        ax.plot(self.qgrid[idx_start:idx_end], self.mf*self.scaling_by[idx_start:idx_end], label= str(self.mf) + '*(Fr(' + str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.fill_between(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end], self.mf*self.scaling_by[idx_start:idx_end], \
            where = [(x>self.area_minQ) and (x<self.area_maxQ) for x in self.qgrid[idx_start:idx_end]], color='green', alpha=0.3)
        ax.legend(prop={'size': 5})
        ax.set_title(f'MF = {self.mf}, Area = {self.area}', fontsize=7)
        ax.grid()

        plt.suptitle(f'Frame - {self.input_fr}')
        plt.tight_layout()
        mplcursors.cursor()

## same as flatten() but faster - not memory intensive ref - https://tech.qvread.com/
def flatall(nested_object):
    #    gather stores the final flattened list
    gather = []
    for item in nested_object:
        #   will flatten lists, tuples and sets
        #   will not operate on string, dictionary
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatall(item))
        else:
            gather.append(item)
    return gather

class Snaking_frames_search:
    def __init__(self, Width, Height):
        self.Width  = Width
        self.Height = Height
        self.sna = snaking(Width, Height , np.arange(0, Width*Height))  #   [  62,   63,   64, ...,  121,  122,  123],
                                                                        #   [  61,   60,   59, ...,    2,    1,    0]])
        # print(self.sna)

        grid = np.zeros((Height,Width),dtype=object)

        for i in range(0,Height):
            for j in range(0,Width):
                grid[i,j]=(i,j)
        self.grid = grid[::-1]      # [(1, 0), (1, 1), (1, 2), ..., (1, 59), (1, 60), (1, 61)],
                                    # [(0, 0), (0, 1), (0, 2), ..., (0, 59), (0, 60), (0, 61)]],

    def frame_to_idx(self,frame):
        idx = [i for i in self.grid[self.sna==frame]]   #  [i for i in fr_idx.grid[fr_idx.sna==1]] ==> [(0, 60)]
        return idx

    def idx_to_frame(self,l):
        idx = []
        for i in l:
            a, b = i
            x = [i==(a,b) for i in self.grid[self.Height-a-1]]
            idx.append(self.sna[self.Height-a-1, np.where(x)].tolist())
        return flatall(idx)

    def frame_idx_to_kernal_frames(self, kernal_size, frame):

        ### fr_idx.frame_idx_to_kernal_frames(3,0) ==> [1, 0, 122, 123]

        kernel_indices = []
        kernal_inc = int(np.floor(kernal_size/2))
        frame_idx = self.frame_to_idx(frame)
        x, y = frame_idx[0]
        for i in np.arange(x-kernal_inc,x+kernal_inc+1):
            for j in np.arange(y-kernal_inc,y+kernal_inc+1):
                if (i,j) ==(x,y) or (i>=0 and j>=0 and i<self.Height and j <self.Width):
                    kernel_indices.append((i,j))

        return self.idx_to_frame(kernel_indices)
