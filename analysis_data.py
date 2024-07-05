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
# from scipy import signal
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

def convolve_scale_slice(Iq, qgrid, window_size, q_min=None, q_max=None, normalize=False, normalize_at_q=None):
    """ N point moving average , Normalize with a range from q_min ~ q_max
        function call: Iq = convolve_scale_slice(Iq, window_size=4, qgrid=qgrid2, q_min=1, q_max=2)
    """
    assert Iq.ndim==2, "Iq dimenstion must be 2"

    if np.any(np.isnan(Iq)):
        print("X contains NaN values - Interpolating on missing data")
        _, Iq   = interpolate_missing(Iq)
    
    # spec window_size = N-point moving average and Iq range for mean shift
    window      = np.ones(window_size)/window_size                                    # window_size = 4 --> 0.25,0.25,0.25,0.225
    Iq = np.array([np.convolve(window, Iq[idx], mode='same') for idx in range(Iq.shape[0])  ]) if window_size>1 else Iq   # filter output using convolution

    # normazlization
    Iq = Iq/np.max(Iq,axis=1).reshape(-1,1) if normalize and not normalize_at_q else Iq      # test how normalizing affecing data points print(Iq[0][:10])
    Iq = Iq/Iq[:,qgrid_to_indices(qgrid, qvalue=normalize_at_q)].reshape(-1,1) if normalize_at_q else Iq      # test how normalizing affecing data points print(Iq[0][:10])

    if q_min and q_max:
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

# def Iq_scaling(Iq_input, Iq_scale, seek_mf, method = 'MSE'):
#     ll, ul, inc = seek_mf
#     mul_factor  = np.arange(ll, ul, inc)                                  # scaling factor lookup region
#     err = {}; 
#     err['MF']  = np.zeros(len(mul_factor)); 
#     err['MSE'] = np.zeros(len(mul_factor)); 
#     err['NEG'] = np.zeros((len(mul_factor), Iq_scale.shape[1]));
#     for idx, mf in enumerate(mul_factor):
#         err['MF'][idx] = mf
#         if method == 'NEG':
#             err['NEG'][idx] = Iq_input - mf*Iq_scale
#         elif method == 'MSE':
#             err['MSE'][idx] = np.mean(np.square(Iq_input - mf*Iq_scale ),axis=1);   # mean square error for scaling

#     if method == 'NEG':
#         temp = np.zeros(len(mul_factor))
#         for idx, mf in enumerate(mul_factor):
#             temp[idx] = (err['NEG'][idx]>0).all()                                         # mf is determined by minimum MSE
#         mf = round(err['MF'][temp.argmin()],4)
#     elif method == 'MSE':
#         minIdx = err['MSE'].argmin()                                         # mf is determined by minimum MSE
#         mf = round(err['MF'][minIdx],4)
#     return mf


def Iq_scaling(Iq_input, Iq_scale, seek_mf, method = 'MSE'):
    ll, ul, inc = seek_mf
    mul_factor  = np.round(np.arange(ll, ul, inc),6)                                  # scaling factor lookup region
    err = {}; 
    err['MSE'] = {}
    err['NEG'] = {}
    for mf in mul_factor:
        if method == 'NEG':
            err['NEG'][mf] = Iq_input - mf*Iq_scale
        elif method == 'MSE':
            err['MSE'][mf] = np.mean(np.square(Iq_input - mf*Iq_scale ), axis=1);   # mean square error for scaling
    mf = np.zeros((Iq_input.shape[0],1))
    for idx in range(len(mf)):
        mf[idx] = sorted(err['MSE'].items(), key= lambda x: x[1][idx])[0][0]
    return mf


class Data_Analysis():

    def __init__(self, file, qgrid, window_size=4, q_min=None, q_max=None, normalize=False, normalize_at_q=None , directory = os.getcwd()):

        """constructor variables:
            self.n_patterns, self.n_qgrid, self.Iq_trans, self.Iq, self.qgrid
        """
        self.file = file
        self.qgrid = qgrid
        self.window_size = window_size
        self.q_min = q_min
        self.q_max = q_max
        self.normalize = normalize
        self.normalize_at_q = normalize_at_q
        self.directory = directory

        # read Iq file
        self.Iq = read_Iq(file, 'merged', directory = self.directory)
        self.Iq = convolve_scale_slice(self.Iq, self.qgrid, self.window_size,  q_min=self.q_min, q_max=self.q_max, normalize=self.normalize, normalize_at_q=self.normalize_at_q)
        self.n_patterns, self.n_qgrid = self.Iq.shape
        

        # read transvalue
        with h5py.File(os.path.join(directory, file),'r') as hdf:
            Iq_trans = hdf.get(f'{h5_top_group(file)}/primary/data')          # Iq = hdf.get('2048_B16/primary/data')
            self.Iq_trans = np.array(Iq_trans.get('em2_sum_all_mean_value'))  # Iq = np.array(2048_B16/em2_sum_all_mean_value')


    def bkg_sub(self, bkg_frame, bkg_file = None, bkg_dir=None):

        """
            returns background subtracted intensity IqBS = bkg_sub(bkg_frame = [6000, 6001])
            IqBS shape ex. 7812,690)
        """
        bkg_file = self.file if not bkg_file else bkg_file
        bkg_dir  = self.directory if not bkg_dir else bkg_dir
        if bkg_frame:
            self.bkg_frame = np.array([bkg_frame])
            n_bkg_frame = len(self.bkg_frame.flatten())   # [6000,6001] - 2 frames

            if not bkg_file: 
                Iq_bkgs = self.Iq[ self.bkg_frame.flatten() ]  # data.Iq[ bkg_frame.flatten() ].shape --> (2, 690)
                Iq_trans_bkg = self.Iq_trans
            else:
                Iq_bkgs = read_Iq(bkg_file, 'merged', directory = bkg_dir)
                Iq_bkgs = convolve_scale_slice(Iq_bkgs, self.qgrid, self.window_size, q_min=self.q_min, q_max=self.q_max, normalize=self.normalize, normalize_at_q=self.normalize_at_q)
                Iq_bkgs = Iq_bkgs[ self.bkg_frame.flatten() ]
                # read transvalue for bkg file
                with h5py.File(os.path.join(bkg_dir, bkg_file),'r') as hdf:
                    Iq_trans_bkg = hdf.get(f'{h5_top_group(bkg_file)}/primary/data')          # Iq = hdf.get('2048_B16/primary/data')
                    Iq_trans_bkg = np.array(Iq_trans_bkg.get('em2_sum_all_mean_value'))       # Iq = np.array(2048_B16/em2_sum_all_mean_value')

            if not (self.Iq_trans==None).all():           # data.Iq_trans.shape = (7812) -> data.Iq_trans==None -> array([False, False, False, False]) --> (data.Iq_trans==None).all() = False [no trans case: array(None, dtype=object) ]
                Iq_trans_norm = np.concatenate([ (self.Iq_trans/Iq_trans_bkg[fr]).reshape(-1,1) for fr in self.bkg_frame.flatten() ],axis=1)  # (7812,2)
                Iq_bkgs = np.divide(np.dot(Iq_trans_norm,Iq_bkgs),n_bkg_frame)   # (7812,690)

                self.IqBS = self.Iq - Iq_bkgs
            else:
                self.IqBS = self.Iq - np.mean(Iq_bkgs, axis=0)  # np.mean(Iq_bkgs, axis=0).shape = (690,)
        else:
            self.bkg_frame = None
            self.IqBS = self.Iq
        return self.IqBS

    def tissue_sub(self, input_fr, tissue_fr, mf_Qindices=(1.55, 1.85), tissue_input = False, method = 'MSE', area_minQ= 1.3, area_maxQ = 1.42, seek_mf = (-8,8,0.1), show_result=False):

        ### will be used for plotting
        self.input_fr  = input_fr
        self.area_minQ = area_minQ
        self.area_maxQ = area_maxQ
        self.input_fr_to_index = {k:v for k,v in zip(input_fr, np.arange(0, len(input_fr)))} if not isinstance(input_fr, int) else {input_fr:0}

        ### extract scaling region intensities
        self.scaling_for = self.IqBS[self.input_fr] if isinstance(input_fr, list|tuple) else self.IqBS[[self.input_fr]]                                  # background subtracted amyloid
        if not tissue_input and (method =='MSE' or method == 'NEG'):
            self.tissue_fr = np.array(tissue_fr) if isinstance(tissue_fr, list|tuple) else np.array([tissue_fr])# in case tissue_fr is a number, not list
            self.scaling_by  = np.mean(self.IqBS[self.tissue_fr,:], axis=0, keepdims=True)  # self.IqBS[self.tissue_fr,:].shape = (1,5,690)
        elif tissue_input:
            self.tissue_fr = str('external')
            self.scaling_by  = np.array(tissue_fr, ndmin=2)
            assert self.scaling_by.shape[1] == self.scaling_for.shape[1], 'External tissue input dimensions are not same'

        ### create multiplication factor search region (idx_start, idx_end)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, mf_Qindices[0]) , qgrid_to_indices(self.qgrid, mf_Qindices[1])       # mf_Qindices --> scaling qgrid regions indices
        
        Iq_input = self.scaling_for[:, idx_start:idx_end]
        Iq_scale = self.scaling_by [:, idx_start:idx_end]
        self.mf = Iq_scaling(Iq_input, Iq_scale, seek_mf, method) # Iq_input.shape = (1,60), Iq_scale.shape = (1,60)

        ### find area - composite trapezoidal rule in ROI - (area_minQ, area_maxQ)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, self.area_minQ) , qgrid_to_indices(self.qgrid, self.area_maxQ)        # area_minQ => (290,310) 
        self.area = np.trapz(y=self.scaling_for[:, idx_start:idx_end] - self.mf*self.scaling_by[:,idx_start:idx_end], x=self.qgrid[idx_start:idx_end], axis=1)

        if show_result:
            print('MF = ', self.mf, f'AREA: {round(self.qgrid[idx_start],3)}~{round(self.qgrid[idx_end],3)} = ', self.area)  # show only if plot

        self.IqBsTs = self.scaling_for - self.mf*self.scaling_by            # background subtracted and tissue subtracted data

        return self.mf, self.area

    def plot(self, plot_fr, ax=None, plot_minQ=1.20, plot_maxQ=1.6, show_legend=True):

        idx = self.input_fr_to_index[plot_fr]

        idx_start, idx_end = qgrid_to_indices(self.qgrid, plot_minQ) , qgrid_to_indices(self.qgrid, plot_maxQ)        # plot_Q => (290,310)

        Iq_input = self.scaling_for[idx, idx_start:idx_end].flatten()
        Iq_scale = self.scaling_by [:, idx_start:idx_end].flatten()
        x = self.qgrid[idx_start:idx_end]
        mf   = self.mf[idx]
        area = self.area[idx]
        print(Iq_input.shape, Iq_scale.shape, mf.shape)

        ax.plot(x, Iq_input - mf*Iq_scale, \
            label='Fr(' + str(plot_fr) + ') - Fr(' + str(self.bkg_frame) + ") - " + str(np.round(mf,2)) + "*(" + 'Fr(' +str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.plot(x, Iq_input,    label=            'Fr(' + str(plot_fr)  + ') - Fr(' + str(self.bkg_frame) + ")")
        ax.plot(x, mf*Iq_scale, label= str(mf) + '*(Fr(' + str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.fill_between(x, Iq_input, mf*Iq_scale, \
            where = [(x>self.area_minQ) and (x<self.area_maxQ) for x in self.qgrid[idx_start:idx_end]], color='green', alpha=0.3)
        
        if show_legend:
            ax.legend(prop={'size': 5})
        ax.set_title(f'MF = {mf}, Area = {area}', fontsize=7)
        ax.grid()

        plt.suptitle(f'Frame - {plot_fr}')
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
