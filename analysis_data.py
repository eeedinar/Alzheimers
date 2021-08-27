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
        
    return A

def file_preprocess(file, L, qgrid, q_min, q_max, normalize=True):
    """ N point moving average , Normalize with a range from q_min ~ q_max
        function call: X = file_preprocess(file = '2048_B8_masked.h5', L=4, qgrid=qgrid2, q_min=1, q_max=2)
    """
    # spec L = N-point moving average and Iq range for mean shift
    b = np.ones((1,L))/L    # numerator co-effs of filter transfer function
    a = np.ones(1)          # denominator co-effs of filter transfer function

    Iq = read_Iq(file, 'merged')
    Iq = signal.fftconvolve(Iq,b,mode='same',) if L>1 else Iq   # filter output using convolution
    
    Iq = Iq/np.max(Iq,axis=1).reshape(-1,1) if normalize else Iq      # test how normalizing affecing data points print(Iq[0][:10])

    
    q_min_idx, q_max_idx = qgrid_to_indices(qgrid, q_min), qgrid_to_indices(qgrid, q_max)    
    X = Iq[:,q_min_idx:q_max_idx]                                                           
    return X

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

    def __init__(self, file, qgrid):

        """constructor variables: 
            self.n_patterns, self.n_qgrid, self.Iq_trans, self.Iq_AF, self.qgrid
        """
        self.qgrid = qgrid
        self.file = file
        # read Iq file
        self.Iq_AF = read_Iq(file, scattering='merged')
        self.n_patterns, self.n_qgrid = self.Iq_AF.shape

        # read transvalue
        with h5py.File(file,'r') as hdf:
            Iq_trans = hdf.get(f'{h5_top_group(file)}/primary/data')          # Iq = hdf.get('2048_B16/primary/data')
            self.Iq_trans = np.array(Iq_trans.get('em2_sum_all_mean_value'))       # Iq = np.array(2048_B16/em2_sum_all_mean_value')


    def bkg_sub(self, bkg_frame):

        """
            returns background subtracted intensity IqBS = bkg_sub(bkg_frame = 2223)
            IqBS shape ex. (3721,690)
        """
        self.bkg_frame = bkg_frame
        Iq_BK = np.broadcast_to(self.Iq_AF[bkg_frame], (self.n_patterns, self.n_qgrid))          # broadcast to (3721,690)

        # background correction calculations
        normalized_beam_intensity = (self.Iq_trans/self.Iq_trans[bkg_frame]).reshape(-1,1)                  # (3721,1) each frame intnsity is normalized by background intensity
        self.IqBS  = self.Iq_AF - Iq_BK*np.broadcast_to(normalized_beam_intensity, (self.n_patterns,self.n_qgrid))      # (3721,690) - (3721,690)*(3721,690) --> normalized_beam_intensity broadcasted from (3721,1) to (3721,690)

        return self.IqBS

    def scaling_frame(self, input_fr, tissue_fr, mf_Qindices, area_minQ= 1.3, area_maxQ = 1.42, seek_mf = (-8,8,0.1), window_size =4, show_result=False):

        ### will be used for plotting
        self.input_fr  = input_fr
        self.tissue_fr = tissue_fr
        self.area_minQ = area_minQ
        self.area_maxQ = area_maxQ

        ### extract scaling region intensities
        self.scaling_for = self.IqBS[input_fr]                                    # background subtracted amyloid
        self.scaling_by  = self.IqBS[tissue_fr]                                   # background subtracted tissue
        
        ### moving average
        window  = np.ones(window_size)/window_size                               # window_size = 4 --> 0.25,0.25,0.25,0.225
        self.scaling_for = np.convolve(window, self.scaling_for, 'same')                   # outputs same length output and will be used for area and plotting
        self.scaling_by  = np.convolve(window, self.scaling_by,  'same')                   # outputs same length output and will be used for area and plotting
        
        ### create multiplication factor search region (idx_start, idx_end)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, mf_Qindices[0]) , qgrid_to_indices(self.qgrid, mf_Qindices[1])       # mf_Qindices --> scaling qgrid regions indices
        ll, ul, inc =seek_mf
        mul_factor = np.arange(ll, ul, inc)                                  # scaling factor lookup region
        
        err = {}; err['MF'] = np.zeros(len(mul_factor)); err['MSE'] = np.zeros(len(mul_factor))
        for idx, mf in enumerate(mul_factor):
            err['MF'][idx] = mf
            err['MSE'][idx] = np.mean(np.sum(np.square(self.scaling_for[idx_start:idx_end] - mf*self.scaling_by[idx_start:idx_end])));   # mean square error for scaling
        
        minIdx = err['MSE'].argmin()                                         # mf is determined by minimum MSE
        self.mf = round(err['MF'][minIdx],4)
        
        ### find area - composite trapezoidal rule in ROI - (area_minQ, area_maxQ)
        idx_start, idx_end = qgrid_to_indices(self.qgrid, self.area_minQ) , qgrid_to_indices(self.qgrid, self.area_maxQ)        # area_minQ => (290,310) 
        #idx  = np.where((self.qgrid[idx_start:idx_end]>area_minQ) & (self.qgrid[idx_start:idx_end]<area_maxQ))   
        self.area = np.trapz(y=self.scaling_for[idx_start:idx_end] - self.mf*self.scaling_by[idx_start:idx_end], x=self.qgrid[idx_start:idx_end])
        
        if show_result:
            print('MF = ', self.mf, f'AREA: {round(self.qgrid[idx_start],3)}~{round(self.qgrid[idx_end],3)} = ', self.area, 'MSE = ', err['MSE'][minIdx])  # show only if plot

        self.IqBsTs = self.scaling_for - self.mf*self.scaling_by            # background subtracted and tissue subtracted data

        return self.mf, self.area

    def plot(self, ax=None, plot_minQ=1.20, plot_maxQ=1.6):
                
        idx_start, idx_end = qgrid_to_indices(self.qgrid, plot_minQ) , qgrid_to_indices(self.qgrid, plot_maxQ)        # plot_Q => (290,310)

        if ax==None: 
            
            f,axs = plt.subplots(nrows=2, ncols=2, num=self.input_fr, figsize=(12,7))
            ax = axs[0,0]

            ### plot IqBS 1-D data
            axs[0,1].plot(self.qgrid, np.log(self.IqBS[self.input_fr]))
            axs[0,1].set_title(f'Background Subtracted 1-D {self.input_fr}')

            ### plot SAXS and WAXS images
            waxs_diff_image(self.file, self.input_fr, f=f, ax=axs[1,0])
            saxs_diff_image(self.file, self.input_fr, f=f, ax=axs[1,1])

        ax.plot(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end] - self.mf*self.scaling_by[idx_start:idx_end], \
            label='Fr(' + str(self.input_fr) + ') - Fr(' + str(self.bkg_frame) + ") - " + str(round(self.mf,2)) + "*(" + 'Fr(' +str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.plot(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end],    label=            'Fr(' + str(self.input_fr)  + ') - Fr(' + str(self.bkg_frame) + ")")
        ax.plot(self.qgrid[idx_start:idx_end], self.mf*self.scaling_by[idx_start:idx_end], label= str(self.mf) + '*(Fr(' + str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.fill_between(self.qgrid[idx_start:idx_end], self.scaling_for[idx_start:idx_end], self.mf*self.scaling_by[idx_start:idx_end], \
            where = [(x>self.area_minQ) and (x<self.area_maxQ) for x in self.qgrid[idx_start:idx_end]], color='green', alpha=0.3)
        ax.legend(prop={'size': 6})
        ax.set_title(f'MF = {self.mf}, Area = {self.area}', fontsize=6)
        
        plt.suptitle(f'Frame - {self.input_fr}')
        plt.tight_layout()
        mplcursors.cursor()


