#!/usr/bin/env python3
"""
Author: Abdullah Al Bashit
Ph.D. Student in Electrical Engineering
Northeastern University, Boston, MA
Date: 04/18/2021
"""
########################### ---------- Analysis Class ---------- ###########################

## import packages
from essential_func import *
import mplcursors


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
        
        ### create multiplication factor search region (idxSearchStart, idxSearchEnd)
        idxSearchStart, idxSearchEnd = qgrid_to_indices(self.qgrid, mf_Qindices[0]) , qgrid_to_indices(self.qgrid, mf_Qindices[1])       # mf_Qindices --> scaling qgrid regions indices
        ll, ul, inc =seek_mf
        mul_factor = np.arange(ll, ul, inc)                                  # scaling factor lookup region
        
        err = {}; err['MF'] = np.zeros(len(mul_factor)); err['MSE'] = np.zeros(len(mul_factor))
        for idx, mf in enumerate(mul_factor):
            err['MF'][idx] = mf
            err['MSE'][idx] = np.mean(np.sum(np.square(self.scaling_for[idxSearchStart:idxSearchEnd] - mf*self.scaling_by[idxSearchStart:idxSearchEnd])));   # mean square error for scaling
        
        minIdx = err['MSE'].argmin()                                         # mf is determined by minimum MSE
        self.mf = round(err['MF'][minIdx],4)
        
        ### find area - composite trapezoidal rule in ROI - (area_minQ, area_maxQ)
        idxSearchStart, idxSearchEnd = qgrid_to_indices(self.qgrid, self.area_minQ) , qgrid_to_indices(self.qgrid, self.area_maxQ)        # area_minQ => (290,310) 
        #idx  = np.where((self.qgrid[idxSearchStart:idxSearchEnd]>area_minQ) & (self.qgrid[idxSearchStart:idxSearchEnd]<area_maxQ))   
        self.area = np.trapz(y=self.scaling_for[idxSearchStart:idxSearchEnd] - self.mf*self.scaling_by[idxSearchStart:idxSearchEnd], x=self.qgrid[idxSearchStart:idxSearchEnd])
        
        if show_result:
            print('MF = ', self.mf, f'AREA: {round(self.qgrid[idxSearchStart],3)}~{round(self.qgrid[idxSearchEnd],3)} = ', self.area, 'MSE = ', err['MSE'][minIdx])  # show only if plot

        self.IqBsTs = self.scaling_for - self.mf*self.scaling_by            # background subtracted and tissue subtracted data

        return self.mf, self.area

    def plot(self, ax=None, plot_minQ=1.20, plot_maxQ=1.6):
                
        idxSearchStart, idxSearchEnd = qgrid_to_indices(self.qgrid, plot_minQ) , qgrid_to_indices(self.qgrid, plot_maxQ)        # plot_Q => (290,310)

        if ax==None: 
            
            f,axs = plt.subplots(nrows=2, ncols=2, num=self.input_fr, figsize=(12,7))
            ax = axs[0,0]

            ### plot IqBS 1-D data
            axs[0,1].plot(self.qgrid, np.log(self.IqBS[self.input_fr]))
            axs[0,1].set_title(f'Background Subtracted 1-D {self.input_fr}')

            ### plot SAXS and WAXS images
            waxs_diff_image(self.file, self.input_fr, f=f, ax=axs[1,0])
            saxs_diff_image(self.file, self.input_fr, f=f, ax=axs[1,1])

        ax.plot(self.qgrid[idxSearchStart:idxSearchEnd], self.scaling_for[idxSearchStart:idxSearchEnd] - self.mf*self.scaling_by[idxSearchStart:idxSearchEnd], \
            label='Fr(' + str(self.input_fr) + ') - Fr(' + str(self.bkg_frame) + ") - " + str(round(self.mf,2)) + "*(" + 'Fr(' +str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.plot(self.qgrid[idxSearchStart:idxSearchEnd], self.scaling_for[idxSearchStart:idxSearchEnd],    label=            'Fr(' + str(self.input_fr)  + ') - Fr(' + str(self.bkg_frame) + ")")
        ax.plot(self.qgrid[idxSearchStart:idxSearchEnd], self.mf*self.scaling_by[idxSearchStart:idxSearchEnd], label= str(self.mf) + '*(Fr(' + str(self.tissue_fr) + ') - Fr(' + str(self.bkg_frame) + "))")
        ax.fill_between(self.qgrid[idxSearchStart:idxSearchEnd], self.scaling_for[idxSearchStart:idxSearchEnd], self.mf*self.scaling_by[idxSearchStart:idxSearchEnd], \
            where = [(x>self.area_minQ) and (x<self.area_maxQ) for x in self.qgrid[idxSearchStart:idxSearchEnd]], color='green', alpha=0.3)
        ax.legend(prop={'size': 6})
        ax.set_title(f'MF = {self.mf}, Area = {self.area}', fontsize=6)
        
        plt.suptitle(f'Frame - {self.input_fr}')
        plt.tight_layout()
        mplcursors.cursor()





