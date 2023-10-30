# ---------------------- main.py ---------------------
from  PyQt5.QtWidgets  import *
from  PyQt5.uic  import  loadUi
from  matplotlib.backends.backend_qt5agg  import  ( NavigationToolbar2QT  as  NavigationToolbar )

import  numpy  as  np
import  random
from essential_func import *

### configure
qgrid2                     = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
samples_csv                = 'data_directory.csv'
default_sample_dir         = 'Test'       # 'July-2021-Sample#6'    '21-Nov'   'July-sorted'   '20-Dec'    '22-Oct'   'Oct-2022-1971'   'Mar-2023-Sample#1898'
csv_and_code_abs_directory = '/Users/bashit.a/Documents/Alzheimer/Codes/'

class  MatplotlibWidget ( QMainWindow ):

    def  __init__ ( self ):
        ## initialization
        QMainWindow.__init__ ( self )                                                 # initiate main window
        loadUi ( "form.ui" , self )                                                   # load UI
        self.setWindowTitle ( "PyQt5 & Matplotlib GUI" )                              # setting window title

        ## click button matplotlib connection
        # self.pushButton.clicked.connect ( self.update_graph    )                           # connect pushbutton action

        self.MplWidget.canvas.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.addToolBar ( NavigationToolbar ( self.MplWidget.canvas ,  self ))               # add matpltlib toolbar at the top

        self.doubleSpinBoxFixedQValue.valueChanged.connect(self.plot_heatmap)
        self.doubleSpinBoxTransQValue.valueChanged.connect(self.plot_heatmap)
  
        self.doubleSpinBoxMf.valueChanged.connect(self.modify_mf)  
        self.pushButtonDelFrame.clicked.connect(self.delete_frame)
        self.pushButtonClearFrame.clicked.connect(self.clear_frames)

        self.df = pd.read_csv(os.path.join(csv_and_code_abs_directory, samples_csv))
        for item, dir_ in zip(self.df["dropdown-name"], self.df['bnl-scan-sample-dir']):
            self.comboBoxDir.addItem(item, cwd_files_search_with('.h5', directory = dir_) )   # comboBox([])
        self.comboBoxDir.activated.connect(self.update_file)

        self.comboBoxFixed.addItems(['Silver_Stained','Avg_q', 'Point_q', 'Fractal'])
        self.comboBoxFixed.setCurrentIndex(0)
        self.comboBoxTransparent.addItems(['Silver_Stained','Avg_q', 'Point_q', 'Fractal'])
        self.comboBoxTransparent.setCurrentIndex(1)
        self.comboBoxFixedTemplate.addItems(cmap_list())
        self.comboBoxFixedTemplate.setCurrentIndex(21)
        self.comboBoxTransTemplate.addItems(cmap_list())
        self.comboBoxTransTemplate.setCurrentIndex(21)

        self.pushButtonGenHeatmap.clicked.connect(self.plot_fixed_map)
        self.doubleSpinBoxTransparency.valueChanged.connect(self.plot_heatmap)

        self.spinBoxHistBin.valueChanged.connect(self.plot_histogram)
        
        self.tableWidgetFrames.setColumnCount(4)
        self.tableWidgetFrames.setHorizontalHeaderLabels(('Dir', 'File', 'Frame','mf'))

    def update_file(self, index):
        self.comboBoxFile.clear()
        self.comboBoxFile.addItems(self.comboBoxDir.itemData(index))

        self.directory = self.df['bnl-scan-sample-dir'][index]
        os.chdir(self.directory)

    def file_initialization(self, file):

        self.file                       = file
        self.Width, self.Height         = width_height(self.file, directory=self.directory) 
        self.frame_cor                  = snaking(self.Width, self.Height)                            # snaking indices for heat map, 
        self.numrows, self.numcols      = self.Height, self.Width                                     # format_coord function requires this global variables

        self.Iq_new                     = read_Iq(self.file, 'merged', directory=self.directory)      # data to plot data        
        if self.spinBoxWindowSize.value() > 1:
            window = np.ones(self.spinBoxWindowSize.value() )/self.spinBoxWindowSize.value()                                       # window_size = 4 --> 0.25,0.25,0.25,0.225
            self.Iq_new = np.array([np.convolve(window, self.Iq_new[idx], mode='same') for idx in range(self.Iq_new.shape[0])  ])  # filter output using convolution

    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Error!")
        msg.setText(f'File may not exist!')
        msg.setIcon(QMessageBox.Information)  # Information Critical Warining Question
        msg.exec_()

    def plot_fixed_map(self):
        self.plot_heatmap()
        self.plot_histogram()


    def plot_heatmap(self):

        try: 
            self.file_initialization(self.comboBoxFile.currentText())
            self.MplWidget.canvas.axes[0,0].clear()
        except:
            self.show_popup()
        else:

            def format_coord(x, y):
                col = int(x)                                  # truncate x values
                row = int(y)                                  # truncate y values
                if 0 <= col < self.numcols and 0 <= row < self.numrows:
                    z = np.flipud(self.frame_cor)[row, col]        # flipping to get correct value of z
                    return 'x=%1.2f, y=%1.2f, FRAME=%d' % (x, y, z)
                else:
                    return 'x=%1.2f, y=%1.2f' % (x, y)        # outside the plotting range, no need


            if self.comboBoxFixed.currentText() == 'Silver_Stained' or self.comboBoxTransparent.currentText() == 'Silver_Stained':
                if os.path.isfile(os.path.join(self.directory, "silver_staining", h5_top_group(self.file)+"_label.png")):
                    stained_img = f'silver_staining/{h5_top_group(self.file)}_label.png'
                    self.img_silver = mpimg.imread(stained_img)
                    self.MplWidget.canvas.axes[0,0].imshow(self.img_silver,  interpolation = 'none', origin='lower', extent=[0, self.Width, 0, self.Height], aspect='equal', norm=None, \
                        alpha = None if self.comboBoxFixed.currentText() == 'Silver_Stained' else self.doubleSpinBoxTransparency.value())
            

            if self.comboBoxFixed.currentText() == 'Point_q':
                diff_patterns = find_rep_value(qgrid2, self.Iq_new , args= self.doubleSpinBoxFixedQValue.value(), method = 'point')
                self.img_orig = snaking(self.Width, self.Height, diff_patterns)
                self.MplWidget.canvas.axes[0,0].imshow(self.img_orig, cmap = self.comboBoxFixedTemplate.currentText(), interpolation = 'none', origin='upper', extent=[0,self.Width,0,self.Height], aspect='equal', norm=None, \
                    alpha=None)
            if self.comboBoxTransparent.currentText() == 'Point_q':
                diff_patterns = find_rep_value(qgrid2, self.Iq_new , args= self.doubleSpinBoxTransQValue.value(), method = 'point')
                self.img_orig = snaking(self.Width, self.Height, diff_patterns)
                self.MplWidget.canvas.axes[0,0].imshow(self.img_orig, cmap = self.comboBoxTransTemplate.currentText(), interpolation = 'none', origin='upper', extent=[0,self.Width,0,self.Height], aspect='equal', norm=None, \
                    alpha=self.doubleSpinBoxTransparency.value())

            if self.comboBoxFixed.currentText() == 'Avg_q' or self.comboBoxTransparent.currentText() == 'Avg_q':
                diff_patterns = find_rep_value(qgrid2, self.Iq_new, method = 'circ')
                self.img_orig = snaking(self.Width, self.Height, diff_patterns)
                
                if self.comboBoxFixed.currentText() == 'Avg_q':
                    self.MplWidget.canvas.axes[0,0].imshow(self.img_orig, cmap = self.comboBoxFixedTemplate.currentText(), interpolation = 'none', origin='upper', extent=[0,self.Width,0,self.Height], aspect='equal', norm=None, \
                    alpha=None )
                elif self.comboBoxTransparent.currentText() == 'Avg_q':
                    self.MplWidget.canvas.axes[0,0].imshow(self.img_orig, cmap = self.comboBoxTransTemplate.currentText(), interpolation = 'none', origin='upper', extent=[0,self.Width,0,self.Height], aspect='equal', norm=None, \
                    alpha=self.doubleSpinBoxTransparency.value())
            
            self.MplWidget.canvas.draw()
            self.MplWidget.canvas.axes[0,0].format_coord = format_coord


    ### matplotlib click action
    def onclick(self, event):

        def get_frame(x, y):
            col  = int(x)                                        # truncate x values
            row  = int(y)                                        # truncate y values
            if 0 <= col < self.numcols and 0 <= row < self.numrows:
                return np.flipud(self.frame_cor)[row, col]       # flipping to get correct value of frame     

        ix, iy = event.xdata, event.ydata
        self.frame  = get_frame(ix, iy)

        ### plot 1d data and dimenstionality plot
        self.plot_1d()
        self.dimensionality_plot()

        ### add frames to the list widget
        self.listWidgetFrames.addItem(str(self.frame))

        ### add frames to the table widget
        rowCount = self.tableWidgetFrames.rowCount()
        self.tableWidgetFrames.insertRow(rowCount)
        self.tableWidgetFrames.setItem(rowCount, 0, QTableWidgetItem(self.directory)  )
        self.tableWidgetFrames.setItem(rowCount, 1, QTableWidgetItem(self.file)       )
        self.tableWidgetFrames.setItem(rowCount, 2, QTableWidgetItem(str(self.frame)) )
        self.tableWidgetFrames.setItem(rowCount, 3, QTableWidgetItem("1"))

    def plot_1d(self):
        self.q_min_idx, self.q_max_idx = qgrid_to_indices(qgrid2, self.doubleSpinBoxQMin.value()), qgrid_to_indices(qgrid2, self.doubleSpinBoxQMax.value())

        ### self.MplWidget.canvas.axes[0,0].set_title( 'Cosinus - Sinus Signal' )
        x  = qgrid2[self.q_min_idx : self.q_max_idx]
        y  = self.Iq_new[self.frame, self.q_min_idx : self.q_max_idx]

        self.MplWidget.canvas.axes[0,1].set(title = self.frame, xlabel = 'q' , ylabel = 'I', xscale='linear', yscale = 'linear')
        self.MplWidget.canvas.axes[0,1].plot(x, y, label = f'{self.file.split(".")[0]}=1.00*{self.frame}')
        self.MplWidget.canvas.axes[0,1].legend(fontsize=5, )
        self.MplWidget.canvas.draw()

    def modify_mf(self):

        ### get current directory and file
        file_current      = self.file
        directory_current = self.directory

        ### tablewidget current row
        row    = self.tableWidgetFrames.currentRow()
        self.directory = self.tableWidgetFrames.item(row,0).text()               # self.tableWidgetFrames.selectedItems()[0].text()
        self.file      = self.tableWidgetFrames.item(row,1).text()               # self.tableWidgetFrames.selectedItems()[1].text()
        self.frame     = self.tableWidgetFrames.item(row,2).text()               # self.tableWidgetFrames.selectedItems()[2].text()
        self.frame     = int(self.frame)
        
        
        ### set mf value
        self.file_initialization(self.file)
        mf = self.doubleSpinBoxMf.value()
        y  = mf*self.Iq_new[self.frame, self.q_min_idx : self.q_max_idx]
        self.MplWidget.canvas.axes[0,1].lines[row].set_ydata(y) 
        self.MplWidget.canvas.axes[0,1].lines[row].set_label(f'{self.file.split(".")[0]}={mf:0.2f}*{self.frame}')
        self.MplWidget.canvas.axes[0,1].legend(fontsize=5, )
        self.MplWidget.canvas.draw()

        ### table mf value is updated
        self.tableWidgetFrames.setItem(row, 3, QTableWidgetItem(f'{mf:0.2f}'))

        ### revert to previous directory and file
        self.file      = file_current
        self.directory = directory_current
        os.chdir(self.directory)
        self.file_initialization(self.file)

    def dimensionality_plot(self):

        x  = np.log(qgrid2     [             self.spinBoxDimStartQPoints.value() : self.spinBoxDimQPoints.value()])
        y  = np.log(self.Iq_new[self.frame,  self.spinBoxDimStartQPoints.value() : self.spinBoxDimQPoints.value()])
        
        coefs = np.polyfit(x,y,1)    # slope = coefs[0] intercept = coefs[1]
        z     = np.polyval(coefs, x)

        self.MplWidget.canvas.axes[1,0].set(title = self.frame, xlabel = 'log(q)' , ylabel = 'log(I)', xscale='linear', yscale = 'linear')
        self.MplWidget.canvas.axes[1,0].plot(x,y, label = f'{self.frame}')
        self.MplWidget.canvas.axes[1,0].plot(x,z, label = f'Dim={6+coefs[0]:0.2f}')
        self.MplWidget.canvas.axes[1,0].legend(fontsize=6, )
        self.MplWidget.canvas.draw()

    def plot_histogram(self):

        x  = np.log(qgrid2     [    self.spinBoxDimStartQPoints.value() : self.spinBoxDimQPoints.value()])
        y  = np.log(self.Iq_new[:,  self.spinBoxDimStartQPoints.value() : self.spinBoxDimQPoints.value()])
        
        idx_l, idx_u, _ = valid_idx_search(qgrid2, y, show_q = False)

        coefs           = np.polyfit(x[idx_l : idx_u], np.transpose(y[:,idx_l : idx_u]),1)    # slope = coefs[0] intercept = coefs[1]
        hist, bin_edges = np.histogram(coefs[0]+6, bins=self.spinBoxHistBin.value(), density=True)


        self.MplWidget.canvas.axes[1,1].clear()
        self.MplWidget.canvas.axes[1,1].stairs(hist[~np.isnan(hist)], bin_edges, fill=True)  # self.MplWidget.canvas.axes[1,1].hist(coefs[0])
        self.MplWidget.canvas.draw()

    def delete_frame(self):

        ### delete table and listwidget data
        frames = []
        items = self.tableWidgetFrames.selectedItems()      # items = self.listWidgetFrames.selectedItems()
        x = []
        for i in range(len(items)):
            frames.append(self.tableWidgetFrames.selectedItems()[i].text())
            x.append(self.tableWidgetFrames.row(self.tableWidgetFrames.selectedItems()[i] ) ) #  = self.listWidgetFrames.currentRow()  

        for clicked_idx in sorted(x, reverse=True):
            self.tableWidgetFrames.removeRow(clicked_idx)   # for list widget self.listWidgetFrames.takeItem(clicked_idx)
            self.MplWidget.canvas.axes[0,1].lines[clicked_idx].remove()   # remove plot

            self.MplWidget.canvas.axes[1,0].lines[2*clicked_idx].remove()

            self.MplWidget.canvas.axes[1,0].lines[2*clicked_idx].remove()

            self.MplWidget.canvas.axes[0,1].legend(fontsize=5, )            
            self.MplWidget.canvas.axes[1,0].legend(fontsize=6, )
            self.MplWidget.canvas.draw()
        print(frames)

    def clear_frames(self):

        self.listWidgetFrames.clear()

        while (self.tableWidgetFrames.rowCount() > 0):
            self.tableWidgetFrames.removeRow(0)

        self.MplWidget.canvas.axes[0,1].clear()
        self.MplWidget.canvas.axes[1,0].clear()

        self.MplWidget.canvas.draw()

app     =  QApplication ([]) 
window  =  MatplotlibWidget () 
window.show() 
app.exec_()