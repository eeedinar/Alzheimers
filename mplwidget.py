# ------------------------------------------------------
# --------------------- mplwidget.py --------------------
# --------------------------------------------------------------------
import  PyQt5.QtWidgets  as qtw
from  matplotlib.backends.backend_qt5agg  import  FigureCanvas
import  matplotlib.figure

    
class  MplWidget ( qtw.QWidget ):
    
    def  __init__ ( self ,  parent  =  None ):

        qtw.QWidget.__init__ ( self ,  parent )                             # initializing QWidget
        
        self.canvas  =  FigureCanvas ( matplotlib.figure.Figure ())         # create canvas
        
        vertical_layout  =  qtw.QVBoxLayout ()                              # define layout
        vertical_layout.addWidget ( self.canvas )                           # add widget to the layout
        
        self.canvas.axes  =  self.canvas.figure.subplots(nrows=2, ncols=4)  # add subplots
        self.canvas.figure.tight_layout()
        self.setLayout ( vertical_layout )                                  # set layout