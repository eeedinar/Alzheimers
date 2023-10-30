from PyQt5.QtWidgets import  QMainWindow, QApplication, QSlider, QLabel
from PyQt5 import uic, QtCore
import sys

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("form.ui", self)
        self.setWindowTitle("Slider!")

        # label properties
        self.label  = self.findChild(QLabel,  "label")
        self.label.setAlignment(QtCore.Qt.AlignCenter)                    # center label

        # Define Widgets
        self.slider = self.findChild(QSlider, "horizontalSlider")
        self.slider.setRange(-10,102)
        self.slider.setValue(12)                                          # default value
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(10)                                     # keyboard arrow steps
        self.slider.valueChanged.connect(self.slide_it)                   # Move the slider

        # Show the App
        self.show()

    def slide_it(self, value):
        self.label.setText(str(value))

# Initialize the App
app = QApplication(sys.argv)
UTWindow = UI()
app.exec_()