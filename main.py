
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
from classes.gui_functions import MainWindow


# add dynamic saftey alpha radius box
# figure out weird crashes
# address stuttering effect
# add closed loop

 


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
