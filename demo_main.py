import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class MentalHealthChatbotApp(QMainWindow):
    def __init__(self):
        super(MentalHealthChatbotApp, self).__init__()

        loadUi('MentalHealthChatbotApp.ui', self)

''' ------------------------ MAIN Function ------------------------- '''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MentalHealthChatbotApp()
    window.show()
    app.exec_()
