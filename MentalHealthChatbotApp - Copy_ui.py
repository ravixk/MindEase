# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\P_R_O_J_E_C_T_S\P_R_O_J_E_C_T_S-2021-2022-2023-2024\PYTHON-2024\MENTAL-HEALTH-CHATBOT\code\MentalHealthChatbotApp - Copy.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(730, 565)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mic_pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.mic_pushButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("e:\\P_R_O_J_E_C_T_S\\P_R_O_J_E_C_T_S-2021-2022-2023-2024\\PYTHON-2024\\MENTAL-HEALTH-CHATBOT\\code\\images/microphone.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.mic_pushButton.setIcon(icon)
        self.mic_pushButton.setIconSize(QtCore.QSize(100, 100))
        self.mic_pushButton.setObjectName("mic_pushButton")
        self.verticalLayout.addWidget(self.mic_pushButton)
        self.horizontalLayout.addWidget(self.groupBox_3)
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.recognised_plainTextEdit = QtWidgets.QPlainTextEdit(self.groupBox)
        self.recognised_plainTextEdit.setObjectName("recognised_plainTextEdit")
        self.gridLayout_2.addWidget(self.recognised_plainTextEdit, 1, 0, 1, 1)
        self.notification_label = QtWidgets.QLabel(self.groupBox)
        self.notification_label.setText("")
        self.notification_label.setObjectName("notification_label")
        self.gridLayout_2.addWidget(self.notification_label, 0, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.predict_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predict_pushButton.sizePolicy().hasHeightForWidth())
        self.predict_pushButton.setSizePolicy(sizePolicy)
        self.predict_pushButton.setObjectName("predict_pushButton")
        self.verticalLayout_2.addWidget(self.predict_pushButton)
        self.horizontalLayout.addWidget(self.groupBox_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.tab)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.response_textBrowser = QtWidgets.QTextBrowser(self.groupBox_4)
        self.response_textBrowser.setOpenExternalLinks(True)
        self.response_textBrowser.setObjectName("response_textBrowser")
        self.gridLayout_3.addWidget(self.response_textBrowser, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.comboBox = QtWidgets.QComboBox(self.tab_3)
        self.comboBox.setGeometry(QtCore.QRect(10, 20, 151, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 601, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(10, 90, 131, 161))
        self.label_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(150, 90, 471, 161))
        self.label_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(180, 20, 181, 28))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_4.addWidget(self.textBrowser)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 730, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HaruMind"))
        self.label.setText(_translate("MainWindow", "HaruMind"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Speech Input"))
        self.groupBox.setTitle(_translate("MainWindow", "Recognised text"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Manual Test"))
        self.predict_pushButton.setText(_translate("MainWindow", "Predict"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Bot Response"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Chat"))
        self.comboBox.setItemText(0, _translate("MainWindow", "d1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "d2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "d3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "d4"))
        self.comboBox.setItemText(4, _translate("MainWindow", "d5"))
        self.comboBox.setItemText(5, _translate("MainWindow", "d5"))
        self.label_2.setText(_translate("MainWindow", "Doctor profile"))
        self.label_3.setText(_translate("MainWindow", "photo"))
        self.label_4.setText(_translate("MainWindow", "description"))
        self.pushButton.setText(_translate("MainWindow", "Book Appointment"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Book Appointment"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:600;\">HaruMind</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "About"))
