# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibration_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(885, 708)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_camera = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_camera.setFont(font)
        self.groupBox_camera.setObjectName("groupBox_camera")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_camera)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.graphicsView_forcamera = QtWidgets.QGraphicsView(self.groupBox_camera)
        self.graphicsView_forcamera.setObjectName("graphicsView_forcamera")
        self.horizontalLayout_4.addWidget(self.graphicsView_forcamera)
        self.horizontalLayout_8.addWidget(self.groupBox_camera)
        self.groupBox_control = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_control.setFont(font)
        self.groupBox_control.setObjectName("groupBox_control")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_control)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_camera_config = QtWidgets.QPushButton(self.groupBox_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera_config.sizePolicy().hasHeightForWidth())
        self.pushButton_camera_config.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_camera_config.setFont(font)
        self.pushButton_camera_config.setObjectName("pushButton_camera_config")
        self.horizontalLayout_9.addWidget(self.pushButton_camera_config)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_charuco = QtWidgets.QLabel(self.groupBox_control)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_charuco.setFont(font)
        self.label_charuco.setObjectName("label_charuco")
        self.horizontalLayout_7.addWidget(self.label_charuco)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.comboBox_charuco = QtWidgets.QComboBox(self.groupBox_control)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.comboBox_charuco.setFont(font)
        self.comboBox_charuco.setObjectName("comboBox_charuco")
        self.comboBox_charuco.addItem("")
        self.comboBox_charuco.addItem("")
        self.horizontalLayout_7.addWidget(self.comboBox_charuco)
        self.horizontalLayout_7.setStretch(0, 5)
        self.horizontalLayout_7.setStretch(1, 1)
        self.horizontalLayout_7.setStretch(2, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_position = QtWidgets.QLabel(self.groupBox_control)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_position.setFont(font)
        self.label_position.setObjectName("label_position")
        self.horizontalLayout_5.addWidget(self.label_position)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.lineEdit_position = QtWidgets.QLineEdit(self.groupBox_control)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit_position.setFont(font)
        self.lineEdit_position.setObjectName("lineEdit_position")
        self.horizontalLayout_5.addWidget(self.lineEdit_position)
        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.pushButton_start = QtWidgets.QPushButton(self.groupBox_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_start.sizePolicy().hasHeightForWidth())
        self.pushButton_start.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout_11.addWidget(self.pushButton_start)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.pushButton_stop = QtWidgets.QPushButton(self.groupBox_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_stop.sizePolicy().hasHeightForWidth())
        self.pushButton_stop.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_stop.setFont(font)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.horizontalLayout_10.addWidget(self.pushButton_stop)
        self.pushButton_continue = QtWidgets.QPushButton(self.groupBox_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_continue.sizePolicy().hasHeightForWidth())
        self.pushButton_continue.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(False)
        self.pushButton_continue.setFont(font)
        self.pushButton_continue.setObjectName("pushButton_continue")
        self.horizontalLayout_10.addWidget(self.pushButton_continue)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(1, 3)
        self.verticalLayout.setStretch(2, 3)
        self.verticalLayout.setStretch(3, 3)
        self.verticalLayout.setStretch(4, 3)
        self.horizontalLayout_8.addWidget(self.groupBox_control)
        self.horizontalLayout_8.setStretch(0, 5)
        self.horizontalLayout_8.setStretch(1, 4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.groupBox_progress = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_progress.setFont(font)
        self.groupBox_progress.setObjectName("groupBox_progress")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_progress)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_progress)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_3.addWidget(self.groupBox_progress)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.label_state = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_state.sizePolicy().hasHeightForWidth())
        self.label_state.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_state.setFont(font)
        self.label_state.setObjectName("label_state")
        self.horizontalLayout_3.addWidget(self.label_state)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.horizontalLayout_3.setStretch(0, 5)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 3)
        self.horizontalLayout_3.setStretch(3, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.groupBox_result = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_result.setFont(font)
        self.groupBox_result.setObjectName("groupBox_result")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_result)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textBrowser_result = QtWidgets.QTextBrowser(self.groupBox_result)
        self.textBrowser_result.setObjectName("textBrowser_result")
        self.horizontalLayout.addWidget(self.textBrowser_result)
        self.pushButton_save = QtWidgets.QPushButton(self.groupBox_result)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_save.setFont(font)
        self.pushButton_save.setObjectName("pushButton_save")
        self.horizontalLayout.addWidget(self.pushButton_save)
        self.verticalLayout_2.addWidget(self.groupBox_result)
        self.groupBox_result.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 885, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_camera.setTitle(_translate("MainWindow", "CAMERA"))
        self.groupBox_control.setTitle(_translate("MainWindow", "CONTROL"))
        self.pushButton_camera_config.setText(_translate("MainWindow", "Camera Config"))
        self.label_charuco.setText(_translate("MainWindow", "Charuco Board Type: "))
        self.comboBox_charuco.setItemText(0, _translate("MainWindow", "4×4"))
        self.comboBox_charuco.setItemText(1, _translate("MainWindow", "4×6"))
        self.label_position.setText(_translate("MainWindow", "Position number:"))
        self.pushButton_start.setText(_translate("MainWindow", "Start Calibration"))
        self.pushButton_stop.setText(_translate("MainWindow", "STOP"))
        self.pushButton_continue.setText(_translate("MainWindow", "CONTINUE"))
        self.groupBox_progress.setTitle(_translate("MainWindow", "PROGRESS"))
        self.label_state.setText(_translate("MainWindow", "PROCESSING..."))
        self.groupBox_result.setTitle(_translate("MainWindow", "RESULT"))
        self.pushButton_save.setText(_translate("MainWindow", "Save As..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
