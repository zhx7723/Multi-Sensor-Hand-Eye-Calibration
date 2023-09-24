from PyQt5 import QtWidgets
from calibration_main import Ui_MainWindow
from camera_module import CameraModule
from result_module import ResultModule



class MainWindowApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.camera_module = CameraModule(self.ui.label_camera)
        self.result_module = ResultModule(self.ui.textBrowser_result, self.ui.pushButton_save)

    def closeEvent(self, event):
        self.camera_module.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindowApp()
    window.show()

    # For test
    window.result_module.update_result("X: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]\n\n"
                                       "Y: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]\n\n"
                                       "Z: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]")

    app.exec_()
