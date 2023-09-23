from PyQt5 import QtWidgets
from calibration_main import Ui_MainWindow
from camera_module import CameraModule


class MainWindowApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.camera_module = CameraModule(self.ui.label_camera)

    def closeEvent(self, event):
        self.camera_module.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindowApp()
    window.show()
    app.exec_()
