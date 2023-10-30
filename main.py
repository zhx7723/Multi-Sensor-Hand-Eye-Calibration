from PyQt5 import QtWidgets
from main_window import Ui_MainWindow
from camera_module import CameraModule
from result_module import ResultModule
from control_module import ControlModule




class MainWindowApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()

    def initUI(self):
        self.state_controller = ControlModule(self.ui)


    def closeEvent(self, event):
        self.state_controller.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindowApp()
    window.show()
    app.exec_()
