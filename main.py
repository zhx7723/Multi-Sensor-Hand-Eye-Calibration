from PyQt5 import QtWidgets
from main_window import Ui_MainWindow
from control_module import ControlModule
from PyQt5.QtWidgets import QMessageBox



class MainWindowApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()

    def initUI(self):
        self.state_controller = ControlModule(self.ui)
        self.ui.pushButton_clear.clicked.connect(self.clear_result)

    def closeEvent(self, event):
        self.state_controller.stop_camera()
        event.accept()

    def clear_result(self):
        reply = QMessageBox.question(self, 'Clear', 'Are you sure to clear all the results？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.ui.textBrowser_result.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindowApp()
    window.show()
    app.exec_()
