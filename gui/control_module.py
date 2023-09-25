from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox


class ControlModule:
    def __init__(self, main_window_ui):
        self.ui = main_window_ui
        self.ui.pushButton_camera_config.clicked.connect(self.handle_config_button)
        self.ui.pushButton_start.clicked.connect(self.handle_start_button)
        self.ui.pushButton_stop.clicked.connect(self.handle_stop_button)
        self.ui.pushButton_continue.clicked.connect(self.handle_continue_button)

        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(False)

    def handle_config_button(self):
        self.ui.label_state.setText("Configuration Complete")
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(False)
        QMessageBox.information(self.ui.pushButton_camera_config, "Configuration", "This is config page")

    def handle_start_button(self):
        # check lineEdit_position
        text_value = self.ui.lineEdit_position.text().strip()  # Use strip() to remove possible leading and trailing spaces

        if not text_value:
            QMessageBox.warning(self.ui.pushButton_start, "Warning", "Please enter a number in the Position Number text box!")
            return

        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(True)
        self.ui.pushButton_camera_config.setEnabled(False)

    def handle_stop_button(self):
        self.ui.label_state.setText("Calibration Stop")
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(True)
        self.ui.pushButton_camera_config.setEnabled(True)

    def handle_continue_button(self):
        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_stop.setEnabled(True)
        self.ui.pushButton_continue.setEnabled(False)
        self.ui.pushButton_camera_config.setEnabled(False)
