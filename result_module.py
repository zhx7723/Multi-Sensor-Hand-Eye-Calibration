from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

class ResultModule:
    def __init__(self, ui):
        self.ui = ui
        self.text_browser = self.ui.textBrowser_result
        self.save_button = self.ui.pushButton_save
        self.save_button.clicked.connect(self.save_result_to_file)

    def update_result(self, message):
        current_text = self.text_browser.toPlainText()
        updated_text = current_text + "\n" + message
        self.text_browser.setPlainText(updated_text)

    def save_result_to_file(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Save Results", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'w') as file:
                file.write(self.text_browser.toPlainText())
