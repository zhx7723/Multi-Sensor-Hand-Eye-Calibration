import cv2
from PyQt5 import QtGui, QtCore

class CameraModule:
    def __init__(self, label):
        self.label = label
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.label.setPixmap(pixmap)

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
