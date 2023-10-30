from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
import cv2
import numpy as np
import robolink

class CameraThread(QThread):
    update_frame_signal = pyqtSignal(QPixmap)

    def __init__(self, RDK, camera_name, dir_path):
        super().__init__()
        self.RDK = RDK
        self.image = None
        self.camera = RDK.Item(camera_name, robolink.ITEM_TYPE_CAMERA)
        self.dir_path = dir_path
        self.is_running = True

    def run(self):
        self.camera.setParam('Open', 1)
        while self.is_running:
            if self.camera.setParam('isOpen') == '1':
                bytes_img = self.RDK.Cam2D_Snapshot('', self.camera)
                if isinstance(bytes_img, bytes) and bytes_img != b'':
                    nparr = np.frombuffer(bytes_img, np.uint8)
                    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.update_frame_signal.emit(pixmap)
            # self.msleep(20)  # Adjust the sleep time according to your needs

    def stop(self):
        self.is_running = False


class CameraModule:
    def __init__(self, label, RDK, dir_path):
        self.label = label
        self.dir_path = dir_path
        self.camera_thread = CameraThread(RDK, 'cf_camera_rgb_324x244', dir_path)
        self.camera_thread.update_frame_signal.connect(self.update_frame)
        self.camera_thread.start()

    def update_frame(self, pixmap: QPixmap):
        self.label.setPixmap(pixmap)

    def save_image(self, i, j):
        cv2.imwrite(f"{self.dir_path}/captures/cap_{i + 1}{j + 1}.png", self.camera_thread.image)

    def stop(self):
        self.camera_thread.stop()
        self.camera_thread.wait()
