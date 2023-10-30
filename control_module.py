import time
import os
import robolink
import numpy as np
import cv2 as cv
from robodk import robomath
from camera_module import CameraModule
from result_module import ResultModule
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


class LoopThread(QThread):
    update_signal = pyqtSignal(int)

    def __init__(self, RDK, camera_module, result_module, dir_path):
        super().__init__()
        self.RDK = RDK
        self.robot = RDK.Item('UR5e', robolink.ITEM_TYPE_ROBOT)
        self.camera_module = camera_module
        self.result_module = result_module

        self.dir_path = dir_path

        self.home = self.RDK.Item('home')
        self.imu_frame = self.RDK.Item('cf_imu_frame', robolink.ITEM_TYPE_FRAME)
        self.camera_frame = self.RDK.Item('cf_camera_frame', robolink.ITEM_TYPE_FRAME)
        self.charuco_frame = self.RDK.Item('charuco_frame', robolink.ITEM_TYPE_FRAME)
        self.tcp_frame = self.RDK.Item('tcp_frame', robolink.ITEM_TYPE_FRAME)
        self.base_frame = self.RDK.Item('UR5e Base', robolink.ITEM_TYPE_FRAME)

        self.tcp_poses_abs = []  # Absolute tcp poses wrt base
        self.tcp_poses = []  # Relative tcp poses (new pose wrt old pose)
        self.imu_poses_abs = []  # Absolute imu poses wrt camera
        self.imu_poses = []  # Relative imu poses (new pose wrt old pose)
        self.camera_GTposes_abs = []  # Absolute target object poses wrt camera
        self.camera_GTposes = []  # Relative target object poses (new pose wrt old pose)
        self.captures = []

        self.is_paused = False
        self.is_running = True

    def run(self):

        self.robot.MoveL(self.home)
        time.sleep(0.2)
        current_iteration = 0

        for i in range(3):
            if not self.is_running:
                break
            for j in range(8):
                if not self.is_running:
                    break
                while self.is_paused:
                    self.msleep(100)

                current_iteration += 1
                self.update_signal.emit(int((current_iteration / 24) * 100))  # 发送信号更新界面

                target = (self.RDK.Item(f'loop{i + 1} {j + 1}'))  # import targets
                if j + 1 < 7:
                    self.robot.MoveL(target)  # moveL to the targets
                elif j + 1 > 6:
                    self.robot.MoveJ(target)  # moveJ to the targets
                time.sleep(0.1)

                # get the tcp pose
                tcp_poses_6param = robomath.Pose_2_UR(self.tcp_frame.PoseWrt(self.base_frame))
                self.tcp_poses_abs.append(np.array(robomath.UR_2_Pose(tcp_poses_6param)).T)

                # get the imu pose
                # imu_poses_6param = robomath.Pose_2_UR(table_frame.PoseWrt(imu_frame))
                imu_poses_6param = robomath.Pose_2_UR(
                    self.imu_frame.PoseWrt(self.base_frame))  # Actually IMU readings taken wrt some global frame
                self.imu_poses_abs.append(np.array(robomath.UR_2_Pose(imu_poses_6param)).T)

                # get the camera Ground Truth pose
                # camera_posesGT_6param = robomath.Pose_2_UR(camera_frame.PoseWrt(charuco_frame))
                camera_posesGT_6param = robomath.Pose_2_UR(self.charuco_frame.PoseWrt(self.camera_frame))  # based on opencv setup
                self.camera_GTposes_abs.append(np.array(robomath.UR_2_Pose(camera_posesGT_6param)).T)

                # capture image images
                self.camera_module.save_image(i, j)
                self.captures.append(f"{self.dir_path}/captures/cap_{i + 1}{j + 1}.png")
                time.sleep(0.2)

        self.calculate()

    def calculate(self):
        self.result_module.update_result("X: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]\n\n"
                                    "Y: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]\n\n"
                                    "Z: \n[[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1],\n[0.1, 0.1, 0.1]]")



class ControlModule:
    def __init__(self, main_window_ui):
        self.ui = main_window_ui
        self.RDK = robolink.Robolink()
        self.dir_path = os.path.realpath(os.path.dirname(__file__))
        self.camera_module = CameraModule(self.ui.label_camera, self.RDK, self.dir_path)
        self.result_module = ResultModule(self.ui.textBrowser_result, self.ui.pushButton_save)

        self.thread = LoopThread(self.RDK, self.camera_module, self.result_module, self.dir_path)
        self.thread.finished.connect(self.on_thread_finished)

        self.ui.pushButton_camera_config.clicked.connect(self.handle_config_button)
        self.ui.pushButton_start.clicked.connect(self.handle_start_button)
        self.ui.pushButton_stop.clicked.connect(self.handle_stop_button)
        self.ui.pushButton_continue.clicked.connect(self.handle_continue_button)

        self.ui.progressBar.setValue(0)
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
        # Use strip() to remove possible leading and trailing spaces
        # text_value = self.ui.lineEdit_position.text().strip()
        #
        # if not text_value:
        #     QMessageBox.warning(self.ui.pushButton_start, "Warning",
        #         "Please enter a number in the Position Number text box!")
        #     return
        self.thread.update_signal.connect(self.ui.progressBar.setValue)
        self.thread.start()

        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(True)
        self.ui.pushButton_camera_config.setEnabled(False)

    def handle_stop_button(self):
        self.thread.is_paused = True

        self.ui.label_state.setText("Calibration Stop")
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(True)
        self.ui.pushButton_camera_config.setEnabled(True)

    def handle_continue_button(self):
        self.thread.is_paused = False

        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_stop.setEnabled(True)
        self.ui.pushButton_continue.setEnabled(False)
        self.ui.pushButton_camera_config.setEnabled(False)

    def on_thread_finished(self):
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.label_state.setText("Calibration Complete!")

    def stop_camera(self):
        self.camera_module.stop()


