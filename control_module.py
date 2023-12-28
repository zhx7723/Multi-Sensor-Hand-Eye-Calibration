import glob
import time
import os

import cv2
import robolink
import numpy as np
from PyQt5.QtGui import QPixmap, QImage

import utility_functions as util
import cv2 as cv
from robodk import robomath
from camera_module import CameraModule
from result_module import ResultModule
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox


def cvimage2Qpixmap(rgb_image):
    rgb_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    return pixmap


class LoopThread(QThread):
    update_signal = pyqtSignal(int)
    update_result = pyqtSignal(str)
    update_graph_signal = pyqtSignal(QPixmap)

    def __init__(self, RDK, camera_module, dir_path, checker_size, resolution):
        super().__init__()
        self.RDK = RDK
        self.robot = RDK.Item('UR5e', robolink.ITEM_TYPE_ROBOT)
        self.camera_module = camera_module

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
                camera_posesGT_6param = robomath.Pose_2_UR(
                    self.charuco_frame.PoseWrt(self.camera_frame))  # based on opencv setup
                self.camera_GTposes_abs.append(np.array(robomath.UR_2_Pose(camera_posesGT_6param)).T)

                # capture image images
                self.camera_module.save_image(i, j)
                self.captures.append(f"{self.dir_path}/captures/cap_{i + 1}{j + 1}.png")
                time.sleep(0.05)

        self.calculate()

    def calculate(self):
        for i in range(len(self.tcp_poses_abs) - 1):
            self.tcp_poses.append((util.pose_inverse(self.tcp_poses_abs[i + 1]) @ self.tcp_poses_abs[i]))

        # camera_GTposes = []
        for i in range(len(self.camera_GTposes_abs) - 1):
            # camera_GTposes.append((camera_GTposes_abs[i])@(util.pose_inverse(camera_GTposes_abs[i+1])))
            self.camera_GTposes.append(
                (self.camera_GTposes_abs[i + 1]) @ (util.pose_inverse(self.camera_GTposes_abs[i])))

        # imu_poses = []
        for i in range(len(self.imu_poses_abs) - 1):
            # imu_poses.append((util.pose_inverse(imu_poses_abs[i])@(imu_poses_abs[i+1])))
            self.imu_poses.append((self.imu_poses_abs[i + 1]) @ (util.pose_inverse(self.imu_poses_abs[i])))

        # calculate the camera pose using PnP algorithm
        rvecs_camera, tvecs_camera = self.calib_and_estimate_pose(10, 7, 640, 480)

        # Convert the rotation vectors to rotation matrix representation
        rmat_camera = []
        for item in rvecs_camera:
            rmat_camera.append(util.euler_2_matrix(item).reshape(3, 3))

        # Create a 4x4 homogenous pose matrix for the camera
        camera_poses_abs = []
        for item_rmat, item_tvec in zip(rmat_camera, tvecs_camera):
            camera_poses_abs.append(util.create_homogeneous_pose(item_rmat, item_tvec))

        # calculate the relative poses of the camera: no_of_samples = (len(absolut_pose)-1)
        camera_poses = []
        for i in range(len(camera_poses_abs) - 1):
            # camera_GTposes.append((camera_GTposes_abs[i])@(util.pose_inverse(camera_GTposes_abs[i+1])))
            camera_poses.append((camera_poses_abs[i + 1]) @ (util.pose_inverse(camera_poses_abs[i])))

        ###############################################################################
        ######      Decompose Pose and Add Noise  to tvec and rot_axis         ########
        ###############################################################################

        # Max and Min noise settings for rotation axis and translation vector
        # ---------------------------------------------------------------------
        tvec_noise_min = -5
        tvec_noise_max = 5
        rot_noise_min = -np.pi / 100
        rot_noise_max = np.pi / 100

        # Decompose the 4x4 homogenous poses into 3x3 rmat and 3x1 tvec for TCP
        rmat_tcp = []
        tvec_tcp = []

        for item in self.tcp_poses:
            rmat_tcp_temp, tvec_tcp_temp = util.pose_decomposition(item)
            rmat_tcp.append(rmat_tcp_temp)
            tvec_tcp_temp = tvec_tcp_temp.reshape(3, 1)

            # Add uniformly distributed noise to the translation vector
            tvec_tcp_temp = util.add_uniform_noise_to_rot_axis(tvec_tcp_temp, tvec_noise_min, tvec_noise_max)
            tvec_tcp.append(tvec_tcp_temp)

        # Decompose the 4x4 homogenous poses into 3x3 rmat and 3x1 tvec for Camera
        rmat_camera = []
        tvec_camera = []

        for item in camera_poses:
            rmat_camera_temp, tvec_camera_temp = util.pose_decomposition(item)
            rmat_camera.append(rmat_camera_temp)
            tvec_camera_temp = tvec_camera_temp.reshape(3, 1)

            # Add uniformly distributed noise to the translation vector
            tvec_camera_temp = util.add_uniform_noise_to_rot_axis(tvec_camera_temp, tvec_noise_min, tvec_noise_max)
            tvec_camera.append(tvec_camera_temp)

        # Decompose the 4x4 homogenous poses into 3x3 rmat and 3x1 tvec for IMU
        rmat_imu = []
        tvec_imu = []

        for item in self.imu_poses:
            rmat_imu_temp, tvec_imu_temp = util.pose_decomposition(item)
            rmat_imu.append(rmat_imu_temp)
            tvec_imu_temp = tvec_imu_temp.reshape(3, 1)

            # Add uniformly distributed noise to the translation vector
            tvec_imu_temp = util.add_uniform_noise_to_rot_axis(tvec_imu_temp, tvec_noise_min, tvec_noise_max)
            tvec_imu.append(tvec_imu_temp)

        # Calculate the axes of rotation: for TCP rot, Camera rot and IMU rot.
        # calculate TCP rotation axis
        axis_tcp = []
        for item in rmat_tcp:
            axis_tcp_temp = util.rmat_2_axis(item)  # temporal axis as a list of values
            axis_tcp_temp = axis_tcp_temp.reshape((3, 1))  # axis as a 3x1 vector

            # Add uniformly distributed noise to the rotation axis
            axis_tcp_temp = util.add_uniform_noise_to_rot_axis(axis_tcp_temp, rot_noise_min, rot_noise_max)

            axis_tcp.append(axis_tcp_temp)

            # calculate camera rotation axis
        axis_camera = []
        for item in rmat_camera:
            axis_camera_temp = util.rmat_2_axis(item)  # temporal axis as a list of values
            axis_camera_temp = axis_camera_temp.reshape((3, 1))  # axis as a 3x1 vector

            # Add uniformly distributed noise to the rotation axis
            axis_camera_temp = util.add_uniform_noise_to_rot_axis(axis_camera_temp, rot_noise_min, rot_noise_max)

            axis_camera.append(axis_camera_temp)

            # calculate IMU rotation axis
        axis_imu = []
        for item in rmat_imu:
            axis_imu_temp = util.rmat_2_axis(item)  # temporal axis as a list of values
            axis_imu_temp = axis_imu_temp.reshape((3, 1))  # axis as a 3x1 vector

            # Add uniformly distributed noise to the rotation axis
            axis_imu_temp = util.add_uniform_noise_to_rot_axis(axis_imu_temp, rot_noise_min, rot_noise_max)

            axis_imu.append(axis_imu_temp)

        ###############################################################################
        ######         Solve Hand-Eye-Calibration: Calculate Rotations         ########
        ###############################################################################

        # Create an observation matrix with the stack of axis from TCP, Camera and IMU

        # create an axis-matrix stack for the tcp
        axis_mat_tcp = np.hstack(axis_tcp)

        # create an axis-matrix stack for the camera
        axis_mat_camera = np.hstack(axis_camera)

        # create an axis-matrix stack for the imu
        axis_mat_imu = np.hstack(axis_imu)

        ###############################################################################

        # Calculate the rotation matrix with least squares optimization solution

        # For TCP and Camera: Rx
        rmat_noisy_tcp_camera = axis_mat_tcp @ util.right_pseudoinverse(axis_mat_camera)

        # For Camera and IMU: Ry
        rmat_noisy_camera_imu = axis_mat_camera @ util.right_pseudoinverse(axis_mat_imu)

        # For TCP and IMU: Rz
        rmat_noisy_tcp_imu = axis_mat_tcp @ util.right_pseudoinverse(axis_mat_imu)

        ###############################################################################

        # Calculate the nearent rotation matrix to the noisy matrix for all loops

        Rx = util.nearest_rotation_matrix_svd(rmat_noisy_tcp_camera)
        # Rx = rmat_noisy_tcp_camera

        Ry = util.nearest_rotation_matrix_svd(rmat_noisy_camera_imu)
        # Ry = rmat_noisy_camera_imu

        Rz = util.nearest_rotation_matrix_svd(rmat_noisy_tcp_imu)
        # Rz = rmat_noisy_tcp_imu

        ###############################################################################
        ######       Solve Hand-Eye-Calibration: Calculate Translations        ########
        ###############################################################################

        # Calculate the translation vector  of the transform by least squares approach
        # Now that we have obtained the value of the rotation matrices, we will proceed
        # to obtain the translation vector from equation (6??)

        ###############################################################################
        # Calculating the translation vector for X transformation (tx)
        ###############################################################################
        # create the 'C' measurment matrix
        rmat_tcp_eye = []
        I = np.identity(3, dtype="int")
        for item in rmat_tcp:
            rmat_tcp_eye.append(item - I)

        # create a vertical stack of R - I matrix
        C_tcp_eye = np.vstack(rmat_tcp_eye)

        # create the 'd' measurment matrix (Rt - t)
        tvec_tcp_cam = []

        Rx_t = []
        for item in tvec_camera:
            Rx_t.append(Rx @ item)

        d_x = []
        for item_cam, item_tcp in zip(Rx_t, tvec_tcp):
            d_x.append((item_cam - item_tcp).reshape(3, 1))

        # create a vertical stack of 'Rx*tB - tA'
        d_tcp_cam = np.vstack(d_x)

        # use the left pseudo inverse to calculate for tx
        tx = (np.linalg.inv((C_tcp_eye.T) @ C_tcp_eye) @ (C_tcp_eye.T)) @ d_tcp_cam

        ###############################################################################
        # Calculating the translation vector for Y transformation (ty)
        ###############################################################################

        # create the 'C' measurment matrix
        rmat_cam_eye = []
        I = np.identity(3, dtype="int")
        for item in rmat_camera:
            rmat_cam_eye.append(item - I)

        # create a vertical stack of R - I matrix
        C_cam_eye = np.vstack(rmat_cam_eye)

        # create the 'd' measurment matrix (Rt - t)
        tvec_cam_imu = []

        Ry_t = []
        for item in tvec_imu:
            Ry_t.append(Ry @ item)

        d_y = []
        for item_imu, item_cam in zip(Ry_t, tvec_camera):
            d_y.append((item_imu - item_cam).reshape(3, 1))

        # create a vertical stack of 'Rx*tB - tA'
        d_cam_imu = np.vstack(d_y)

        # use the left pseudo inverse to calculate for tx
        ty = (np.linalg.inv((C_cam_eye.T) @ C_cam_eye) @ (C_cam_eye.T)) @ d_cam_imu

        ###############################################################################
        # Calculating the translation vector for Z transformation (tz)
        ###############################################################################

        # create the 'd' measurment matrix (Rt - t)
        tvec_tcp_imu = []

        Rz_t = []
        for item in tvec_imu:
            Rz_t.append(Rz @ item)

        d_z = []
        for item_imu, item_tcp in zip(Rz_t, tvec_tcp):
            d_z.append((item_imu - item_tcp).reshape(3, 1))

        # create a vertical stack of 'Ry*tC - tA'
        d_tcp_imu = np.vstack(d_z)

        # use the left pseudo inverse to calculate for tx
        tz = (np.linalg.inv((C_tcp_eye.T) @ C_tcp_eye) @ (C_tcp_eye.T)) @ d_tcp_imu

        ##############################################################################
        # Obtaining the HEC 4 x 4 homogenous transformation matrices of the sensors
        ##############################################################################

        # The homogenous transformation matrix can now be obtained from Rx and tx
        X = np.vstack((np.hstack((Rx, tx)), np.array([0, 0, 0, 1])))
        # The homogenous transformation matrix can now be obtained from Ry and ty
        Y = np.vstack((np.hstack((Ry, ty)), np.array([0, 0, 0, 1])))
        # The homogenous transformation matrix can now be obtained from Rz and tz
        Z = np.vstack((np.hstack((Rz, tz)), np.array([0, 0, 0, 1])))

        results = f"The tansformation of TCP to Camera, X:\n {X}\n"
        results += f"The tansformation of TCP to Camera, Y:\n {Y}\n"
        results += f"The tansformation of TCP to Camera, Z:\n {Z}\n"

        self.update_result.emit(results)

    def drawBoxes(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img

    def calib_and_estimate_pose(self,
                                chessboard_width,
                                chessboard_height,
                                image_frame_width,
                                image_frame_height):
        ###########################################################################
        #                          1. Camera Calibration                          #
        ###########################################################################

        ####### FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ##############

        chessboardSize = (chessboard_width, chessboard_height)
        frameSize = (image_frame_width, image_frame_height)

        # chessboardSize = (10,7)
        # frameSize = (640,480)

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

        size_of_chessboard_squares_mm = 15
        objp = objp * size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('captures/*.png')

        for image in images:

            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                # cv.imshow('img', img)
                pixmap = cvimage2Qpixmap(img)
                self.update_graph_signal.emit(pixmap)
                cv.waitKey(500)  # wait for 1 second

        # cv.destroyAllWindows()

        ############################## CALIBRATION ################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results = f"The completion time of the calibration: {time_str}---------\n"
        results += f"Camera Calibrated:\n {ret}\n"
        results += f"Camera Matrix:\n {cameraMatrix}\n"
        results += f"Distortion Parameters:\n {dist}\n"
        self.update_result.emit(results)
        # print("\nCamera Calibrated: ", ret)
        # print("\nCamera Matrix:\n", cameraMatrix)
        # print("\nDistortion Parameters:\n", dist)

        np.savez("CameraParams", cameraMatrix=cameraMatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

        ############################## UNDISTORTION ###############################
        # '''
        img = cv.imread('captures/cap_11.png')
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        # Undistort
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        # '''

        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("total error: {}".format(mean_error / len(objpoints)))

        ###########################################################################
        #                            2. Pose Estimation                           #
        ###########################################################################

        # Load previously saved data
        with np.load('CameraParams.npz') as file:
            mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix', 'dist', 'rvecs', 'tvecs')]

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        axisBoxes = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                                [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

        rvecs = []
        tvecs = []
        for image in glob.glob('captures/*.png'):

            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            if ret == True:
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors.
                ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
                rvecs.append(rvec)
                tvecs.append(tvec)

                # Project 3D points to image plane
                imgpts, jac = cv.projectPoints(axisBoxes, rvec, tvec, mtx, dist)

                img = self.drawBoxes(img, corners2, imgpts)
                # cv.imshow('img', img)
                pixmap = cvimage2Qpixmap(img)
                self.update_graph_signal.emit(pixmap)
                cv.waitKey(500)  # wait for 1 second

                '''k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                cv.imwrite('pose'+image, img)'''

        # cv.destroyAllWindows()

        return rvecs, tvecs


class ControlModule:
    def __init__(self, main_window_ui):
        self.ui = main_window_ui
        self.RDK = robolink.Robolink()
        self.dir_path = os.path.realpath(os.path.dirname(__file__))
        self.camera_module = CameraModule(self.ui.label_camera, self.RDK, self.dir_path)
        self.result_module = ResultModule(self.ui)

        self.thread = None

        #         self.ui.pushButton_camera_config.clicked.connect(self.handle_config_button)
        self.ui.pushButton_start.clicked.connect(self.handle_start_button)
        self.ui.pushButton_stop.clicked.connect(self.handle_stop_button)
        self.ui.pushButton_continue.clicked.connect(self.handle_continue_button)
        self.ui.pushButton_connect.clicked.connect(self.handle_connect_button)
        self.ui.pushButton_disconnect.clicked.connect(self.handle_disconnect_button)

        self.ui.progressBar.setValue(0)
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(False)

    def handle_connect_button(self):
        self.camera_module.start()

    def handle_disconnect_button(self):
        self.camera_module.stop()
        self.ui.label_camera.setText("Camera Disconnected")

    def handle_start_button(self):
        # check lineEdit_position
        # Use strip() to remove possible leading and trailing spaces
        square_size = self.ui.lineEdit_square_size.text().strip()
        height = self.ui.lineEdit_height.text().strip()
        width = self.ui.lineEdit_width.text().strip()
        if not square_size or not height or not width:
            QMessageBox.warning(self.ui.pushButton_start, "Warning",
                                "Missing parameters for calibration configuration!")
            return

        height = float(height)
        width = float(width)
        square_size = float(square_size)

        if height > width:
            height, width = width, height

        self.thread = LoopThread(self.RDK, self.camera_module, self.dir_path, [height, width, square_size])
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.update_signal.connect(self.ui.progressBar.setValue)
        self.thread.start()
        self.thread.update_result.connect(self.result_module.update_result)
        self.thread.update_graph_signal.connect(self.update_graph)

        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(True)

    def handle_stop_button(self):
        self.thread.is_paused = True
        self.ui.label_state.setText("Calibration Stop")
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_continue.setEnabled(True)
        self.ui.pushButton_start.setText("Restart")
        self.ui.pushButton_start.setEnabled(True)

    def handle_continue_button(self):
        self.thread.is_paused = False

        self.ui.label_state.setText("Calibrating")
        self.ui.pushButton_start.setText("Start")
        self.ui.pushButton_start.setEnabled(False)
        self.ui.pushButton_stop.setEnabled(True)
        self.ui.pushButton_continue.setEnabled(False)

    def on_thread_finished(self):
        self.ui.pushButton_start.setText("Start")
        self.ui.pushButton_start.setEnabled(True)
        self.ui.pushButton_stop.setEnabled(False)
        self.thread.deleteLater()
        self.thread = None
        self.ui.label_state.setText("Calibration Complete!")

    def update_graph(self, pixmap: QPixmap):
        self.ui.label_graph.setPixmap(pixmap)

    def stop_camera(self):
        self.camera_module.stop()
