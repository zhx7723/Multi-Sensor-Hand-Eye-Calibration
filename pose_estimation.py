import numpy as np
import cv2 as cv
import glob


def drawBoxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def calib_and_estimate_pose(chessboard_width,
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
            cv.imshow('img', img)
            cv.waitKey(1000)  # wait for 1 second

    cv.destroyAllWindows()

    ############################## CALIBRATION ################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print("Camera Calibrated: ", ret)
    print("\nCamera Matrix:\n", cameraMatrix)
    print("\nDistortion Parameters:\n", dist)
    # print("\nRotation Vectors:\n", rvecs)
    # print("\nTranslation Vectors:\n", tvecs)

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

            img = drawBoxes(img, corners2, imgpts)
            cv.imshow('img', img)
            cv.waitKey(1000)  # wait for 1 second

            '''k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
            cv.imwrite('pose'+image, img)'''

    cv.destroyAllWindows()

    return rvecs, tvecs

# rvecs_camera, tvecs_camera = calib_and_estimate_pose(10, 7, 640, 480)
