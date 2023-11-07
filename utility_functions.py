import random
import numpy as np


def rmat_2_rod(rmat):
    trace = np.trace(rmat)
    angle = np.arccos((trace - 1) / 2)

    skew_mat = (rmat - rmat.T) / (2 * np.sin(angle))
    axis = np.array([skew_mat[2, 1], skew_mat[0, 2], skew_mat[1, 0]])

    rod_vec = axis * angle  # calculate the rodrigues vector
    return rod_vec


# This function splits the poses to lists for each repetition
# sets the translations in meters or milimeters
# if you want the translations in meters, set the in_meters flag to 'True'
def split_poses(poses, in_meters=True):
    tvecs = []
    rvecs = []
    if in_meters == False:
        for pose in poses:
            tvec = np.array([pose[0] / 1000, pose[1] / 1000, pose[2] / 1000])
            rvec = np.array([pose[3], pose[4], pose[5]])
            tvecs.append(tvec)
            rvecs.append(rvec)
    else:
        for pose in poses:
            tvec = np.array([pose[0], pose[1], pose[2]])
            rvec = np.array([pose[3], pose[4], pose[5]])
            tvecs.append(tvec)
            rvecs.append(rvec)
    return tvecs, rvecs


# This function converts the euler rotation vector XYZ (in degrees)
# to a 3x3 rotation matrix
def euler_2_matrix(rvec):
    theta1 = rvec[0]
    theta2 = rvec[1]
    theta3 = rvec[2]

    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    rmat = np.array([[c2 * c3, -c2 * s3, s2],
                     [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                     [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])

    return rmat


# This function converts a rotation matrix to its axis angle representation
# This instance returns only the axis of rotation 
# (you can choose to return both axis and angle)

def rmat_2_axis(rmat):
    epsilon = 1e-12  # Small value to avoid division by zero

    # Ensure the input matrix is valid rotation matrix
    # if np.abs(np.linalg.det(rmat) - 1.0) > epsilon:
    #    raise ValueError("Invalid rotation matrix")

    # Compute the trace of the rotation matrix
    trace = np.trace(rmat)

    # Calculate the angle of rotation
    angle = np.arccos((trace - 1.0) / 2.0)

    # Check for special cases of zero or 180 degrees rotation
    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0])
    elif np.isclose(angle, np.pi):
        # Find a non-zero element of the rotation matrix
        for i in range(3):
            for j in range(3):
                if np.abs(rmat[i, j]) > epsilon:
                    axis = np.cross([0.0, 0.0, 1.0], [i, j, 0.0])
                    return axis / np.linalg.norm(axis)

    # Calculate the axis of rotation
    axis = 1.0 / (2.0 * np.sin(angle)) * np.array([
        rmat[2, 1] - rmat[1, 2],
        rmat[0, 2] - rmat[2, 0],
        rmat[1, 0] - rmat[0, 1]
    ])

    return axis / np.linalg.norm(axis)


# This function adds noise to the translation and rotation vectors
def add_noise(t, r, tnm, tns, rnm, rns):
    translation_noise_mean = tnm  # Mean of translation noise
    translation_noise_stddev = tns  # Standard deviation of translation noise

    rotation_noise_mean = rnm  # Mean of rotation noise in degrees
    rotation_noise_stddev = rns  # Standard deviation of rotation noise in degrees

    nt = []
    nr = []

    for i in range(len(t)):
        nti = []
        nri = []

        # Add noise to translation vector
        translation_noise = [random.gauss(translation_noise_mean, translation_noise_stddev)]
        nti.append((t[i][0] + translation_noise)[0])
        translation_noise = [random.gauss(translation_noise_mean, translation_noise_stddev)]
        nti.append((t[i][1] + translation_noise)[0])
        translation_noise = [random.gauss(translation_noise_mean, translation_noise_stddev)]
        nti.append((t[i][2] + translation_noise)[0])

        # Add noise to rotation vector
        rotation_noise = [random.gauss(rotation_noise_mean, rotation_noise_stddev)]
        nri.append((r[i][0] + rotation_noise)[0])
        rotation_noise = [random.gauss(rotation_noise_mean, rotation_noise_stddev)]
        nri.append((r[i][1] + rotation_noise)[0])
        rotation_noise = [random.gauss(rotation_noise_mean, rotation_noise_stddev)]
        nri.append((r[i][2] + rotation_noise)[0])

        nt.append(np.array(nti))
        nr.append(np.array(nri))

    return nt, nr


# Right Moore-Penrose pseudoinverse
def right_pseudoinverse(mat):
    right_p_inv = mat.T @ np.linalg.inv(mat @ mat.T)
    return right_p_inv


# Right Moore-Penrose pseudoinverse
def left_pseudoinverse(mat):
    left_p_inv = np.linalg.inv((mat.T) @ mat) @ mat.T
    return left_p_inv


'''
Iterative orthonormalization method for obtaining nearest rotation matrix
Two (2) iterations are sufficient
'''


def nearest_rotation_matrix(noisy_rmat):
    I = np.identity(3, dtype="int")

    # Iteration 1
    S1 = noisy_rmat.T @ noisy_rmat
    noisy_rmat1 = noisy_rmat @ ((3 * I) + S1) @ np.linalg.inv(I + (3 * S1))

    # Iteration 2
    S2 = noisy_rmat1.T @ noisy_rmat1
    noisy_rmat2 = noisy_rmat1 @ ((3 * I) + S2) @ np.linalg.inv(I + (3 * S2))

    denoised_rmat = noisy_rmat2

    distance = np.linalg.norm(noisy_rmat - denoised_rmat, 'fro')
    print(f"The Frobenius norm distance is: {distance}")

    return denoised_rmat


def nearest_rotation_matrix_svd(noisy_rmat):
    U, _, Vt = np.linalg.svd(noisy_rmat)
    R = np.dot(U, Vt)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    distance = np.linalg.norm(noisy_rmat - R, 'fro')
    # print(f"The Frobenius norm distance is: {distance}")

    return R


def create_homogeneous_pose(rmat, tvec):
    # assert rmat.shape == (3, 3) "Rotation matrix should be a 3x3 matrix"
    # assert len(tvec) == 3 "Translation vector should have 3 elements"

    # hom_pose = np.eye(4)
    # hom_pose[:3, :3] = rmat
    # hom_pose[:3, 3] = tvec

    hom_pose = np.vstack((np.hstack((rmat, tvec)), (0, 0, 0, 1)))

    return hom_pose


def pose_decomposition_list(hom_matrix):
    rmats = []
    tvecs = []
    for item in hom_matrix:
        rmat = hom_matrix[:3, :3]
        tvec = hom_matrix[:3, 3]
        rmats.append(rmat)
        tvecs.append(tvec)

    return rmats, tvecs


def pose_decomposition(hom_matrix):
    rmat = hom_matrix[:3, :3]
    tvec = hom_matrix[:3, 3]

    return rmat, tvec


# Function to calculate the mean of a list
def mean(lst):
    return sum(lst) / len(lst)


# Function to calculate the inverse of a 4x4 homogenous pose matrix 

def pose_inverse(pose):
    # Extract the rotation matrix (3x3) from the homogeneous pose matrix
    rot_mat = pose[:3, :3]

    # Extract the translation vector (3x1) from the homogeneous pose matrix
    tvec = pose[:3, 3]

    # Compute the inverse of the rotation matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    # rot_mat_inv = rot_mat.T

    # Compute the negative of the rotation matrix multiplied by the translation vector
    tvec_inv = -np.dot(rot_mat_inv, tvec)

    # Construct the inverse homogeneous pose matrix
    inv_pose = np.identity(4)
    inv_pose[:3, :3] = rot_mat_inv
    inv_pose[:3, 3] = tvec_inv

    return inv_pose


def rotation_loss(rmat_sensor1, rmat_sensor2, rmat_HEC):
    errors_rot = []
    for item1, item2 in zip(rmat_sensor1, rmat_sensor2):
        error_mat = item1 @ rmat_HEC - rmat_HEC @ item2
        errors_rot.append(np.linalg.norm(error_mat))

    error_rot = sum(errors_rot) / len(errors_rot)
    return error_rot


# This function takes in the rotational pose components and calculates the constraint rotation error
def constraint_rot_loss(Rx, Ry, Rz):
    error_rot_const = np.linalg.norm(Rx @ Ry - Rz)
    return error_rot_const


# This function adds uniformly distributed translation noise with zero mean to 
# the translation vectors 
def add_uniform_noise_to_tvec(tvec, tvec_noise_min, tvec_noise_max):
    # Generate random noise for the translation vector
    translation_noise = np.random.uniform(tvec_noise_min, tvec_noise_max, size=(3, 1))

    # Calculate the mean of the noise
    noise_mean = (tvec_noise_min + tvec_noise_max) / 2.0

    # Subtract the mean to achieve zero-mean noise
    translation_noise -= noise_mean

    # Add noise to the translation vector
    noisy_translation_vector = tvec + translation_noise

    return noisy_translation_vector


# This function adds uniformly distributed rotation noise with zero mean to 
# the rotation axes 
def add_uniform_noise_to_rot_axis(rot_axis, rot_noise_min, rot_noise_max):
    # Generate random noise for the translation vector
    rotation_noise = np.random.uniform(rot_noise_min, rot_noise_max, size=(3, 1))

    # Calculate the mean of the noise
    noise_mean = (rot_noise_min + rot_noise_max) / 2.0

    # Subtract the mean to achieve zero-mean noise
    rotation_noise -= noise_mean

    # Add noise to the translation vector
    noisy_rot_axis = rot_axis + rotation_noise

    return noisy_rot_axis


# This function calculates the derivative of a rotation matrix with respect to theta(angle)
def rmat_derivative(rmat):
    # Compute the trace of the rotation matrix
    trace = np.trace(rmat)

    # Calculate the angle of rotation
    theta = np.arccos((trace - 1.0) / 2.0)

    # Compute the trace of the rotation matrix
    axis = util.rmat_2_axis(rmat)

    # Compute the skew symmetric matrix (cross product matrix) 
    S = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    dR_dtheta = S @ rmat
    return dR_dtheta


# This function takes in a rotation matrix and calculates its 3 partial derivative 
# in terms of the axis parameters s1, s2, s3.

def dR_ds_k(rmat):
    # calculate the axis of the rotation matrix: s_k = [s1, s2, s3]
    # Note: DO NOT NORMALISE THE AXIS!
    # --------------------------------------------------------------------------
    epsilon = 1e-12  # Small value to avoid division by zero

    # Compute the trace of the rotation matrix
    trace = np.trace(rmat)

    # Calculate the angle of rotation
    angle = np.arccos((trace - 1.0) / 2.0)

    # Check for special cases of zero or 180 degrees rotation
    if np.isclose(angle, 0.0).any():
        return np.array([1.0, 0.0, 0.0])
    elif np.isclose(angle, np.pi):
        # Find a non-zero element of the rotation matrix
        for i in range(3):
            for j in range(3):
                if np.abs(rmat[i, j]) > epsilon:
                    axis = np.cross([0.0, 0.0, 1.0], [i, j, 0.0])
                    return axis  # / np.linalg.norm(axis)

    # Calculate the axis of rotation
    axis = 1.0 / (2.0 * np.sin(angle)) * np.array([
        rmat[2, 1] - rmat[1, 2],
        rmat[0, 2] - rmat[2, 0],
        rmat[1, 0] - rmat[0, 1]
    ])

    s0 = axis[0]
    s1 = axis[1]
    s2 = axis[2]

    t = np.sqrt(s0 ** 2 + s1 ** 2 + s2 ** 2)

    alpha = np.sin(t) / t
    beta = (1 - np.cos(t)) / (t ** 2)
    gamma = (np.sin(t) - t * np.cos(t)) / (t ** 3)
    delta = (2 * (1 - np.cos(t)) - t * np.sin(t)) / (t ** 4)

    # --------------------------------------------------------------------------

    dR_ds0 = np.array([[delta * s0 * (s1 ** 2 + s2 ** 2), beta * s1 + s0 * (gamma * s2 - delta * s0 * s1),
                        beta * s2 - s0 * (gamma * s1 + delta * s0 * s2)],
                       [beta * s1 - s0 * (gamma * s2 + delta * s0 * s1),
                        delta * s0 * (s0 ** 2 + s2 ** 2) - 2 * beta * s0, s0 * (gamma * s0 - delta * s1 * s2) - alpha],
                       [s2 * (beta - delta * s0 ** 2) + gamma * s0 * s1, alpha - s0 * (gamma * s0 + delta * s1 * s2),
                        delta * s0 * (s0 ** 2 + s1 ** 2) - 2 * beta * s0]])

    dR_ds1 = np.array([[delta * s1 * (s1 ** 2 + s2 ** 2) - 2 * beta * s1,
                        beta * s0 + s1 * (gamma * s2 - delta * s0 * s1), alpha - s1 * (gamma * s1 + delta * s0 * s2)],
                       [beta * s0 - s1 * (gamma * s2 + delta * s0 * s1), delta * s1 * (s0 ** 2 + s2 ** 2),
                        s2 * (beta - delta * s1 ** 2) + gamma * s0 * s1],
                       [s1 * (gamma * s1 - delta * s0 * s2) - alpha, beta * s2 - s1 * (gamma * s0 + delta * s1 * s2),
                        delta * s1 * (s0 ** 2 + s1 ** 2) - 2 * beta * s1]])

    dR_ds2 = np.array([[delta * s2 * (s1 ** 2 + s2 ** 2) - 2 * beta * s2, s2 * (gamma * s2 - delta * s0 * s1) - alpha,
                        beta * s0 - s2 * (gamma * s1 + delta * s0 * s2)],
                       [alpha - s2 * (gamma * s2 + delta * s0 * s1), delta * s2 * (s0 ** 2 + s2 ** 2) - 2 * beta * s2,
                        beta * s1 + s2 * (gamma * s0 - delta * s1 * s2)],
                       [beta * s0 + s2 * (gamma * s1 - delta * s0 * s2),
                        beta * s1 - s2 * (gamma * s0 + delta * s1 * s2), delta * s2 * (s0 ** 2 + s1 ** 2)]])

    # print(dR_ds0)
    # print(dR_ds1)
    # print(dR_ds2)

    return dR_ds0, dR_ds1, dR_ds2, s0, s1, s2


# This function calculates the derivative of the loss function J(x) wrt R.
# We use it to calculate for dJx_dR
# It takes inputs: axis_tcp, axis_camera, Ry, Rz
def dJx_dR(axis_tcp, axis_camera, Ry, Rz):
    dJx_dR_list = []
    for item_tcp, item_camera in zip(axis_tcp, axis_camera):
        expression = np.linalg.norm((item_camera.T) * (Rz * (Ry.T) * item_camera - axis_tcp) + (item_camera.T) * (
            (Rz * (Ry.T)).T) * item_camera - (item_tcp.T) * item_camera)
        dJx_dR_list.append(expression)
    dJx_dR = sum(dJx_dR_list) * (1 / len(dJx_dR_list))

    return dJx_dR


def dJy_dR(axis_camera, axis_imu, Rx, Rz):
    dJy_dR_list = []
    for item_imu, item_camera in zip(axis_imu, axis_camera):
        expression = np.linalg.norm(
            (item_imu.T) * ((Rx.T) * Rz * item_imu - item_camera) + (item_imu.T) * (((Rx.T) * Rz).T) * item_imu - (
                item_camera.T) * item_imu)
        dJy_dR_list.append(expression)
    dJy_dR = sum(dJy_dR_list) * (1 / len(dJy_dR_list))

    return dJy_dR


def dJz_dR(axis_tcp, axis_imu, Rx, Ry):
    dJz_dR_list = []
    for item_tcp, item_imu in zip(axis_tcp, axis_imu):
        expression = np.linalg.norm(
            (item_imu.T) * (Rx * Ry * item_imu - item_tcp) + (item_imu.T) * ((Rx * Ry).T) * item_imu - (
                item_tcp.T) * item_imu)
        dJz_dR_list.append(expression)
    dJz_dR = sum(dJz_dR_list) * (1 / len(dJz_dR_list))

    return dJz_dR


def rmat_in_s(s0, s1, s2):
    t = np.sqrt(s0 ** 2 + s1 ** 2 + s2 ** 2)

    alpha = np.sin(t) / t
    beta = (1 - np.cos(t)) / (t ** 2)
    gamma = (np.sin(t) - t * np.cos(t)) / (t ** 3)
    delta = (2 * (1 - np.cos(t)) - t * np.sin(t)) / (t ** 4)

    rmat_s = np.array([[1 - beta * (s1 ** 2 + s2 ** 2), -alpha * s2 + beta * s0 * s1, alpha * s1 + beta * s0 * s2],
                       [alpha * s2 + beta * s0 * s1, 1 - beta * (s0 ** 2 + s2 ** 2), -alpha * s0 + beta * s1 * s2],
                       [-alpha * s1 + beta * s0 * s2, alpha * s0 + beta * s1 * s2, 1 - beta * (s0 ** 2 + s1 ** 2)]])

    return rmat_s


# Trial 1
'''
# This function calculates the derivative of the loss function J(x) wrt the axis parameters.
# We use it to calculate for dJx_ds0, dJx_ds1, and dJx_ds2
# It takes inputs: dRz_dsk and dRyt_dsk
def dJx_dsk(dRz_dsk, dRyt_dsk, rmat_tcp, rmat_camera, Rx, Ry, Rz):
    
    dJx_dsk_list = []
    for item_tcp, item_camera in zip(rmat_tcp, rmat_camera):
        expression = (item_tcp@Rz@np.linalg.inv(Ry) - Rz@np.linalg.inv(Ry)@item_camera)@(item_tcp@(dRz_dsk@np.linalg.inv(Ry) + Rz@dRyt_dsk) - (dRz_dsk@np.linalg.inv(Ry) + Rz@dRyt_dsk)@item_camera)
        exression_norm = np.linalg.norm(expression)
        dJx_dsk_list.append(exression_norm)
        
    dJx_ds = sum(dJx_dsk_list)*(2/len(dJx_dsk_list))
    
    return dJx_ds

# This function calculates the derivative of the loss function J(y) wrt the axis parameters.
# We use it to calculate for dJy_ds0, dJy_ds1, and dJy_ds2
# It takes inputs: dRxt_dsk and dRz_dsk
def dJy_dsk(dRxt_dsk, dRz_dsk, rmat_camera, rmat_imu, Rx, Ry, Rz):
    
    dJy_dsk_list = []
    for item_camera, item_imu in zip(rmat_camera, rmat_imu):
        expression = (item_camera@np.linalg.inv(Rx)@Rz - np.linalg.inv(Rx)@Rz@item_imu)@(item_camera@(dRxt_dsk@Rz + np.linalg.inv(Rx)@dRz_dsk) - (dRxt_dsk@Rz + np.linalg.inv(Rx)@dRz_dsk)@item_imu)
        exression_norm = np.linalg.norm(expression)
        dJy_dsk_list.append(exression_norm)
        
    dJy_ds = sum(dJy_dsk_list)*(2/len(dJy_dsk_list))
    
    return dJy_ds


# This function calculates the derivative of the loss function J(z) wrt the axis parameters.
# We use it to calculate for dJz_ds0, dJz_ds1, and dJz_ds2
# It takes inputs: dRx_dsk and dRy_dsk
def dJz_dsk(dRx_dsk, dRy_dsk, rmat_tcp, rmat_imu, Rx, Ry, Rz):
    
    dJz_dsk_list = []
    for item_tcp, item_imu in zip(rmat_tcp, rmat_imu):
        expression = (item_tcp@Rx@Ry - Rx@Ry@item_imu)@(item_tcp@(dRx_dsk@Ry + Rx@dRy_dsk) - (dRx_dsk@Ry + Rx@dRy_dsk)@item_imu)
        exression_norm = np.linalg.norm(expression)
        dJz_dsk_list.append(exression_norm)
        
    dJz_ds = sum(dJz_dsk_list)*(2/len(dJz_dsk_list))
    
    return dJz_ds

'''

# Trial 2
'''
# This function calculates the derivative of the loss function J(z) wrt the axis parameters s0, s1, s2.
# We use it to calculate for dJz_ds0, dJz_ds1, and dJz_ds2
# It takes inputs: dRz_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz
def dJz_dsk(dRz_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz):
    
    dJz_dsk_list = []
    for item_tcp, item_camera, item_imu in zip(axis_tcp, axis_camera, axis_imu):
        expression = np.linalg.norm(((item_camera.T)@Ry@(dRz_dsk.T)@Rz - (item_tcp.T)@(dRz_dsk))@(Ry.T)@item_camera + ((item_imu.T)@(dRz_dsk.T)@Rz - (item_camera.T)@(Rx.T)@(dRz_dsk))@item_imu)
        dJz_dsk_list.append(expression)    
    dJz_ds = sum(dJz_dsk_list)*(2/len(dJz_dsk_list))
    
    return dJz_ds


# This function calculates the derivative of the loss function J(x) wrt the axis parameters s0, s1, s2.
# We use it to calculate for dJx_ds0, dJx_ds1, and dJx_ds2
# It takes inputs: dRx_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz
def dJx_dsk(dRx_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz):
    
    dJx_dsk_list = []
    for item_tcp, item_camera, item_imu in zip(axis_tcp, axis_camera, axis_imu):
        expression = np.linalg.norm(((item_camera.T)@(dRx_dsk.T)@Rx - (item_imu.T)@(Rz.T)@(dRx_dsk))@item_camera + ((item_imu.T)@(Ry.T)@(dRx_dsk.T)@Rx - (item_tcp.T)@(dRx_dsk))@Ry@item_imu)
        dJx_dsk_list.append(expression)    
    dJx_ds = sum(dJx_dsk_list)*(2/len(dJx_dsk_list))
    
    return dJx_ds


# This function calculates the derivative of the loss function J(y) wrt the axis parameters s0, s1, s2.
# We use it to calculate for dJy_ds0, dJy_ds1, and dJy_ds2
# It takes inputs: dRy_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz
def dJy_dsk(dRy_dsk, axis_tcp, axis_camera, axis_imu, Rx, Ry, Rz):
    
    dJy_dsk_list = []
    for item_tcp, item_camera, item_imu in zip(axis_tcp, axis_camera, axis_imu):
        expression = np.linalg.norm(((item_tcp.T)@Rz@(dRy_dsk.T)@Ry - (item_camera.T)@(dRy_dsk))@(Rz.T)@item_tcp + ((item_imu.T)@(dRy_dsk.T)@Ry - (item_tcp.T)@(Rx)@(dRy_dsk))@item_imu)
        dJy_dsk_list.append(expression)    
    dJy_ds = sum(dJy_dsk_list)*(2/len(dJy_dsk_list))
    
    return dJy_ds

'''
