import numpy as np

def get_epipole(F):
    '''
    Epipole is the eigenvector associated with smallest eigenvalue of F
    '''
    evalue, evector = np.linalg.eig(F)  # normalized evector
    index = np.argmin(evalue)
    epipole = evector[:, index]
    return epipole


def get_rotation_axis(d):
    # d_i to make image plane parallel
    # intersection line
    axis = np.array([-d[1], d[0], 0])
    return axis


def get_angle(epipole, axis):
    return np.arctan(epipole[2] / (axis[1] * epipole[0] - axis[0] * epipole[1]))


def get_plane_rotation_matrix(axis, angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dx = axis[0]
    dx2 = dx ** 2
    dy = axis[1]
    dy2 = dy ** 2
    return np.array([[dx2 + (1 - dx2) * cos_angle, dx * dy * (1 - cos_angle), dy * sin_angle],
                     [dx * dy * (1 - cos_angle), dy2 + (1 - dy2) * cos_angle, -dx * sin_angle],
                     [-dy * sin_angle, dx * sin_angle, cos_angle]])


def get_scanline_rotation_matrix(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def compute_prewarp(F, all=False):
    # apply a 3D rotation to make the image planes parall
    epipole1 = get_epipole(F)
    epipole2 = get_epipole(F.T)
    # step 1: rotate image plane
    axis1 = get_rotation_axis(epipole1)
    theta1 = get_angle(epipole1, axis1)
    R1_plane = get_plane_rotation_matrix(axis1, theta1)

    conjugate = F @ axis1
    axis2 = get_rotation_axis(conjugate)
    theta2 = get_angle(epipole2, axis2)
    R2_plane = get_plane_rotation_matrix(axis2, theta2)

    # step2: align scanline
    new_epipole1 = R1_plane @ epipole1
    new_epipole2 = R2_plane @ epipole2

    phi1 = -np.arctan(new_epipole1[1] / new_epipole1[0])
    phi2 = -np.arctan(new_epipole2[1] / new_epipole2[0])
    R1_scanline = get_scanline_rotation_matrix(phi1)
    R2_scanline = get_scanline_rotation_matrix(phi2)

    # step 3: get homography
    H1 = R1_scanline @ R1_plane
    H2 = R2_scanline @ R2_plane
    H1 /= H1[-1,-1]
    H2 /= H2[-1,-1]
    if all:
        return H1, H2, R1_plane, R2_plane, R1_scanline, R2_scanline
    return H1, H2




