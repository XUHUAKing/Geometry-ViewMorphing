import numpy as np
import cv2

def computeH(p1, p2):
    """
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2
    #############################
    # TO DO ...
    n = p1.shape[1]
    homo_p2 = np.concatenate((p2, np.ones((1, n))), axis=0).T

    seg1 = np.zeros((2 * n, 3))
    seg1[::2] = -homo_p2

    seg2 = np.zeros((2 * n, 3))
    seg2[1::2] = -homo_p2

    pp2 = np.repeat(p2.T, 2, axis=0)
    col_p1 = p1.T.flatten()
    pp1 = np.repeat(col_p1[np.newaxis, :], 2, 0).T

    A = np.concatenate((seg1, seg2, pp1 * pp2, col_p1.reshape(-1, 1)), axis=1)
    # print("A", A.shape)

    e_value, e_vector = np.linalg.eig(np.dot(A.T, A))
    H2to1 = e_vector[:, np.argmin(e_value)]
    H2to1 = H2to1.reshape((3, 3))

    return H2to1
