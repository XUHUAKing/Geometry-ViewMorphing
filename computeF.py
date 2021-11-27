import numpy as np
import cv2

"""redirect the centroid of points to origin"""
def normalize(points):
    points = np.array(points)
    mean_xy = np.mean(points, axis=0)
    shifted_points = points - mean_xy

    # transform distance form origin to \sqrt{2}
    scale_f = np.sqrt(2) / np.mean(np.sqrt(shifted_points[:, 0] ** 2 + shifted_points[:, 1] ** 2))
    t_norm = np.array([[scale_f, 0, -scale_f * mean_xy[0]],
                       [0, scale_f, -scale_f * mean_xy[1]],
                       [0, 0, 1]
                       ])
    homo_points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    norm_points = np.matmul(t_norm, homo_points.T).T

    return norm_points, t_norm


"""compute fundamental matrix using 8 points"""
def computeF(pts1, pts2):
    pts1, t1 = normalize(pts1)
    pts2, t2 = normalize(pts2)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    # denormalize
    F = t2.T @ F @ t1
    return F

"""Find F using RANSAC"""
# def compute_fundamental_matrix(pts1, pts2, t1, t2, nIters=1000, tol=0.42):
#     pts1 = np.array(pts1)
#     pts2 = np.array(pts2)
#     N = pts1.shape[0]
#
#     max_cnt, curr_itr = 0, 0
#     bestF = np.zeros((3, 3))
#     best_inliers = None
#
#     while curr_itr < nIters:
#         # randomly sample 8 points to estimate a F
#         indices = np.random.choice(np.arange(N), size=8, replace=False)
#         F, _ = cv2.findFundamentalMat(pts1[indices, :], pts2[indices, :], cv2.FM_8POINT)
#         # find epipolar lines on image 2
#         L2s = np.matmul(pts1, F.T)
#         # distance from lines to points pts2
#         diffs = np.abs(np.sum(L2s * pts2, axis=1)) / np.sqrt(np.sum(L2s[:, :2] ** 2, axis=1))
#         # count inliers and update best F
#         inliers = (diffs <= tol) * 1
#         cnt = np.sum(inliers)
#         if cnt > max_cnt:
#             # update best F
#             max_cnt = cnt
#             bestF = F
#             best_inliers = inliers
#
#         curr_itr += 1
#
#     # denormalize
#     bestF = t2.T@bestF@t1
#     return  bestF
