import cv2
import numpy as np

def homography_points(m_points, p_points):
    m_points = np.asarray(m_points, dtype=np.uint8)
    p_points = np.asarray(p_points, dtype=np.uint8)
    print(p_points.shape, m_points.shape)

    pts_src = []
    pts_dest = []
    for i in range(0, len(m_points)-1):
        pts_src.append([m_points[i, 0], m_points[i, 1]])
        pts_dest.append([p_points[i, 0], p_points[i, 1]])

    pts_src = np.asarray(pts_src, dtype=np.uint8)
    pts_dest = np.asarray(pts_dest, dtype=np.uint8)

    H_s, _ = cv2.findHomography(pts_src, pts_dest)
    return H_s