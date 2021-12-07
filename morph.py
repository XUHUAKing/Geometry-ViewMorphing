# TODO: Xiaochen Han

import numpy as np
import cv2
from scipy import linalg
from math import sin, cos, asin, atan, pi, sqrt, floor, ceil

def apply_affine(input, target, res, t1, t2, t, alpha):
    # get bounding boxes
    bb1 = list(cv2.boundingRect(t1))
    bb1[2], bb1[3] = bb1[0] + bb1[2], bb1[1] + bb1[3]
    bb2 = list(cv2.boundingRect(t2))
    bb2[2], bb2[3] = bb2[0] + bb2[2], bb2[1] + bb2[3]
    bb = list(cv2.boundingRect(t))
    bb[2], bb[3] = bb[0] + bb[2], bb[1] + bb[3]

    # shift bounding boxes, make it starting from the upper left corner
    t1[:, 0] -= bb1[0]
    t1[:, 1] -= bb1[1]

    t2[:, 0] -= bb2[0]
    t2[:, 1] -= bb2[1]

    t[:, 0] -= bb[0]
    t[:, 1] -= bb[1]

    # solve affine between each triangle pairs via Af=0, from input to res and target to res both sides

    t1_to_t = cv2.getAffineTransform(np.float32(t1), np.float32(t))
    t2_to_t = cv2.getAffineTransform(np.float32(t2), np.float32(t))

    inputRect = input[bb1[1]: bb1[3], bb1[0]: bb1[2]]
    targetRect = target[bb2[1]: bb2[3], bb2[0]: bb2[2]]

    warpInput = cv2.warpAffine(inputRect, t1_to_t, (bb[2] - bb[0], bb[3] - bb[1]), None, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
    warpTarget = cv2.warpAffine(targetRect, t2_to_t, (bb[2] - bb[0], bb[3] - bb[1]), None, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

    imgRect = (1.0 - alpha) * warpInput + alpha * warpTarget
    mask = np.zeros(imgRect.shape, dtype=np.float32)
    cv2.fillConvexPoly(mask, t, (1.0, 1.0, 1.0), 16, 0)
    # mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)

    res[bb[1]: bb[3], bb[0]: bb[2]] = res[bb[1]: bb[3], bb[0]: bb[2]] * (1 - mask) + imgRect * mask

    return res


def morphing(input, points_input, trai_input, target, points_target, trai_target, alpha=0.5):
    res = np.zeros((max(input.shape[0], target.shape[0]), max(input.shape[1], target.shape[1]), 3))

    # target location
    points = []
    for i in range(len(points_input)):
        x = (1 - alpha) * points_input[i][0] + alpha * points_target[i][0]
        y = (1 - alpha) * points_input[i][1] + alpha * points_target[i][1]
        points.append((round(x), round(y)))

    for i, item in enumerate(trai_input):
        v1 = points_input.index((item[0], item[1]))
        v2 = points_input.index((item[2], item[3]))
        v3 = points_input.index((item[4], item[5]))

        t1 = np.array([points_input[v1], points_input[v2], points_input[v3]])
        t2 = np.array([points_target[v1], points_target[v2], points_target[v3]])
        t = np.array([points[v1], points[v2], points[v3]])
        t = t.astype(np.int32)

        # apply affine for the exact triangle
        res = apply_affine(input, target, res, t1, t2, t, alpha)

        # save morphing process for visualization
        #if alpha == 0.5:
        #    cv2.imwrite("viz/" + str(i) + ".jpg", res)

    res = res.astype(np.uint8)
    return res
