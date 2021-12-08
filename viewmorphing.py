import os
import dlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import linalg, optimize
from math import sin, cos, asin, atan, pi, sqrt, floor, ceil

import utils
from feature_detection import feature_points_detection
from prewarp import compute_prewarp
from postwarp import computeH
from computeF import computeF
from delaunay import delaunay
from morph import solve_affine, apply_affine, morphing


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='View Morphing')
    parse.add_argument('-o', dest='outdir', type=str, default='./output', help='output or intermediate results dir')
    parse.add_argument('-i', dest='input', type=str, default='./images/part2-2/source_1.png', help='input image file location')
    parse.add_argument('-t', dest='target', type=str, default = './images/part2-2/target_1.png', help='input target file location')
    parse.add_argument('-s', dest='sequence', type=int, default = 5, help='length of the morphing sequence')
    parse.add_argument('-a', dest='auto', default=False, help='use feature points label or dlib automatical detection')
    parse.add_argument('-x', dest='shiftx', type=int, default = 5, help='shift x pixels for image after prewarping')
    parse.add_argument('-y', dest='shifty', type=int, default = 200, help='shift y pixels for image after prewarping')
    args = parse.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if not args.input or not args.target:
        print("Invalid parameters")
        exit(-1)

    if not isinstance(args.sequence, int) or args.sequence <= 0:
        print("Sequence should be an positive integr")
        exit(-1)

    input, target = cv2.imread(args.input), cv2.imread(args.target)
    #input, target  = cv2.resize(input, dsize=(386, 340)), cv2.resize(target, dsize=(382, 340))

    # extract feature points manually
    if not args.auto:
        input_points, target_points = utils.read_feature_points(args.input), utils.read_feature_points(args.target)
    else:
        # extract feature points automatically
        input_points_auto, target_points_auto = feature_points_detection(input, show=False), feature_points_detection(target, show=False)
        eigth_points = [18, 27, 40, 43, 49, 55, 6, 12] # for test3
        input_points_eight = [input_points_auto[p] for p in eigth_points]
        target_points_eight = [target_points_auto[p] for p in eigth_points]
        input_points, target_points = input_points_eight, target_points_eight

    # pre-warping
    F = computeF(input_points, target_points)
    H0, H1= compute_prewarp(F)
    input_prewarping, input_corners, Ht0 = utils.warpImage(input, H0, True)
    target_prewarping, target_corners, Ht1 = utils.warpImage(target, H1, True)
    # utils.displayEpipolarF(input_prewarping, target_prewarping, np.linalg.inv(Ht1 @ H1).T @ F @ np.linalg.inv(Ht0 @ H0))
    input_prewarping, input_corners = utils.warpImage(input, H0)
    target_prewarping, target_corners = utils.warpImage(target, H1)

    cv2.imwrite(os.path.join(args.outdir, args.input.split('/')[-1].split('.')[0] + '_prewarping.png'), input_prewarping)
    cv2.imwrite(os.path.join(args.outdir, args.target.split('/')[-1].split('.')[0] + '_prewarping.png'), target_prewarping)

    # features detection
    input_prewarping_points = feature_points_detection(input_prewarping, show=False)
    target_prewarping_points = feature_points_detection(target_prewarping, show=False)

    # delaunay triangulation
    points_input, trai_input = delaunay(input_prewarping, input_prewarping_points, input_corners, show=False)
    points_target, trai_target = delaunay(target_prewarping, target_prewarping_points, target_corners, show=False)
    
    # plot input image
    plt.figure(figsize=(20, 12))
    plt.subplot(1, args.sequence + 2, 1)
    plt.imshow(input[:, :, ::-1])
    plt.title('Origin Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])

    # view morphing and postwarping
    for i in range(args.sequence):
        alpha = round((i + 1) / (args.sequence + 1), 2)
        print("working on alpha: %f..."%(alpha))
        res = morphing(input_prewarping, points_input, trai_input, target_prewarping, points_target, trai_target, alpha)

        # control points
        control_points_four = [(int((1 - alpha) * input_corners[p][0] + alpha * target_corners[p][0]),
                             int((1 - alpha) * input_corners[p][1] + alpha * target_corners[p][1])) for p in range(4)]

        new_height = max(input.shape[0], target.shape[0])
        new_width = max(input.shape[1], target.shape[1])
        postwarping_four = [(0, 0), (0, new_height - 1), (new_width - 1, new_height-1), (new_width - 1, 0)]

        # find homography using the points
        H_s = computeH(np.array(postwarping_four).T, np.array(control_points_four).T)
        H_s /= H_s[-1,-1]
        # warp image to desired plane
        res = cv2.warpPerspective(res, H_s, (new_width, new_height), flags=cv2.INTER_LINEAR)

        plt.subplot(1, args.sequence + 2, i + 2)
        plt.imshow(res[:, :, ::-1])
        plt.title('Morphing \\alpha={:.2f}'.format(alpha), fontsize=12)
        plt.xticks([])
        plt.yticks([])
    
    # plot target image
    plt.subplot(1, args.sequence + 2, args.sequence + 2)
    plt.imshow(target[:, :, ::-1])
    plt.title('Target Image', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()
