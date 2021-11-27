import json
import numpy as np
import cv2

def read_feature_points(fpath):
    json_raw = {}
    points = []
    with open(fpath.replace('.png', '.json').replace('.jpg', '.json'), 'r') as f:
        json_raw = json.load(f)
    for item in json_raw['shapes']:
        points.append([round(item['points'][0][0]), round(item['points'][0][1])])
    return points

def warpImage(img, H):
    # warp image while keeping whole images in view
    # ref: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    h,w = img.shape[:2]
    corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    corners_ = cv2.perspectiveTransform(corners, H)
    [xmin, ymin] = np.int32(corners_.min(axis=0).ravel() - 1.0)
    [xmax, ymax] = np.int32(corners_.max(axis=0).ravel() + 1.0)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin), flags=cv2.INTER_LINEAR)
    new_corners = cv2.perspectiveTransform(corners, Ht.dot(H))
    # append 4 more middle points
    mid_corners = []
    for i in range(3,-1,-1):
        mid_corners.append((new_corners[i] + new_corners[i-1])/2)
    # combine
    new_corners = list(np.squeeze(new_corners)) + list(np.squeeze(mid_corners))
    # convert all corners to tuple
    final_corners = []
    for cor in new_corners:
        final_corners.append((int(cor[0]), int(cor[1])))
    return result, final_corners

def drawPoints(img, pts):
    vis_img = img.copy()
    pts = np.squeeze(pts)
    # pts: N, 2
    for i in range(pts.shape[0]):
        cv2.circle(vis_img, (pts[i,0], pts[i,1]), radius=3, color=(0, 0, 255), thickness=-1)
    return vis_img
