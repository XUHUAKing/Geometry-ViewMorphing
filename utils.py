import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prewarp import get_epipole

def normalize(pts):
    '''
    Input: pts: list of tuple or np.array n x 2
    Apply mean normalize for stability
    Ouput: normalized_pts: np.array n x 3
    '''
    centriod = np.mean(pts, axis=0)
    assert len(centriod) == 2
    pts_offset = pts - centriod

    scale = np.sqrt(2) / np.mean(np.sqrt(pts_offset[:, 0] ** 2 + pts_offset[:, 1] ** 2))
    # transform pts using
    transformation = np.diag([scale, scale, 1])
    transformation[0, 2] = -scale * centriod[0]
    transformation[1, 2] = -scale * centriod[1]

    homogenous_pts = np.hstack([pts, np.ones([len(pts), 1])])
    normalized_pts = (transformation @ homogenous_pts.T).T
    # return as len(pts) x 3
    assert len(normalized_pts) == len(pts)
    return normalized_pts, transformation

def read_feature_points(fpath):
    json_raw = {}
    points = []
    with open(fpath.replace('.png', '.json').replace('.jpg', '.json'), 'r') as f:
        json_raw = json.load(f)
    for item in json_raw['shapes']:
        points.append([round(item['points'][0][0]), round(item['points'][0][1])])
    return points

def warpImage(img, H, full=False):
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
    if full:
        return result, final_corners, Ht
    return result, final_corners

def drawPoints(img, pts):
    vis_img = img.copy()
    pts = np.squeeze(pts)
    # pts: N, 2
    for i in range(pts.shape[0]):
        cv2.circle(vis_img, (pts[i,0], pts[i,1]), radius=3, color=(0, 0, 255), thickness=-1)
    return vis_img

def displayEpipolarF(I1, I2, F):
    e1 =  get_epipole(F)
    e2 =  get_epipole(F.T)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))

    ax1.imshow(np.array(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)))
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(np.array(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)))
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()
    ax2.autoscale(enable=False, axis='both')
    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = x
        yc = y
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            raise Exception('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2)
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()