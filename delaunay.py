import cv2
import numpy as np

def is_inner_trai(trai, p):
    a = 1e-30 + (-trai[1][1]*trai[2][0] + trai[0][1]*(-trai[1][0] + trai[2][0]) + trai[0][0]*(trai[1][1] - trai[2][1]) + trai[1][0]*trai[2][1])/2
    s = 1/(2*a)*(trai[0][1]*trai[2][0] - trai[0][0]*trai[2][1] + (trai[2][1] - trai[0][1])*p[0] + (trai[0][0] - trai[2][0])*p[1])
    t = 1/(2*a)*(trai[0][0]*trai[1][1] - trai[0][1]*trai[1][0] + (trai[0][1] - trai[1][1])*p[0] + (trai[1][0] - trai[0][0])*p[1])
    # return s > 0 and t > 0 and s + t < 1 # for excluding pixels on boundary
    return s >= 0 and t >= 0 and s + t <= 1 # pixels on boundary are included

def delaunay(img, points, corners=None, show=False):
    height, width, channel = img.shape

    # add points on boundary
    points.extend(corners)

    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    for point in points:
        subdiv.insert(point)
    trai_list = subdiv.getTriangleList()

    '''
    is_in_rect = lambda rect, point : point[0] >= rect[0] and point[0] <= rect[2] \
            and point[1] >= rect[1] and point[1] <= rect[3]

    trai_list_to_delete = []
    for i in range(len(trai_list)):
        if not is_in_rect(rect, (trai_list[i][0], trai_list[i][1])) or not is_in_rect(rect,
         (trai_list[i][2], trai_list[i][3])) or not is_in_rect(rect, (trai_list[i][4], trai_list[i][5])):
            trai_list_to_delete.append(i)
    trai_list = np.delete(trai_list, trai_list_to_delete, axis=0)

    if show:
        # draw delaunay
        img_copy = img.copy()
        for i, trai in enumerate(trai_list):
            p1 = (trai[0], trai[1])
            p2 = (trai[2], trai[3])
            p3 = (trai[4], trai[5])
            if is_in_rect(rect, p1) and is_in_rect(rect, p2) and is_in_rect(rect, p3):
                cv2.line(img_copy, p1, p2, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, p2, p3, (255, 245, 0), 1, cv2.LINE_AA, 0)
                cv2.line(img_copy, p3, p1, (255, 245, 0), 1, cv2.LINE_AA, 0)

        # draw points
        draw_points = lambda img, p, color: cv2.circle(img, p, 2, color, -1, cv2.LINE_AA, 0)

        for point in points:
            draw_points(img_copy, (int(point[0]), int(point[1])), (0, 0, 255))

        cv2.imshow("Delaunay", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    return points, trai_list.astype(np.uint32)