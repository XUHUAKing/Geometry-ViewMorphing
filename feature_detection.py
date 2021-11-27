import cv2
import dlib
import numpy as np
from imutils import face_utils
from utils import normalize

pathdir = './detection/shape_predictor_68_face_landmarks.dat'

def feature_points_detection(im, show=False):
    '''
    Input: im: np.array h x w x 3
    Using dlib for 68 points detection
    Ouput: tuple_shape: list of tuple 68 x 2
    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pathdir)

    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    detects = detector(gray_im, 1)
    print("Detect face num", len(detects))
    assert len(detects) > 0, "NO face detected"
    for (i, detect) in enumerate(detects):
        shape = predictor(gray_im, detect)
        # return the list of (x, y)-coordinates
        shape_np = face_utils.shape_to_np(shape)
        assert shape_np.shape == (68, 2)

    if show:
        im_new = im.copy()
        for i in range(len(shape_np)):
            cv2.circle(im_new, tuple(shape_np[i]), 2, (0, 255, 0), -1)
        cv2.imshow("feature points", im_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # to be consistent
    tuple_shape = list(map(tuple, shape_np))
    return tuple_shape


if __name__ == '__main__':
    image1 = cv2.imread('./images/part2-2/source_3.png')
    image2 = cv2.imread('./images/part2-2/target_3.png')
    shape_1 = feature_points_detection(image1, show=True)