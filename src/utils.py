import numpy as np
from shapely.geometry import Polygon

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def box2corners_2d(box):
    # box -> corners
    # x, y, theta, l, w = box
    x, y, l, w, theta = box
    sin_s = np.sin(theta)
    cos_s = np.cos(theta)
    tmp = np.array([[-l / 2, -w / 2],
                    [l / 2, -w / 2],
                    [l / 2, w / 2],
                    [-l / 2, w / 2]]).T
    Rmat = np.array([[cos_s, -sin_s], [sin_s, cos_s]])
    corners = np.array([[x], [y]]) + np.dot(Rmat, tmp)
    corners = corners.T
    return corners

def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def iou_2d(corners1, corners2, criterion='union'):
    ''' Compute 2D bounding box IoU.
        Input:
            corners1: numpy array (4,2)
            corners2: numpy array (4,2)
        Output:
            iou_2d: bird's eye view 2D bounding box IoU
        '''
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(0, 4, 1)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(0, 4, 1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    polygon = Polygon(rect1)
    other_polygon = Polygon(rect2)
    intersection = polygon.intersection(other_polygon)
    inter_area = intersection.area
    if criterion.lower() == 'union':
        iou_2d = inter_area / (area1 + area2 - inter_area)
    elif criterion.lower() == 'a':
        iou_2d = inter_area / area1
    else:
        raise TypeError("Unkown type for criterion")
    return iou_2d