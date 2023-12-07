import numpy as np
import math


def rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array(((c, -s), (s, c)))
    xy_arr = np.array(xy)
    return np.matmul(R,xy_arr)

def rotate_around(xy, pt, angle):
    xy = np.array(xy) - np.array(pt)
    xy = rotate(xy, angle)
    xy = xy + pt
    return xy