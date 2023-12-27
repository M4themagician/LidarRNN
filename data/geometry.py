import numpy as np
import math
import random


def rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array(((c, -s), (s, c)))
    xy_arr = np.array(xy)
    return np.matmul(R,xy_arr)

def rotate_around(xy, center_of_rot, angle):
    xy = np.array(xy) - np.array(center_of_rot)
    xy = rotate(xy, angle)
    xy = xy + np.array(center_of_rot)
    return xy

def random_number(range):
    return range[0] + (range[1]-range[0])*random.random()


if __name__ == "__main__":
    xy = np.array([1, 0])
    xy_rot = rotate(xy, np.pi)
    assert np.allclose(xy_rot, -xy, atol=1e-9)

    xy = np.random.uniform(-10, 10, 2)
    rot_center = np.random.uniform(-10, 10, 2)
    angle = np.random.uniform(0, 2*np.pi)
    norm_diff = np.linalg.norm(xy - rot_center) - np.linalg.norm(rotate_around(xy, rot_center, angle) - rot_center)
    assert abs(norm_diff) < 1e-7, f"Rotated and original should be equal in norm, but got difference in norm of {norm_diff}."