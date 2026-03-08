import numpy as np
from scipy.interpolate import CubicSpline

def smooth_path(path, num_points=100):
    """smooths a jagged A* path using Cubic Splines"""
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]

    t = np.arange(len(path))

    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    t_new = np.linspace(0, len(path) - 1, num_points)

    x_smooth = cs_x(t_new)
    y_smooth = cs_y(t_new)

    return np.vstack((x_smooth, y_smooth)).T
