import numpy as np
from scipy.interpolate import CubicSpline

def smooth_path(p_list, steps=100):
    # spline smoothing for the jagged path
    pts = np.array(p_list)
    px, py = pts[:, 0], pts[:, 1]
    
    idx = np.arange(len(pts))
    sx = CubicSpline(idx, px)
    sy = CubicSpline(idx, py)
    
    idx_new = np.linspace(0, len(pts) - 1, steps)
    ox = sx(idx_new)
    oy = sy(idx_new)
    return np.vstack((ox, oy)).T
