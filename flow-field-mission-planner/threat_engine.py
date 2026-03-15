import numpy as np

class ThreatEngine:

    @staticmethod
    def inject_radar_bubble(env, p, rad, max_c):
        # circular cost zone with decay
        h, w = env.h, env.w
        y, x = np.ogrid[:h, :w]
        
        # dist squared
        d2 = ((x - p[0]) ** 2 + (y - p[1]) ** 2) + 0.1
        mask = np.sqrt(d2) <= rad
        
        # inv square: cost = max / d^2
        cost = max_c / d2[mask]
        env.g[mask] = np.maximum(env.g[mask], cost)

    @staticmethod
    def inject_no_fly_zone(env, p1, p2):
        # hard block for a box
        x1, y1 = p1
        x2, y2 = p2
        env.g[y1:y2, x1:x2] = float('inf')

    @staticmethod
    def inject_wall_repulsion(env, pad=10.0):
        # aura around walls to stop corner cutting
        walls = np.isinf(env.g)
        tmp = np.zeros_like(walls)
        
        # 8 direction shift
        tmp[:-1, :] |= walls[1:, :]   
        tmp[1:, :] |= walls[:-1, :]   
        tmp[:, :-1] |= walls[:, 1:]   
        tmp[:, 1:] |= walls[:, :-1]   
        tmp[:-1, :-1] |= walls[1:, 1:] 
        tmp[1:, 1:] |= walls[:-1, :-1] 
        tmp[:-1, 1:] |= walls[1:, :-1] 
        tmp[1:, :-1] |= walls[:-1, 1:] 
        
        mask = tmp & (~walls)
        env.g[mask] = np.maximum(env.g[mask], pad)
