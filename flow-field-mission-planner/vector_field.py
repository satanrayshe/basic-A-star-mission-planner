import numpy as np
import heapq

try:
    from numba import njit
except:
    def njit(f): return f

RT2 = 1.414

def get_adj(p, size):
    # x,y neighbors
    x, y = p
    h, w = size[0], size[1]
    
    if y < h - 1: yield ((x, y + 1), 1.0)
    if y > 0:     yield ((x, y - 1), 1.0)
    if x < w - 1: yield ((x + 1, y), 1.0)
    if x > 0:     yield ((x - 1, y), 1.0)
    
    # diagonals
    if x < w - 1 and y < h - 1: yield ((x + 1, y + 1), RT2)
    if x < w - 1 and y > 0:     yield ((x + 1, y - 1), RT2)
    if x > 0 and y < h - 1:     yield ((x - 1, y + 1), RT2)
    if x > 0 and y > 0:         yield ((x - 1, y - 1), RT2)

def generate_integration_field(g, goal):
    # cost to goal from everywhere
    try:
        import skfmm
        f = np.ones_like(g)
        f[goal[1], goal[0]] = 0
        
        m = np.isinf(g)
        phi = np.ma.MaskedArray(f, m)
        speed = np.where(m, 1.0, 1.0 / np.maximum(g, 0.001))
        
        f = skfmm.travel_time(phi, speed)
        return f.filled(np.inf)
    except:
        return _slow_field(g, goal)

def _slow_field(g, goal):
    f = np.full(g.shape, np.inf)
    f[goal[1], goal[0]] = 0  
    q = [(0, goal)]
    
    while q:
        cost, curr = heapq.heappop(q)
        if cost > f[curr[1], curr[0]]: continue
            
        for nxt, dist in get_adj(curr, g.shape):
            val = g[nxt[1], nxt[0]]
            if np.isinf(val): continue
                
            new_c = cost + val * dist
            if new_c < f[nxt[1], nxt[0]]:
                f[nxt[1], nxt[0]] = new_c
                heapq.heappush(q, (new_c, nxt))
    return f

def generate_vector_field(f):
    # gradient downhill
    h, w = f.shape
    uy, ux = np.zeros((h, w)), np.zeros((h, w))
    
    pad = np.pad(f, 1, constant_values=np.inf)
    
    # stack 8 neighbors
    nb = np.stack([
        pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:],
        pad[1:-1, :-2], pad[1:-1, 2:],
        pad[2:, :-2], pad[2:, 1:-1], pad[2:, 2:]
    ])
    
    vecs = np.array([
        [-1, -1], [ 0, -1], [ 1, -1],
        [-1,  0],           [ 1,  0],
        [-1,  1], [ 0,  1], [ 1,  1]
    ], dtype=float)
    
    mags = np.linalg.norm(vecs, axis=1, keepdims=True)
    dirs = vecs / mags
    
    idx = np.argmin(nb, axis=0)
    v_min = np.min(nb, axis=0)
    
    ok = (v_min < f) & (~np.isinf(f))
    
    chosen_m = mags[idx, 0]
    grad = np.zeros_like(f)
    grad[ok] = (f[ok] - v_min[ok]) / chosen_m[ok]
    
    dx = dirs[idx, 0]
    dy = dirs[idx, 1]
    
    ux[ok] = dx[ok] * grad[ok]
    uy[ok] = dy[ok] * grad[ok]
        
    return ux, uy

@njit
def _run_trace(sx, sy, gx, gy, ux, uy, step, limit):
    h, w = ux.shape
    px = [sx]
    py = [sy]
    x, y = sx, sy
    
    for _ in range(limit):
        dist = ((x - gx)**2 + (y - gy)**2)**0.5
        if dist < 0.5:
            px.append(gx)
            py.append(gy)
            break
            
        x_c = min(max(x, 0.0), w - 1.001)
        y_c = min(max(y, 0.0), h - 1.001)
        
        x0, y0 = int(x_c), int(y_c)
        x1, y1 = x0 + 1, y0 + 1
        
        dx, dy = x_c - float(x0), y_c - float(y0)
        
        vx1 = ux[y0, x0] * (1.0 - dx) + ux[y0, x1] * dx
        vx2 = ux[y1, x0] * (1.0 - dx) + ux[y1, x1] * dx
        vxv = vx1 * (1.0 - dy) + vx2 * dy
        
        vy1 = uy[y0, x0] * (1.0 - dx) + uy[y0, x1] * dx
        vy2 = uy[y1, x0] * (1.0 - dx) + uy[y1, x1] * dx
        vyv = vy1 * (1.0 - dy) + vy2 * dy
        
        mag = (vxv**2 + vyv**2)**0.5
        if mag == 0: break
            
        x += (vxv / mag) * step
        y += (vyv / mag) * step
        
        px.append(x)
        py.append(y)
        
    return px, py

def trace_path(start, goal, ux, uy, step=0.1, limit=5000):
    # follow the field
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    
    x_list, y_list = _run_trace(sx, sy, gx, gy, ux, uy, step, limit)
    
    path = np.zeros((len(x_list), 2))
    path[:, 0] = x_list
    path[:, 1] = y_list
    return path
