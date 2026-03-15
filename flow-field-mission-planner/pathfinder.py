import heapq
import numpy as np

def score(p1, p2):
    # manhattan
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def adj_cells(p, size):
    res = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = p[0] + dx, p[1] + dy
        if 0 <= nx < size[0] and 0 <= ny < size[1]:
            res.append((nx, ny))
    return res

def find_path(g, start, goal):
    # standard a-star
    heap = []
    heapq.heappush(heap, (0, start))
    came_from = {}
    dists = {start: 0}

    while heap:
        _, curr = heapq.heappop(heap)
        
        if curr == goal:
            path = []
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            return path[::-1]

        for nxt in adj_cells(curr, g.shape):
            val = g[nxt[0], nxt[1]]
            if np.isinf(val): continue
            
            d = dists[curr] + val
            if d < dists.get(nxt, float('inf')):
                came_from[nxt] = curr
                dists[nxt] = d
                f = d + score(nxt, goal)
                heapq.heappush(heap, (f, nxt))
    return None
