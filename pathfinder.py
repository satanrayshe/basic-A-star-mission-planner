import heapq
import numpy as np

def heuristic(a, b):
    """calculates Manhattan distance for grid-based movement"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, grid_shape):
    """returns valid grid neighbors (Up, Down, Left, Right)"""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
            neighbors.append((nx, ny))
    return neighbors

def find_path(grid, start, goal):
    """performs A* search.
    Args:
        grid: np.ndarray (the cost map)
        start: tuple (x, y)
        goal: tuple (x, y)
    Returns:
        List of coordinates constituting the path, or None if no path exists"""
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current, grid.shape):
            cost = grid[neighbor[0], neighbor[1]]

            if np.isinf(cost):
                continue

            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None
