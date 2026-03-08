import numpy as np
import matplotlib.pyplot as plt

class GridManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width), dtype=float)

    def set_obstacle(self, x, y):
        """sets a coordinate as impassable (Infinite cost)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = float('inf')

    def set_threat(self, x, y, threat_level):
        """sets a coordinate with a specific threat cost"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = threat_level

    def visualize(self):
        """renders the grid state"""
        plt.imshow(self.grid, cmap='viridis', origin='lower')
        plt.colorbar(label='Cost Weight')
        plt.title("Operational Environment (Cost Map)")
        plt.show()

env = GridManager(20, 20)
env.set_obstacle(5, 5)
env.set_threat(10, 10, 5)
