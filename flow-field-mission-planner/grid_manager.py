import numpy as np
import matplotlib.pyplot as plt

class GridManager:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.g = np.ones((h, w), dtype=float)

    def set_obstacle(self, x, y):
        # kill zone
        if 0 <= x < self.w and 0 <= y < self.h:
            self.g[y, x] = float('inf')

    def set_threat(self, x, y, val):
        # radar/threat stuff
        if x < 0 or x >= self.w or y < 0 or y >= self.h: return
        self.g[y, x] = val

    def visualize(self):
        plt.imshow(self.g, cmap='viridis', origin='lower')
        plt.colorbar(label="Cost Weight")
        plt.title("Operational Environment")
        plt.show()

if __name__ == '__main__':
    env = GridManager(20, 20)
    env.set_obstacle(5, 5)
    env.set_threat(10, 10, 5)
    env.visualize()
