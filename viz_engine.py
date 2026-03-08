import matplotlib.pyplot as plt
import numpy as np

def plot_mission(grid, path, smoothed_path):
    """renders the environment, A* path, and smoothed trajectory"""
    plt.figure(figsize=(10, 8))

    plt.imshow(grid, cmap='viridis', origin='lower', alpha=0.6)

    path_arr = np.array(path)
    plt.plot(path_arr[:, 1], path_arr[:, 0], 'r--', label='Raw A* Path', alpha=0.5)

    plt.plot(smoothed_path[:, 1], smoothed_path[:, 0], 'b-', linewidth=2, label='Smoothed Trajectory')

    plt.legend()
    plt.title("Tactical Mission Plan: Optimal Path vs Flyable Trajectory")
    plt.show()
