import matplotlib.pyplot as plt
import numpy as np

def plot_mission(env_g, path, flow, ux, uy):
    # quick plot of the map
    plt.figure(figsize=(10, 8))
    
    # environment cost map
    plt.imshow(env_g, cmap='viridis', origin='lower', alpha=0.6, vmax=10)
    
    # vectors
    h, w = env_g.shape
    y, x = np.mgrid[0:h, 0:w]
    plt.quiver(x, y, ux, uy, color="white", alpha=0.5, scale=30)
    
    # A* ref
    p_arr = np.array(path)
    plt.plot(p_arr[:, 1], p_arr[:, 0], 'r--', label='A* (Raw)', alpha=0.5)
    
    # Flow Field
    plt.plot(flow[:, 0], flow[:, 1], 'b-', linewidth=2, label='Flow Field')
    
    plt.legend()
    plt.title("Tactical Mission Map")
    plt.savefig('mission_plot.png')
    plt.show()
