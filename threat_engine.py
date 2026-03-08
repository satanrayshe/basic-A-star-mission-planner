import numpy as np

class ThreatEngine:
    @staticmethod
    def inject_radar_bubble(grid_manager, center, radius, threat_cost):
        """injects a circular high-cost zone (radar bubble) into the grid.

        Args:
            grid_manager: The instance of GridManager.
            center: tuple (x, y)
            radius: int (distance in grid units)
            threat_cost: float (weight applied inside the bubble)"""
        y, x = np.ogrid[:grid_manager.height, :grid_manager.width]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        mask = dist_from_center <= radius

        grid_manager.grid[mask] = np.maximum(grid_manager.grid[mask], threat_cost)

    @staticmethod
    def inject_no_fly_zone(grid_manager, top_left, bottom_right):
        """injects a rectangular no-fly zone (infinite cost)"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        grid_manager.grid[y1:y2, x1:x2] = float('inf')
