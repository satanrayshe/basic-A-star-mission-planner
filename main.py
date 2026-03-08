import numpy as np
from grid_manager import GridManager
from threat_engine import ThreatEngine
from pathfinder import find_path
from trajectory_optimizer import smooth_path
from viz_engine import plot_mission

GRID_SIZE = (20, 20)
START_POS = (0, 0)
GOAL_POS = (19, 19)
OBSTACLES = [(17, 0), (19, 17), (7, 0)]
THREAT_CENTER = (2, 10)
THREAT_RADIUS = 7
THREAT_COST = 5

env = GridManager(*GRID_SIZE)

for obs in OBSTACLES:
    env.set_obstacle(*obs)

ThreatEngine.inject_radar_bubble(env, THREAT_CENTER, THREAT_RADIUS, THREAT_COST)

raw_path = find_path(env.grid, START_POS, GOAL_POS)

if raw_path:
    smooth = smooth_path(raw_path)
    plot_mission(env.grid, raw_path, smooth)
else:
    print("Mission aborted: No viable path.")
