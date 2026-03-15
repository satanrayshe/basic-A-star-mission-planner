import numpy as np
from grid_manager import GridManager
from threat_engine import ThreatEngine
from pathfinder import find_path
from vector_field import generate_integration_field, generate_vector_field, trace_path
from viz_engine import plot_mission

# config
DIMS = 20, 20
ST, GL = (0, 0), (19, 19)
WALLS = [(5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5), (5, 6), (5, 7), (5, 8)]
RAD_X, RAD_Y = 10, 15
RAD_R = 5
D_VAL = 5

env = GridManager(*DIMS)
for w in WALLS:
    env.set_obstacle(*w)
    
# threats
ThreatEngine.inject_radar_bubble(env, (RAD_X, RAD_Y), RAD_R, D_VAL)
ThreatEngine.inject_wall_repulsion(env, pad=10.0)

# 1. A* ref
old_p = find_path(env.g, ST, GL)

# 2. Flow field
f = generate_integration_field(env.g, GL)
ux, uy = generate_vector_field(f)
flow_p = trace_path(ST, GL, ux, uy, step=0.2)

if old_p:
    plot_mission(env.g, old_p, flow_p, ux, uy)
else:
    print("No route found.")
