import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import casadi as ca
import opengen as og
import path_planning
from parameters import Parameters
import plotting
import mpcopEn
import yaml

if __name__ == '__main__':
    config = 'test_config2'
    path, obstacles, boundary, padded_obstacles = path_planning.gen_path(config)
    dynobs = [([1.0, 4.0], [2.0, 7.0], 0.1, 0.2, 0.5, 0.1)]
    p = Parameters(obstacles,boundary,dynobs)
    ref_trajectory = path_planning.generate_reftrajectory(p,path)
    sim_traj = mpcopEn.run_mpc(p,ref_trajectory)
    len_of_prevpath = 0
    boundary = p.boundaries
    obstacles = p.obstacles
    dynobs = p.dynobs

    #Animate trajectory
    plotting.videoanim(boundary,obstacles,dynobs,sim_traj,p,len_of_prevpath)
