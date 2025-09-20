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
    oscx = 8.17127
    oscy = 29.0021
    dynobs = [([oscx, oscy], [oscx, oscy+1], 0.1, 0.2, 0.5, 0.1)] #p1,p2,freq,x_rad,y_rad, seg_heading(rads)
    p = Parameters(obstacles,boundary,dynobs)
    ref_trajectory = path_planning.generate_reftrajectory(p,path)
    sim_traj, commands = mpcopEn.run_mpc(p,ref_trajectory)
    len_of_prevpath = 0
    boundary = p.boundaries
    obstacles = p.obstacles
    dynobs = p.dynobs
    plotting.videoanim(boundary,obstacles,dynobs,sim_traj,p,len_of_prevpath)
    plotting.animate_commands(commands,p)
    #Animate trajectory
    plotting.plot_traj(sim_traj,ref_trajectory,boundary,obstacles,padded_obstacles)
    plotting.plot_commands(commands)
    paths = ['postrun_plots/linvel.gif','postrun_plots/angvel.gif','postrun_plots/mpcrun.gif']
    plotting.view_gif_together(paths)
    np.savetxt('straj.txt', sim_traj, delimiter=' ')
    