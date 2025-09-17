import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import path_planning
from parameters import Parameters

def plot_traj(sim_traj,ref_trajectory,boundary,obstacles,padded_obstacles):
    plt.figure()
    sim_traj = np.array(sim_traj)
    plt.plot(sim_traj[:,0], sim_traj[:,1],label='Simulated Trajectory')
    plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', label='Reference Trajectory')
    plt.plot(*zip(*boundary, boundary[0]), "k-")
    for hole in obstacles:
        plt.plot(*zip(*hole, hole[0]), "k-")
        # Plot a circle of radius 0.5 around each vertex of the hole
        for vertex in hole:
            circle = plt.Circle(vertex, 0.5, color='g', fill=False, linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
    for hole in padded_obstacles:
        plt.plot(*zip(*hole, hole[0]), "k-")    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Simulated Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

def plot_commands(commands):
    
    plt.figure()
    plt.title('Linear Velocity over time')
    plt.plot(commands[:,0], label='v')
    plt.legend(); plt.grid(); plt.show()

    plt.figure()
    plt.title('Angular velocity over time')
    plt.plot(commands[:,1], label='omega')
    plt.legend(); plt.grid(); plt.show()

def animate_dymobs(obstacles,boundary,p:Parameters):
    fig,ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    bx,by = zip(*boundary)
    ax.plot(list(bx)+[bx[0]], list(by)+[by[0]], 'k-')

    patches = []
    for i in range(len(obstacles)):
        x,y,x_rad,y_rad,angle = obstacles[i][0]
        ellipse = Ellipse((x, y), x_rad, y_rad, angle=np.degrees(angle), color='g', fill=True, linestyle='--', linewidth=1)
        ax.add_patch(ellipse)
        patches.append(ellipse)
    
    plt.ion()
    plt.show()
    dt = p.dt
    for i in range(max(len(obs) for obs in obstacles)):
        for index, ellipse in enumerate(patches):
            frame = min(i, len(obstacles[index])-1)
            x,y,x_rad,y_rad,angle = obstacles[index][frame]
            ellipse.set_center((x,y))
            ellipse.width,ellipse.height = 2*x_rad,2*y_rad
            ellipse.angle = np.degrees(angle)
        fig.canvas.draw_idle()
        plt.pause(dt)


def videoanim(boundary,static_obslist,dyn_obslist,sim_traj,p : Parameters,completed_len, opfile_path = 'mpcrun.gif'):
    sim_traj = np.stack(sim_traj, axis=0) #ensure right size
    simxy = sim_traj[:,:2] #pos and heading only
    T = len(simxy)
    dt = p.dt
    t0 = 0.0
    fps = int(round(1.0/dt))

    fig,ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal',adjustable='box')

    #Plot boundary
    bx,by = zip(*boundary)
    ax.plot(list(bx)+[bx[0]], list(by)+[by[0]], 'k-', lw=1.5)

    #Plot static obstacles
    for obs in static_obslist:
        sx,sy = zip(*obs)
        ax.plot(list(sx)+[sx[0]], list(sy)+[sy[0]], 'k-')

    robot_dot, = ax.plot([],[],'bo',ms=5)
    completed_path, = ax.plot([],[],'b--',lw=1)

    #dynamic obstacles

    ellipses = []
    obs0 = path_planning.get_dynobs_path_at_t(dyn_obslist,t0,p)
    for (x,y,x_rad,y_rad,ang) in obs0:
        ellipse = Ellipse((x, y), width=2*x_rad, height=2*y_rad, angle=np.degrees(ang),
                    fill=False, linestyle='--', linewidth=1, edgecolor='g')
        ax.add_patch(ellipse)
        ellipses.append(ellipse)
    
    #Helpers 
    def init():
        robot_dot.set_data([],[])
        completed_path.set_data([],[])

        return [robot_dot,completed_path,*ellipses]
    
    def update(k):
        t = t0 + k*dt
        robot_dot.set_data(sim_traj[k,0],sim_traj[k,1]) #select current point
        start = 0 if completed_path is None else max(0,k-completed_len) #Start so we have completed_len points left
        completed_path.set_data(sim_traj[:k+1,0],sim_traj[:k+1,1]) #select completed path until k

        #ellipse at time t

        obs = path_planning.get_dynobs_path_at_t(dyn_obslist,t,p)
        for ellipse, (x,y,x_rad,y_rad,angle) in zip(ellipses,obs):
            ellipse.set_center((x,y))
            ellipse.width, ellipse.height = 2*x_rad,2*y_rad
            ellipse.angle = np.rad2deg(angle)
        
        return [robot_dot, completed_path, *ellipses]
    
    anim = animation.FuncAnimation(fig,update, frames=T, init_func=init, interval=int(1000*dt),
                                   blit=False, repeat=False)

    writer = animation.PillowWriter(fps)
    anim.save(opfile_path, writer=writer)
    plt.close(fig)
    print(f'Saved {opfile_path} succesfully')
