import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import path_planning
from parameters import Parameters
import imageio.v2 as imageio
from PIL import Image, ImageSequence
import os

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
    plt.savefig("postrun_plots/traj_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_commands(commands):
    
    plt.figure()
    plt.title('Linear Velocity over time')
    plt.plot(commands[:,0], label='v')
    plt.legend(); plt.grid(); 
    plt.savefig("postrun_plots/linvel_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.title('Angular velocity over time')
    plt.plot(commands[:,1], label='omega')
    plt.legend(); plt.grid();
    plt.savefig("postrun_plots/angvel_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

def animate_commands(commands,p:Parameters):
    print('Working on command gifs')
    # Linear Velocity
    fig,ax = plt.subplots()
    linvel, = ax.plot([],[],'b--',lw=1,label='v(t)')
    dt = p.dt
    T = len(commands)
    t = np.arange(T) * dt
    fps = int(round(1.0/dt))

    # axis limit stuff
    ax.set_xlim(0, t[-1] if T > 1 else p.dt)
    v = commands[:, 0]
    v_min, v_max = float(v.min()), float(v.max())
    pad = max(0.1 * (v_max - v_min), 0.1)  # Ensure padding is always positive
    if v_max == v_min:
        ax.set_ylim(v_min - pad, v_max + pad)
    else:
        ax.set_ylim(v_min - pad, v_max + pad)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('linear velocity [m/s]')
    ax.grid(True)
    ax.legend(loc='upper right')

    #Helpers
    def init_vel():
        linvel.set_data([],[])
        return [linvel]

    def update_vel(k):
        linvel.set_data(t[:k+1],v[:k+1]) #select until k
        return [linvel]
    
    animvel = animation.FuncAnimation(fig,update_vel, frames=T, init_func=init_vel, interval=int(1000*dt),
                                   blit=False, repeat=False)

    writer = animation.PillowWriter(fps)
    opfile_path = 'postrun_plots/linvel.gif'
    animvel.save(opfile_path, writer=writer)
    plt.close(fig)
    print(f'Saved linvel GIF succesfully')

    #Angular Velocity
    fig1,ax1 = plt.subplots()
    angvel, = ax1.plot([],[],'b--',lw=1,label='w(t)')
    ax1.set_xlim(0, t[-1] if T > 1 else p.dt)
    w = commands[:, 1]
    w_min, w_max = float(w.min()), float(w.max())
    pad = max(0.1 * (w_max - w_min), 0.1)  # Ensure padding is always positive
    if w_max == w_min:
        ax1.set_ylim(w_min - pad, w_max + pad)
    else:
        ax1.set_ylim(w_min - pad, w_max + pad)
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('angular velocity [rad/s]')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    #Helpers
    def init_angvel():
        angvel.set_data([],[])
        return [angvel]

    def update_angvel(k):
        angvel.set_data(t[:k+1],w[:k+1]) #select until k
        return [angvel]
    
    animangvel = animation.FuncAnimation(fig1,update_angvel, frames=T, init_func=init_angvel, interval=int(1000*dt),
                                   blit=False, repeat=False)

    writer = animation.PillowWriter(fps)
    opfile_path = 'postrun_plots/angvel.gif'
    animangvel.save(opfile_path, writer=writer)
    plt.close(fig1)
    print(f'Saved angvel GIF succesfully')



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


def videoanim(boundary,static_obslist,dyn_obslist,sim_traj,p : Parameters,completed_len, opfile_path = 'postrun_plots/mpcrun.gif'):
    print('Working on prdoucing GIF')
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

def view_gif_together(gif_paths,outpath='postrun_plots/together.gif'):

    #Create outpath 
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)  

    #Loading Helper
    def load_gif(path):
        im = Image.open(path)
        frames,durations = [],[]
        for f in ImageSequence.Iterator(im):
            frames.append(f.convert('RGBA'))
            durations.append(f.info.get('duration',im.info.get('duration',40)))
        return frames, durations
    
    gifs = [load_gif(gif) for gif in gif_paths]
    frames_list = [gif[0] for gif in gifs]
    duration_list = [gif[1] for gif in gifs]
    lengths = [len(frame) for frame in frames_list]

    #stop when first ends
    N = min(lengths)
    idxmaps = [list(range(N)) for _ in gif_paths]

    height = frames_list[0][0].height
    
    #Helper for resize
    def resize_byh(frame,h):
        w = int(round(frame.width * (h / frame.height)))
        return frame.resize((w,h), Image.BICUBIC)
    
    #Resize all the frames
    frames_list = [
        [resize_byh(frame, height) for frame in frames] 
        for frames in frames_list
    ]

    #Put together frames
    fps = 10
    dur_ms = max(fps,int(round(1000.0 / fps)))
    op_frames = []
    durations = [dur_ms] * N
    bg = (255,255,255)

    for i in range(N):
        frames = [frames_list[g][idxmaps[g][i]] for g in range(len(gif_paths))] #doubleloop thru to select frames

        w = sum(frame.width for frame in frames) #canvas width
        h = height

        canvas = Image.new('RGBA',(w,h),(*bg, 255))
        x_cursor = 0
        
        for frame in frames:
            canvas.paste(frame,(x_cursor,0), frame)
            x_cursor += frame.width
        
        op_frames.append(canvas.convert('P',palette = Image.ADAPTIVE))
        

    
    #save as GIF
    op_frames[0].save(outpath,save_all = True,
                       append_images = op_frames[1:],
                       duration = durations,
                       loop = 0,
                       disposal = 2,
                       optimize = False)
    
    return outpath




    
