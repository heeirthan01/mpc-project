import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import opengen as og
import path_planning
import yaml

class Parameters:
    def __init__(self, obstacles,boundaries):
        self.vel_min = -0.5
        self.vel_max = 1.5
        self.ang_vel_min = -0.5
        self.ang_vel_max = 0.5
        self.ang_acc_max = 3.0
        self.ang_acc_min = -3.0
        self.lin_acc_max = 1
        self.lin_acc_min = -1
        self.N_hor = 20
        self.dt = 0.1
        
        # Weights 
        self.lin_vel_pen = 1.0
        self.lin_acc_pen = 10.0
        self.ang_vel_pen = 1.0
        self.ang_acc_pen = 5.0
        self.pos_dev = 1.0
        self.vel_dev = 10.0
        self.heading_dev = 1.0 
        self.termcost_pos = 200.0
        self.termcost_heading = 50.0 

        #Helpers 
        self.n_states = 3
        self.n_cmds = 2

        #Obstacles and boundaries
        self.obstacles = obstacles
        self.boundaries = boundaries
        self.r_safe = 0.5
        self.max_vert = 20 #max no of vertices
        self.w_obs = 1e6 #obstacle weigth


def generate_reftrajectory(p:Parameters,init_path):
    x_ref = init_path[:,0]
    y_ref = init_path[:,1] 
    dx = np.diff(x_ref,prepend=init_path[0,0])
    dy = np.diff(y_ref,prepend=init_path[0,1])
    theta_ref = np.unwrap(np.arctan2(dy,dx))
    return np.vstack((x_ref, y_ref, theta_ref)).T

def dyn_prop(x,u, p:Parameters):
    xp,yp,thetap = x[0],x[1],x[2]
    v,w = u[0],u[1]
    return ca.vertcat(xp + p.dt*v*ca.cos(thetap),
                      yp + p.dt*v*ca.sin(thetap),
                        thetap + p.dt*w)

def dyn_prop_np(x,u, p:Parameters):
    xp,yp,thetap = x
    v,w = u
    return np.array([xp + p.dt*v*np.cos(thetap),
                      yp + p.dt*v*np.sin(thetap),
                        thetap + p.dt*w])

def angle_wrapper(angle):
    return ca.atan2(ca.sin(angle), ca.cos(angle))


def open_solver(p : Parameters,build_dir="build_dir", name="nmpc_open"):
    no_x,no_u,N,dt = p.n_states, p.n_cmds, p.N_hor, p.dt
    obstacles_stat = p.obstacles
    boundary = p.boundaries
    w_obs = p.w_obs
    # Define optimization variables
    u = ca.SX.sym('u', no_u*N)  # 2 commands for each time step (v, omega)
    z = ca.SX.sym('x',no_x + no_u + no_x*(N+1) + 2*p.max_vert + 1) # vector with x0, u_prev for rate limits, ref state vector along tajectory, max number of vertices
    x0 = z[0:3] #initial state
    u_prev = z[3:5] #Previous command v & w
    ref = ca.reshape(z[5:5 + (N+1)*no_x],no_x,N+1) #ref traj with final pos and heading in 3D
    #Obstacle defintions
    max_vert = p.max_vert
    verts_flat_start = 5 + (N+1)*no_x
    verts_flat_end = verts_flat_start+2*max_vert
    verts_flat = z[verts_flat_start:verts_flat_end] #flat
    verts = ca.reshape(verts_flat,max_vert,2)
    r_safe = z[verts_flat_end]
    ob_terms = []

    #Initialize weights
    Q = ca.diag(ca.SX([p.pos_dev, p.pos_dev, p.heading_dev]))  # State deviation weights
    R = ca.diag(ca.SX([p.lin_vel_pen, p.ang_vel_pen]))  # Control effort weights
    Ra = ca.diag(ca.SX([p.lin_acc_pen, p.ang_acc_pen]))  # Acc change weights
    QN = ca.diag(ca.SX([p.termcost_pos, p.termcost_pos, p.termcost_heading]))  # Terminal state weights
    
    v_seq = u[0::2] #every second element starting from 0
    w_seq = u[1::2] #every second element starting from 1

    x = ca.SX(x0) #Current state
    J = 0 #Cost function

    # Accerelation terms for initial state
    dv0 = (v_seq[0] - u_prev[0]) / dt # Linear acc
    dw0 = (w_seq[0] - u_prev[1]) / dt # Angular acc
    J += ca.mtimes([ca.vertcat(dv0,dw0).T, Ra, ca.vertcat(dv0,dw0)])
    
    for i in range(N):
        #Obstacle constraint at each stage
        '''
        #pen constraint structure
        for j in range(max_vert):
            vx,vy = verts[j,0], verts[j,1] #center of circle
            in_circle = r_safe**2 - (x[0]-vx)**2 - (x[1]-vy)**2 #positive val if inside circle
            ob_terms.append(ca.fmax(0,in_circle)) #0 if cnts satisfied, else positive 
        
        '''
        
        for j in range(max_vert):
            vx,vy = verts[j,0], verts[j,1] #center of circle
            dx = x[0] - vx
            dy = x[1] - vy
            dist = r_safe**2 - (dx**2 + dy**2)
            ob_terms.append(dist) 
            #J += w_obs *ca.fmax(0,dist)
        
        #Stage cost
        xref = ref[:,i]
        err = ca.vertcat(x[0]-xref[0], x[1]-xref[1], angle_wrapper(x[2]-xref[2]))
        u_curr = ca.vertcat(v_seq[i], w_seq[i])
        J += ca.mtimes([err.T, Q, err]) + ca.mtimes([u_curr.T, R, u_curr])

        #Dynamics
        x = dyn_prop(x, u_curr, p)

        # Accel cost if not
        if i > 0:
            dv = (v_seq[i] - v_seq[i-1]) / dt
            dw = (w_seq[i] - w_seq[i-1]) / dt
            J += ca.mtimes([ca.vertcat(dv,dw).T, Ra, ca.vertcat(dv,dw)])
        
    ob_cntrs = ca.vertcat(*ob_terms) #make casadi vector if not doesnt work
    #J += 1e7 * ca.sumsqr(ob_cntrs)
    # Terminal cost
    err_N = ca.vertcat(x[0]-ref[0,N], x[1]-ref[1,N], angle_wrapper(x[2]-ref[2,N]))
    J += ca.mtimes([err_N.T, QN, err_N])

    # Input constraints
    umin = [p.vel_min, p.ang_vel_min]*N
    umax = [p.vel_max, p.ang_vel_max]*N
    vel_bounds = og.constraints.Rectangle(umin,umax)

    # Augmented Lagrangian for acc constaints
    acc = [dv0,dw0]
    for k in range(1,N):
        acc+= [(v_seq[k]-v_seq[k-1])/dt, (w_seq[k]-w_seq[k-1])/dt]
    acc = ca.vertcat(*acc)

    acc_min = [p.lin_acc_min, p.ang_acc_min]*N
    acc_max = [p.lin_acc_max, p.ang_acc_max]*N
    acc_bounds = og.constraints.Rectangle(acc_min, acc_max)

    ob_set = og.constraints.Rectangle([-1e10]*int(ob_cntrs.size1()),[0.0]*int(ob_cntrs.size1())) #create set for each vertex


    problem = og.builder.Problem(u,z,J) \
        .with_constraints(vel_bounds) \
        .with_aug_lagrangian_constraints(acc, acc_bounds) \
        .with_aug_lagrangian_constraints(ob_cntrs,ob_set) 

    build_cfg = og.config.BuildConfiguration() \
        .with_build_directory(build_dir) \
        .with_build_mode("release") \
        .with_tcp_interface_config()

    meta = og.config.OptimizerMeta().with_optimizer_name(name)

    solver_config = og.config.SolverConfiguration() \
        .with_tolerance(1e-6) \
        .with_max_duration_micros(500_000) \
        .with_initial_penalty(1e4) \
        .with_penalty_weight_update_factor(10.0)
    
        
    builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_cfg, solver_config) \
        .with_verbosity_level(1)
    
    builder.build()

    return build_dir,name

def start_manager(build_dir, name):
    mng = og.tcp.OptimizerTcpManager(f'{build_dir}/{name}')
    mng.start()
    return mng

def pack_params(x0, u_prev, xref,p: Parameters): #Xref is N+1,3
    stat_v = np.vstack([np.asarray(h, dtype=np.float64) for h in p.obstacles])
    svlen = min(len(stat_v), p.max_vert)
    padded = np.zeros((p.max_vert,2), dtype=np.float64) #initialize vert matrix
    if svlen > 0:
        padded[:svlen,:] = stat_v[:svlen,:] # populating matrix
    if svlen < p.max_vert:
        padded[svlen:,:] = 1e3 #fake distances for the rest of the empty entries
    
    verts_flat = padded.reshape(-1) #2*max_vert array
    
    return np.concatenate([
        np.asarray(x0, dtype=np.float64),
        np.asarray(u_prev, dtype=np.float64),
        np.asarray(xref, dtype=np.float64).reshape(-1),
        verts_flat,
        np.asarray([p.r_safe],dtype=np.float64)
    ])

def plot_trajectory(ref_trajectory):
    plt.figure(figsize=(10, 6))
    plt.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], 'r--', label='Reference Trajectory')
    #plt.plot(states[:, 0], states[:, 1], 'b-', label='Actual Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()


# ---------- Main Simulation Loop ----------
if __name__ == '__main__':
    config = 'test_config2'
    path, obstacles, boundary, padded_obstacles = path_planning.gen_path(config)
    p = Parameters(obstacles,boundary)
    build_dir, name = open_solver(p)
    mng = start_manager(build_dir, name)

    ref_trajectory = generate_reftrajectory(p,path)
    print(f'No of waypoints {ref_trajectory.shape}')
    x = ref_trajectory[0] # Initial state
    u_prev = np.array([0.0, 0.0]) # Initial previous command
    sim_time = len(ref_trajectory)*p.dt # Simulation time
    #x_end = ref_trajectory[-1]
    #end_thres = 0.25
    
    steps = int(sim_time / p.dt)
    states = np.zeros((steps, p.n_states))
    commands = np.zeros((steps, p.n_cmds))
    sim_traj = [x]
    crash_test = []

    for i in range(steps):
        #Segment based on current position
        end = min(i + p.N_hor, ref_trajectory.shape[0]-1)
        seg = ref_trajectory[i:end+1,:]
        if seg.shape[0] < p.N_hor + 1:
            # Pad with last state
            seg = np.vstack([seg,np.repeat(seg[-1][None,:],p.N_hor + 1 - seg.shape[0],axis=0)])
        static_vertices = np.vstack([np.asarray(hole,dtype=np.float64) for hole in obstacles])
        z = pack_params(x, u_prev, seg,p) # z for solver

        sol = mng.call(z)
        if not sol.is_ok():
            mng.kill() # stop rust
            raise RuntimeError(f"Solver failed {sol.get().message}")
        
        # decode the exact centers you enforced this step
        vstart = 5 + (p.N_hor+1)*p.n_states
        vend   = vstart + 2*p.max_vert
        verts_used = z[vstart:vend].reshape(p.max_vert, 2)
        r_enf = float(z[vend])          # 0.75 if you passed that

        # current executed node (what you plot)
        p_now = np.asarray(x[:2], float)

        # 1) distance to the ENFORCED centers
        mask = verts_used[:,0] < 1e8    # skip padded sentinels
        if mask.any():
            d_enf = np.linalg.norm(verts_used[mask] - p_now, axis=1)
            d_enf_min = float(d_enf.min())
        else:
            d_enf_min = np.inf

        # 2) distance to ALL plotted centers (all map vertices)
        all_verts = np.vstack([np.asarray(h, float) for h in obstacles])
        d_plot = np.linalg.norm(all_verts - p_now, axis=1)
        d_plot_min = float(d_plot.min())
        crash_test.append(d_enf_min)

        print(
            f"d_enforced_min={d_enf_min:.3f}, r_enforced={r_enf:.3f}, "
            f"d_plot_min={d_plot_min:.3f}, r_plot=0.500"
        )
        
        print(f'Step {i} Solver Success, cost is {sol.get().cost}, time spent is {sol.get().solve_time_ms} ms')
        u_opt = sol.get().solution #best control sequence
        vcurr,wcurr = float(u_opt[0]), float(u_opt[1]) #first command
        u_prev = np.array([vcurr,wcurr])

        #Apply first command
        x = dyn_prop_np(x, u_prev, p).flatten()
        states[i,:] = x
        commands[i,:] = u_prev
        sim_traj.append(x)

    mng.kill() # stop rust
    print("Done. Collected", len(sim_traj), "states.")


for element in crash_test:
    if element < 0.5:
        print(f'crashed at {element}')

# Plot simulated trajectory
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

#Debugiging COmmands
plt.figure()
plt.plot(commands[:,0], label='v')
plt.legend(); plt.grid(); plt.show()

plt.figure()
plt.plot(commands[:,1], label='omega')
plt.legend(); plt.grid(); plt.show()

print(f'Initial pos sim {sim_traj[0]} and initial pos ref {ref_trajectory[0]}')
