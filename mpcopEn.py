import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import opengen as og
import path_planning


class Parameters:
    def __init__(self):
        self.vel_min = -0.5
        self.vel_max = 1.5
        self.ang_vel_min = -0.5
        self.ang_vel_max = 0.5
        self.ang_acc_max = 3.0
        self.ang_acc_min = -3.0
        self.lin_acc_max = 1
        self.lin_acc_min = -1
        self.N_hor = 70
        self.dt = 0.1
        
        # Weights 
        self.lin_vel_pen = 1.0/10
        self.lin_acc_pen = 10.0
        self.ang_vel_pen = 1.0
        self.ang_acc_pen = 5.0
        self.pos_dev = 1.0
        self.vel_dev = 10.0
        self.heading_dev = 1.0 
        self.termcost_pos = 10.0
        self.termcost_heading = 10.0 

        #Helpers 
        self.n_states = 3
        self.n_cmds = 2


def generate_reftrajectory(p:Parameters,init_path):
    # Generate a simple circular trajectory as a reference
    dt = p.dt # Time step for the reference trajectory
    T = 10  # Total time
    t = np.arange(0.0, T+dt, dt)
    x_ref = init_path[:,0]
    y_ref = init_path[:,1] 
    theta_ref = np.arctan2(np.gradient(y_ref), np.gradient(x_ref))
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


def open_solver(p : Parameters, build_dir="build_dir", name="nmpc_open"):
    no_x,no_u,N,dt = p.n_states, p.n_cmds, p.N_hor, p.dt
    # Define optimization variables
    u = ca.SX.sym('u', no_u*N)  # 2 commands for each time step (v, omega)
    z = ca.SX.sym('x',no_x + no_u + no_x*(N+1)) # vector with x0, u_prev for rate limits and ref state vector along tajectory
    x0 = z[0:3] #initial state
    u_prev = z[3:5] #Previous command v & w
    ref = ca.reshape(z[5:],no_x,N+1) #ref traj with final pos and heading in 3D
     
    
    #Initialize weights
    Q = ca.diag(ca.SX([p.pos_dev, p.pos_dev, p.heading_dev]))  # State deviation weights
    R = ca.diag(ca.SX([p.lin_vel_pen, p.ang_vel_pen]))  # Control effort weights
    Ra = ca.diag(ca.SX([p.lin_acc_pen, p.ang_acc_pen]))  # Acc change weights
    QN = ca.diag(ca.SX([p.termcost_pos, p.termcost_pos, p.termcost_heading]))  # Terminal state weights
    
    v_seq = u[0::2] #every second element starting from 0
    w_seq = u[1::2] #every second element starting from 1

    x = ca.SX(x0) #Current state
    J = 0 #Cost function

    # Accerelation (rate) terms for initial state
    dv0 = (v_seq[0] - u_prev[0]) / dt # Linear acc
    dw0 = (w_seq[0] - u_prev[1]) / dt # Angular acc
    J += ca.mtimes([ca.vertcat(dv0,dw0).T, Ra, ca.vertcat(dv0,dw0)])
    
    for i in range(N):
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

    problem = og.builder.Problem(u,z,J) \
        .with_constraints(vel_bounds) \
        .with_aug_lagrangian_constraints(acc, acc_bounds)

    build_cfg = og.config.BuildConfiguration() \
        .with_build_directory(build_dir) \
        .with_build_mode("release") \
        .with_tcp_interface_config()

    meta = og.config.OptimizerMeta().with_optimizer_name(name)

    solver_config = og.config.SolverConfiguration() \
        .with_tolerance(1e-4) \
        .with_max_duration_micros(300000) \
        
    builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_cfg, solver_config) \
        .with_verbosity_level(1)
    
    builder.build()

    return build_dir,name

def start_manager(build_dir, name):
    mng = og.tcp.OptimizerTcpManager(f'{build_dir}/{name}')
    mng.start()
    return mng

def pack_params(x0, u_prev, xref): #Xref is N+1,3
    return np.concatenate([
        np.asarray(x0, dtype=np.float64),
        np.asarray(u_prev, dtype=np.float64),
        np.asarray(xref, dtype=np.float64).reshape(-1)
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
    p = Parameters()
    path = path_planning.gen_path()
    build_dir, name = open_solver(p)
    mng = start_manager(build_dir, name)

    ref_trajectory = generate_reftrajectory(p,path)
    print(f'No of waypoints {ref_trajectory.shape}')
    x = ref_trajectory[0] # Initial state
    u_prev = np.array([0.0, 0.0]) # Initial previous command
    sim_time = len(ref_trajectory)*p.dt # Total simulation time
    
    steps = int(sim_time / p.dt)
    states = np.zeros((steps, p.n_states))
    commands = np.zeros((steps, p.n_cmds))
    sim_traj = [x]

    for i in range(steps):
        #Segment based on current position
        end = min(i + p.N_hor, ref_trajectory.shape[0]-1)
        seg = ref_trajectory[i:end+1,:]
        if seg.shape[0] < p.N_hor + 1:
            # Pad with last state
            seg = np.vstack([seg,np.repeat(seg[-1][None,:],p.N_hor + 1 - seg.shape[0],axis=0)])
        z = pack_params(x, u_prev, seg) # z for solver

        sol = mng.call(z)
        if not sol.is_ok():
            mng.kill() # stop rust
            raise RuntimeError(f"Solver failed {sol.get().message}")
        
        print(f'Step {i} Solver Success, cost is {sol.get().cost}')
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


# Plot simulated trajectory
plt.figure()
sim_traj = np.array(sim_traj)
plt.plot(sim_traj[:,0], sim_traj[:,1], label='Simulated Trajectory')
plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', label='Reference Trajectory')
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
