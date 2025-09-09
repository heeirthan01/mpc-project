import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import opengen as og

class Parameters:
    def __init__(self, vel_min=-0.5, vel_max=1.5, ang_vel_min=-0.5, ang_vel_max=0.5, lin_acc_min=-0.1, lin_acc_max=0.1, ang_acc_min=-3.0, ang_acc_max=3.0):
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.ang_vel_min = ang_vel_min
        self.ang_vel_max = ang_vel_max
        self.ang_acc_max = 3.0
        self.ang_acc_min = -3.0
        self.lin_acc_max = 1
        self.lin_acc_min = -1
        self.N_hor = 70
        self.dt = 0.1
        
        # Weights 
        self.lin_vel_pen = 1.0
        self.lin_acc_pen = 10.0
        self.ang_vel_pen = 1.0
        self.ang_acc_pen = 5.0
        self.pos_dev = 10.0
        self.vel_dev = 10.0
        self.heading_dev = 1.0 
        self.termcost_pos = 50.0
        self.termcost_heading = 10.0 

        #Helpers 
        self.n_states = 3
        self.n_cmds = 2


def generate_reftrajectory(p:Parameters):
    # Generate a simple circular trajectory as a reference
    dt = p.dt # Time step for the reference trajectory
    T = 10  # Total time
    t = np.arange(0.0, T+dt, dt)
    #x_ref = 5 * np.cos(t) + 5 
    #y_ref = 5 * np.sin(t)
    x_ref = t  
    y_ref = 2*np.sin(0.5*t) 
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
    return ca.atan2(np.sin(angle), np.cos(angle))


def mpc_solver(p : Parameters):
    no_x,no_u,N,dt = p.n_states, p.n_cmds, p.N_hor, p.dt
    # Define optimization variables
    u = ca.MX.sym('u', no_u,N)  # 2 commands for each time step (v, omega)
    x = ca.MX.sym('x',no_x,N+1)
    x0 = ca.MX.sym('x0',no_x)  # Initial state
    ref = ca.MX.sym('ref',no_x,N+1) #ref traj with final pos and heading
    u0 = ca.MX.sym('u0',no_u)  #Previous command 
    
    #Initialize weights
    Q = ca.diag(ca.MX([p.pos_dev, p.pos_dev, p.heading_dev]))  # State deviation weights
    R = ca.diag(ca.MX([p.lin_vel_pen, p.ang_vel_pen]))  # Control effort weights
    Ra = ca.diag(ca.MX([p.lin_acc_pen, p.ang_acc_pen]))  # Acc change weights
    QN = ca.diag(ca.MX([p.termcost_pos, p.termcost_pos, p.termcost_heading]))  # Terminal state weights
    
    g = []  # Constraints vector
    cost = ca.MX(0)  # Objective function
    
    # Regular constraints
    for k in range(N):
        v= u[0,k]
        w = u[1,k]
        g+=[p.vel_min-v, v-p.vel_max,p.ang_vel_min-w,w-p.ang_vel_max] #vel and ang vel constraints

    #Rate Limiters
    for k in range(N):
        if k == 0:
            v_prev= u0[0]
            w_prev = u0[1]
        else:
            v_prev = u[0,k-1]
            w_prev = u[1,k-1]    
        dv = (u[0,k] - v_prev)/dt
        dw = (u[1,k] - w_prev)/dt
        g += [p.lin_acc_min - dv, dv - p.lin_acc_max, p.ang_acc_min -  dw, dw - p.ang_acc_max]
        cost += ca.dot(ca.vertcat(dv,dw), Ra @ ca.vertcat(dv,dw)) # Acceleration penalty 

    for k in range(N):
        xk,uk,xk_ref = x[:,k],u[:,k], ref[:,k]
        x_knext = x[:,k+1]
        e = xk - xk_ref
        cost += ca.dot(e, Q @ e) + ca.dot(uk, R @ uk)
        g.append(x_knext - dyn_prop(xk,uk,p))  # Dynamics constraint

    # Terminal cost
    eN = x[:,N] - ref[:,N]
    eN = ca.vertcat(eN[0], eN[1], angle_wrapper(eN[2]))  # Keep angle error within [-pi, pi]
    cost += ca.dot(eN, QN @ eN)
    g.append(x[:,0] - x0)  # Initial state constraint
    
    w = ca.vertcat(ca.vec(u), ca.vec(x))  # Decision variables vector
    g = ca.vertcat(*g)  # Constraints vector

    #Bounds for constraints
    n_eq = no_x*(N+1)
    n_ineq = 8*N #4 for vel, 4 for acc
    lbg_eq = ca.DM.zeros((n_eq))
    ubg_eq = ca.DM.zeros((n_eq))
    lbg_ineq = -ca.inf*ca.DM.ones(n_ineq)
    ubg_ineq = ca.DM.zeros((n_ineq))
    lbg = ca.vertcat(lbg_ineq, lbg_eq)
    ubg = ca.vertcat(ubg_ineq, ubg_eq)

    nlp = {'x': w, 'f': cost, 'g': g, 'p': ca.vertcat(x0, u0,ca.vec(ref))}
    opts = {'ipopt': {'print_level': 0, 'max_iter': 200, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6}}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def unpack_sol(w_opt):
        u_opt = w_opt[0:no_u*N].full().reshape((no_u, N))
        x_opt = w_opt[no_u*N:].full().reshape((no_x, N+1))
        return u_opt, x_opt
    bounds = dict(lbg=lbg, ubg=ubg,lbx=-ca.inf, ubx=ca.inf)
    return solver, unpack_sol,bounds

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
    solver, unpack_sol, bounds = mpc_solver(p)
    
    # Initial command guess
    u_prev = np.array([0.0, 0.0])
    
    # Reference trajectory
    ref_trajectory = generate_reftrajectory(p)
    # Initial state
    x = ref_trajectory[0].copy()
    # Simulation parameters
    sim_time = 10.0  # Total simulation time
    dt = p.dt
    steps = int(sim_time / dt)
    
    # Storage for states and commands
    states = np.zeros((steps, p.n_states))
    commands = np.zeros((steps, p.n_cmds))
    
    sim_traj = [x.copy()]
    
    for t in range(steps):
        
        # Select remaining reference trajectory 
        end = min(t + p.N_hor, ref_trajectory.shape[0]-1)
        segm = ref_trajectory[t:end+1,:]
        if segm.shape[0] < p.N_hor + 1:
            segm = np.vstack([segm,np.repeat(segm[-1][None,:], p.N_hor+1-segm.shape[0], axis=0)])
        xref = segm

        P = np.concatenate((x.ravel(), u_prev.ravel(), xref.ravel())) #Flatten vectors for solver
        w0 = np.zeros(p.n_cmds*p.N_hor + p.n_states*(p.N_hor+1))  # Initial guess sequence
        
        sol = solver(x0=w0, p=P, **bounds)
        print(f'Solver status at step {t}: {solver.stats()["return_status"]}')
        u_opt, x_opt = unpack_sol(sol['x'])
        u_applied = u_opt[:, 0]

        #Use first command
        x = np.array(dyn_prop_np(x, u_applied, p)).flatten()
        u_prev = u_applied
        commands[t, :] = u_applied
        sim_traj.append(x.copy())

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