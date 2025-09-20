class Parameters:
    def __init__(self, obstacles, boundaries, dynobs):
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
        self.ang_vel_pen = 0.3
        self.ang_acc_pen = 1.0
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
        self.dynobs = dynobs
        self.r_safe = 0.5
        self.max_vert = 20 #max no of vertices
        self.w_obs = 1e6 #obstacle weigth
        self.vehicle_margin = 0.25
        self.n_dynobs = 1
