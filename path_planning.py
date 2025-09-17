import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
from extremitypathfinder import PolygonEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pyclipper
import yaml
import math
from parameters import Parameters


SCALE = 1000

def to_clipper(polygon):
    return [(int(round(x*SCALE)),int(round(y*SCALE))) for x,y in polygon] #float to int

def from_clipper(polygon):
    return [(x/SCALE,y/SCALE) for x,y in polygon] #int to float


def signed_area(polygon):
    # Returns True if polygon vertices are clockwise
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += (x1 * y2) - (x2 * y1)
    return area*0.5 

def make_ccw(polygon):
    return polygon if signed_area(polygon) > 0 else polygon[::-1]

def make_cw(polygon):
    return polygon if signed_area(polygon) < 0 else polygon[::-1]


def path_offset(path_int,vehicle_width): #"Path" is the list of obstacles
    clipoff = pyclipper.PyclipperOffset()
    clipoff.AddPath(path_int,pyclipper.JT_MITER,pyclipper.ET_CLOSEDPOLYGON)
    delta_int = int(round(vehicle_width*SCALE))
    out = clipoff.Execute(delta_int)
    if not out:
        raise RuntimeError('Offset Did Not Succeed')
    return out[0]


def inflate_obstacle(obstacle,vehicle_width=0.5):
    og_path = to_clipper(obstacle)
    #Need negative scale for CW
    ext_path = path_offset(og_path, vehicle_width)
    return from_clipper(ext_path)

def inflate_obstacles(list_of_holes, vehicle_width=0.5):
    inflated_obstacles = []
    for obs in list_of_holes:
        obs = make_cw(obs)
        inflated_obstacle = inflate_obstacle(obs, vehicle_width)
        inflated_obstacle = make_cw(inflated_obstacle)
        inflated_obstacles.append(inflated_obstacle)
    return inflated_obstacles

def shrink_boundary(boundary, vehicle_width):
    boundary = make_ccw(boundary)
    ogpath = to_clipper(boundary)
    shrunkpath = path_offset(ogpath,-vehicle_width)
    return make_ccw(from_clipper(shrunkpath))

def rotate_object(origin,point,seg_heading):
    '''
    Helper for generate dyn obs
    Inputs
    origin: first point in segment p1, center of local frame
    point: point of interest p3 along line we want to convert
    seg_heading: angle of segmetn wrt global frame
    '''
    cx,cy = origin 
    px,py = point 
    #Rotate into local frame
    qx = math.cos(seg_heading) * (px-cx) - math.sin(seg_heading)*(py-cy) + cx
    qy = math.sin(seg_heading) * (px-cx) - math.cos(seg_heading)*(py-cy) + cy

    return np.array([qx,qy])

def rotate_and_add(p1,p3,seg_heading,addition):

    rot = rotate_object(p1,p3,seg_heading) #rotate and translate into local frame
    rot[1] += addition #offset perpendicularly to segment (squiglywiggly)
    rot = rotate_object(np.array([0,0]),rot,-seg_heading) #rotate back into global frame

    return rot + p1 #translation back into 0,0 origin

def gen_dynamic_obstacle(p1,p2,freq,time,amp=1.5):
    p1 = np.array(p1)
    p2 = np.array(p2)
    seg_heading = np.arctan2(p1[1],p2[0]) #segment heading
    time = np.array(time) 
    t = abs(np.sin(freq*time)) #time "trajectory"
    if type(t) == np.ndarray: #check if numpy array
        t = np.expand_dims(t,1) #ensure (n,1) instead of (n,)
    p3 = t*p1 + (1-t)*p2
    add = amp*np.cos(10*freq*time) #the addition vertical sqwig

    return rotate_and_add(p1,p3,seg_heading,add) #return time specific point in global frame
    

def get_dynobs_paths(dynobs_total,t,p:Parameters):
    dt = p.dt
    horizon = p.N_hor
    time = np.linspace(t,t+horizon*dt,horizon) #time array according to prediction horizon
    dynobs_paths = []
    for obs in dynobs_total:
        p1,p2,freq,x_rad,y_rad, seg_heading = obs #TODO: Engrave obstacle info into .yaml file
        x_rad = x_rad + p.r_safe/2 + p.vehicle_margin #Expand object based on vehicle width and margin
        y_rad = y_rad + p.r_safe/2 + p.vehicle_margin
        obap = [(*gen_dynamic_obstacle(p1,p2,freq,t),x_rad,y_rad,seg_heading) for t in time] #the 5 time based parameters mentioned in paper
        dynobs_paths.append(obap)
    
    return dynobs_paths

def get_dynobs_path_at_t(dynobs_total,t,p:Parameters):
    dynobs_paths = []
    for obs in dynobs_total:
        p1,p2,freq,x_rad,y_rad, seg_heading = obs #TODO: Engrave obstacle info into .yaml file
        x_rad = x_rad + p.r_safe/2 + p.vehicle_margin #Expand object based on vehicle width and margin
        y_rad = y_rad + p.r_safe/2 + p.vehicle_margin
        x,y = gen_dynamic_obstacle(p1,p2,freq,t)
        dynobs_paths.append((x,y,x_rad,y_rad,seg_heading))
    return dynobs_paths

def path_interpolate(path,ds = 0.1):
    '''
    path = 2d array of shape (N,2)
    ds = desired distance between points
    returns (P,2) array of interpolated path w endpoints
    '''
    P = np.asarray(path,dtype=float)
    #remove duplicate consecutive pts
    mask = np.ones(len(P),dtype=bool)
    mask[1:] = np.any(np.diff(P,axis=0) != 0.0,axis=1) # Booleans for whether points are different or not
    P = P[mask] #Selected points where mask is True
    
    if len(P) < 2:
        return P
    
    seg = np.diff(P,axis=0) # vector difference between pts
    seg_len = np.linalg.norm(seg,axis=1) #length of those differences
    cum = np.r_[0.0,np.cumsum(seg_len)] #cumulative arc length
    total = cum[-1]
    if total == 0.0:
        return P[:1]
    
    #arclength samples
    s_samples = np.arange(0.0,total,ds)

    x = np.interp(s_samples,cum,P[:,0]) #find point at those sampled lengths
    y = np.interp(s_samples,cum,P[:,1])

    return np.column_stack([x,y])

def gen_path(config):
    #Main part
    environment = PolygonEnvironment()

    with open('obsbounds.yaml','r') as file:
        config_data = yaml.safe_load(file)

    boundary_coordinates = config_data[config]['boundary_coordinates']
    list_of_holes = config_data[config]['list_of_holes']
    #dynobs = config_data[config]['dynobs']
    
    obstacles_processed = inflate_obstacles(list_of_holes, 0.5)
    boundary_processed = shrink_boundary(boundary_coordinates, vehicle_width=0.5)

    environment.store(boundary_processed, obstacles_processed, validate=True)
    environment.prepare()

    start_coordinates = (1.0, 25.0)
    goal_coordinates = (49.0, 30.0)
    path, length = environment.find_shortest_path(start_coordinates, goal_coordinates)
    path = np.array(path, dtype=np.float32)
    path = path_interpolate(path)
    padded_vertices = obstacles_processed
    return path , list_of_holes, boundary_coordinates, padded_vertices

def generate_reftrajectory(p:Parameters,init_path):
    x_ref = init_path[:,0]
    y_ref = init_path[:,1] 
    dx = np.diff(x_ref,prepend=init_path[0,0])
    dy = np.diff(y_ref,prepend=init_path[0,1])
    theta_ref = np.unwrap(np.arctan2(dy,dx))
    return np.vstack((x_ref, y_ref, theta_ref)).T

'''
# Working test animation of obstacles
config = 'test_config2'
path, obstacles, boundary, padded_obstacles = gen_path(config)
testp = Parameters(obstacles,boundary)
dynobslist = [([1.0, 4.0], [2.0, 7.0], 0.1, 0.2, 0.5, 0.1)]
t = 500
obstacles = get_dynobs_paths(dynobslist,t,testp)
reps = 50
for i in range(reps):
    animate_dymobs(obstacles,boundary,testp)
'''
