import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
from extremitypathfinder import PolygonEnvironment
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import yaml

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

    with open('params.yaml','r') as file:
        config_data = yaml.safe_load(file)

    boundary_coordinates = config_data[config]['boundary_coordinates']
    list_of_holes = config_data[config]['list_of_holes']
    
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

'''
path = gen_path()
#print("Path length:", length)
print("Path:", path)
plt.figure()
plt.plot(path[:, 0], path[:, 1], "r-")
plt.plot(start_coordinates[0], start_coordinates[1], "go")
plt.plot(goal_coordinates[0], goal_coordinates[1], "bx")
plt.plot(*zip(*boundary_coordinates, boundary_coordinates[0]), "k-")
for hole in list_of_holes:
    plt.plot(*zip(*hole, hole[0]), "k-")
plt.axis("equal")
plt.show()

path_interp = path_interpolate(path)
print(f"Interpolated path length: {len(path_interp)} points")
plt.figure()
plt.plot(path_interp[:, 0], path_interp[:, 1], "r-")
plt.show()
'''