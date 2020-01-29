'''
Created on Jan 16, 2020

@author: Atrisha
'''
import sqlite3
import math
import numpy as np
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
import matplotlib.pyplot as plt
import ast

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    A,B,P = np.asarray(A),np.asarray(B),np.asarray(P)
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    angle_with_A = arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A)))
    angle_with_B = arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2
    if angle_with_A > pi / 2:
        return norm(P - A) * math.sin(pi - angle_with_A)
    if angle_with_B > pi / 2:
        return norm(P - B) * math.sin(pi - angle_with_B)
    #return norm(cross(A-B, A-P))/norm(B-A),
    return cross(A-B, A-P)/norm(B-A)

def get_centerline(lane_segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+lane_segment+"' and REGION_PROPERTY='center_line' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    center_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        center_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    return center_coordinates


def dist_along_yaw(pt1,pt2,yaw,pos):
    ''' r*cos(yaw-slope + 90) = d '''
    d = abs(distance_numpy([pt1[0],pt1[1]], [pt2[0],pt2[1]], [pos[0],pos[1]]))
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    r = d / (math.cos(yaw - slope + (.5*math.pi)))
    return abs(r)

def kph_to_mps(kph):
    return kph/3.6

def fresnet_to_map(o_x,o_y,X,Y,centerline_angle):
    theta = abs(centerline_angle)
    M_X,M_Y = [],[]
    for x,y in zip(X,Y):
        m_x = o_x + x*math.cos(theta)
        point_angle_with_fresnet = math.atan2(y, x)
        _a = math.hypot(x, y)
        m_y = o_y - (math.hypot(x, y) * math.sin(theta-point_angle_with_fresnet))
        M_X.append(m_x)
        M_Y.append(m_y)
    return M_X,M_Y
        
        

def split_in_n(pt1,pt2,N):
    step = (pt2[0] - pt1[0])/N
    x_coords = [pt1[0] + (step*i) for i in np.arange(N)]
    x_coords = x_coords + [pt2[0]]
    step = (pt2[1] - pt1[1])/N
    y_coords = [pt1[1] + (step*i) for i in np.arange(N)]
    y_coords = y_coords + [pt2[1]]
    return list(zip(x_coords,y_coords))

def construct_state_grid(pt1,pt2,N,tol):
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    slope_comp = (math.pi/2) - slope
    central_coords = split_in_n(pt1,pt2,N)
    grid = [central_coords]
    for r in tol:
        grid.append([(x[0]-(r*math.cos(slope_comp)),x[1]+(r*math.sin(slope_comp))) for x in central_coords])
    return grid
'''
plt.plot([0,20],[0,30],'ro')
split_pts = split_in_n((0,0), (20,30), 10)
plt.plot([x[0] for x in split_pts],[x[1] for x in split_pts],'bx')
plt.show()
'''

def dist(x,y):
    x = (float(x[0]),float(x[1]))
    y = (float(y[0]),float(y[1]))
    return math.hypot(x[0]-y[0], x[1]-y[1])



def get_all_vehicles_on_lane(time,list_of_lanes):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND TRAFFIC_REGIONS LIKE '%l_n_s%';"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles

def get_n_s_vehicles_on_intersection(time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND TRAFFIC_REGIONS LIKE '%l_n_s%';"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles

def get_n_e_vehicles_before_intersection(time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND TRAFFIC_REGIONS LIKE '%l_n_1%';"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles

def get_n_e_vehicles_on_intersection(time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND TRAFFIC_REGIONS LIKE '%l_n_e%';"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles

def get_closest_east_vehicles_before_intersection(time,loc):
    vehicles = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_e_1%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_e_2%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_e_3%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    conn.close()
    return vehicles


def get_closest_west_vehicles_before_intersection(time,loc):
    vehicles = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_w_1%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_w_2%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    
    q_string = "SELECT DISTINCT TRACK_ID,X,Y FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND  TRAFFIC_REGIONS LIKE '%ln_w_3%'"
    c.execute(q_string)
    res = c.fetchall()
    all_vehicles = []
    for row in res:
        all_vehicles.append(row)
    dist_list = [dist(loc,d) for d in [(x[1],x[2]) for x in all_vehicles]]
    closest_index = dist_list.index(min(dist_list))
    vehicles.append(all_vehicles[closest_index][0])
    conn.close()
    return vehicles

    
def get_n_s_vehicles_before_intersection(time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND (TRAFFIC_REGIONS LIKE '%ln_n_2%' OR TRAFFIC_REGIONS LIKE '%ln_n_3%')"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles

def oncoming_vehicles_on_intersection_cond():
    return entry_exit_gate_cond(60, 18)

def entry_exit_gate_cond(entry_gate,exit_gate):
    return  "WHERE ((ENTRY_GATE = "+str(entry_gate)+" AND EXIT_GATE = "+str(exit_gate)+" )" + \
                        " OR (EXIT_GATE = "+str(exit_gate)+" AND ENTRY_GATE IS NULL))"
                        
