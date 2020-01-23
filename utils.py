'''
Created on Jan 16, 2020

@author: Atrisha
'''
import sqlite3
import math
import numpy as np
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    A,B,P = np.asarray(A),np.asarray(B),np.asarray(P)
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

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
                        
                        


