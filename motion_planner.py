'''
Created on Jan 23, 2020

@author: Atrisha
'''

import constants
import math
import sqlite3
import ast 
from utils import distance_numpy
import numpy as np
import itertools

def get_stop_position():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = 'prep-turn_s' and REGION_PROPERTY='exit_boundary' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    stop_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        stop_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    return stop_coordinates

def sample_boundary_states(init_state):
    init_pos_x = init_state[0][0]
    init_pos_y = init_state[0][1]
    init_yaw_rads = init_state[1]
    
    boundary_state_list = []
    
    ''' sample boundary points for WAIT maneuver '''
    stop_pos = list(get_stop_position())
    dist_to_stop_pos = distance_numpy([stop_pos[0][0],stop_pos[0][1]], [stop_pos[1][0],stop_pos[1][1]], [init_pos_x,init_pos_y])
    dist_to_stop_pos = np.random.normal(dist_to_stop_pos,constants.STOP_LOC_STD_DEV,size=constants.N_STOP_POS_SAMPLES['prep-turn_s'])
    stop_poss = [(x*np.cos(init_yaw_rads),x*np.sin(init_yaw_rads)) for x in dist_to_stop_pos]
    goal_vels = np.random.normal(0,constants.STOP_VEL_STD_DEV,size=constants.N_STOP_VEL_SAMPLES['prep-turn_s'])
    max_dec_normal = np.arange(0.1,constants.MAX_DEC_NORMAL,5)
        
    
    
    
    print(stop_pos)
    
sample_boundary_states(((538842.19,4814000.61),1.658))