'''
Created on Jan 23, 2020

@author: Atrisha
'''

import constants
import math
import sqlite3
import ast 
import utils
import numpy as np
import itertools
from QuinticPolynomialsPlanner.quintic_polynomials_planner import *
from quartic_planner import *
import matplotlib.pyplot as plt
import motion_planner


class VehicleState:
    
    def __init__(self,vehicle_track_info):
        self.track_id = vehicle_track_info[0,]
        self.x = vehicle_track_info[1,]
        self.y = vehicle_track_info[2,]
        self.speed = vehicle_track_info[3,]
        self.tan_acc = vehicle_track_info[4,]
        self.long_acc = vehicle_track_info[5,]
        self.time = vehicle_track_info[6,]
        self.yaw = vehicle_track_info[7,]
        self.traffic_region = vehicle_track_info[8,]
        
    def set_current_segment(self,segment):
        self.current_segment = segment
    
    def set_path(self,self,path):
        self.path = path
    
class TrajectoryPlan:
    
    def __init__(self,action_l1):
        self.action_l1 = action_l1
        
    def generate_trajectory(self):
        t = 5
    
    
def get_stopline(init_segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+init_segment+"' and REGION_PROPERTY='exit_boundary' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    stop_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        stop_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    return stop_coordinates

def get_exitline(exit_segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+exit_segment+"' and REGION_PROPERTY='exit_boundary' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    stop_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        stop_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    return stop_coordinates

def get_wait_states(init_state,init_segment):
    ''' sample boundary points for WAIT maneuver '''
    boundary_state_list = []
    init_yaw_rads = init_state[1]
    stop_pos_ends = list(get_stopline(init_segment))
    stop_poss = utils.construct_state_grid(stop_pos_ends[0], stop_pos_ends[1], constants.N_STOP_POS_SAMPLES[init_segment],\
                                           constants.LATERAL_TOLERANCE_STOPLINE)
    #goal_vels_nonzero = np.random.normal(constants.STOP_VEL_TOLERANCE,constants.STOP_VEL_TOLERANCE,size=constants.N_STOP_VEL_SAMPLES[init_segment])
    #goal_acc = np.random.normal(0,constants.STOP_DEC_STD_DEV,size=constants.N_STOP_VEL_SAMPLES[init_segment])
    
    goal_vels_zero = [0]*constants.N_STOP_VEL_SAMPLES[init_segment]
    goal_yaws = np.random.normal(init_yaw_rads,constants.STOP_YAW_TOLERANCE,size = constants.N_STOP_VEL_SAMPLES[init_segment])
    
    max_abs_dec_normal = abs(constants.MAX_LONG_DEC_NORMAL)
    max_abs_dec_aggr = abs(constants.MAX_LONG_DEC_AGGR)
    #max_dec_normal = np.arange(0.1,max_abs_dec_normal,(max_abs_dec_normal-0.1)/6)
    #max_dec_aggr = np.arange(max_abs_dec_normal,max_abs_dec_aggr,(max_abs_dec_aggr-max_abs_dec_normal)/6)
    max_dec_normal = [max_abs_dec_normal]*5
    max_dec_aggr = [max_abs_dec_aggr]*5
    for s_c in stop_poss:
        for p,v,y in zip(s_c,goal_vels_zero,goal_yaws):
            for d in max_dec_normal:
                boundary_state_list.append((p[0],p[1],y,v,0,d))
            for d in max_dec_aggr:
                boundary_state_list.append((p[0],p[1],y,v,0,d))
    
    return boundary_state_list

def get_continue_states(init_state,init_segment):
    ''' sample boundary points for CONTINUE maneuver '''
    boundary_state_list = []
    init_yaw_rads = init_state[1]
    curr_vel = init_state[2]
    curr_acc = init_state[3]
    proceed_pos_ends = list(get_exitline(constants.EXIT_SEGMENTS[init_segment]))
    #proceed_poss = utils.construct_state_grid(proceed_pos_ends[0], proceed_pos_ends[1], constants.N_PROCEED_POS_SAMPLES[init_segment],constants.LATERAL_TOLERANCE_EXITLINE)
    proceed_poss_x = np.random.normal((proceed_pos_ends[1][0]+proceed_pos_ends[0][0])/2,1,constants.N_PROCEED_POS_SAMPLES[init_segment])
    proceed_poss_y = np.random.normal((proceed_pos_ends[1][1]+proceed_pos_ends[0][1])/2,1,constants.N_PROCEED_POS_SAMPLES[init_segment])
    plt.plot([init_state[0][0]],[init_state[0][1]],'go')
    plt.plot([x[0] for x in proceed_pos_ends],[x[1] for x in proceed_pos_ends],'r-')
    plt.plot(proceed_poss_x,proceed_poss_y,'bx')
    plt.show()
    proceed_poss = zip(proceed_poss_x,proceed_poss_y)
    goal_vels_nonzero = np.random.normal(curr_vel,constants.MAINTAIN_VEL_SD,size=constants.N_PROCEED_VEL_SAMPLES[init_segment])
    #goal_acc = np.random.normal(curr_acc,constants.PROCEED_ACC_SD,size=constants.N_PROCEED_VEL_SAMPLES[init_segment])
    goal_acc = [curr_acc]*constants.N_PROCEED_VEL_SAMPLES[init_segment]
    ''' TODO : find the yaw of the lane boundary at exit point '''
    exit_yaw = init_yaw_rads   
    '''
    plt.plot([proceed_pos_ends[0][0]],[proceed_pos_ends[0][1]],'rx')
    plt.plot([proceed_pos_ends[1][0]],[proceed_pos_ends[1][1]],'bx')
    plot_arrow(proceed_pos_ends[0][0]+1, proceed_pos_ends[0][1]-1,exit_yaw)
    plt.show()
    '''
    #goal_yaws = np.random.normal(exit_yaw,constants.PROCEED_YAW_TOLERANCE,size = constants.N_PROCEED_VEL_SAMPLES[init_segment])
    goal_yaws = [exit_yaw]*constants.N_PROCEED_VEL_SAMPLES[init_segment]
    max_abs_acc_normal = abs(constants.MAX_ACC_NORMAL)
    max_abs_acc_aggr = abs(constants.MAX_ACC_AGGR)
    #max_dec_normal = np.arange(0.1,max_abs_dec_normal,(max_abs_dec_normal-0.1)/6)
    #max_dec_aggr = np.arange(max_abs_dec_normal,max_abs_dec_aggr,(max_abs_dec_aggr-max_abs_dec_normal)/6)
    max_acc_normal = [max_abs_acc_normal]*5
    max_acc_aggr = [max_abs_acc_aggr]*5
    for p in proceed_poss:
        for v,a,y in zip(goal_vels_nonzero,goal_acc,goal_yaws):
            for d in max_acc_normal:
                boundary_state_list.append((p[0],p[1],y,v,a,d))
            for d in max_acc_aggr:
                boundary_state_list.append((p[0],p[1],y,v,a,d))
    return boundary_state_list

    

def get_proceed_states(init_state,init_segment):
    boundary_state_list = []
    ''' sample boundary points for PROCEED maneuver '''
    proceed_pos_ends = list(get_exitline(constants.EXIT_SEGMENTS[init_segment]))
    proceed_poss = utils.construct_state_grid(proceed_pos_ends[0], proceed_pos_ends[1], constants.N_PROCEED_POS_SAMPLES[init_segment],\
                                              constants.LATERAL_TOLERANCE_EXITLINE)
    goal_vels_nonzero = np.random.normal(constants.PROCEED_VEL_MEAN,constants.PROCEED_VEL_SD,size=constants.N_PROCEED_VEL_SAMPLES[init_segment])
    goal_acc = np.random.normal(constants.PROCEED_ACC_MEAN,constants.PROCEED_ACC_SD,size=constants.N_PROCEED_VEL_SAMPLES[init_segment])
    
    _tan_v = (proceed_pos_ends[0][1]-proceed_pos_ends[1][1])/(proceed_pos_ends[0][0]-proceed_pos_ends[1][0])
    proceed_line_angle = math.atan(_tan_v)
    exit_yaw = math.pi/2 + proceed_line_angle if proceed_line_angle > 0 else proceed_line_angle - (math.pi/2) - .15   
    '''
    plt.plot([proceed_pos_ends[0][0]],[proceed_pos_ends[0][1]],'rx')
    plt.plot([proceed_pos_ends[1][0]],[proceed_pos_ends[1][1]],'bx')
    plot_arrow(proceed_pos_ends[0][0]+1, proceed_pos_ends[0][1]-1,exit_yaw)
    plt.show()
    '''
    #goal_yaws = np.random.normal(exit_yaw,constants.PROCEED_YAW_TOLERANCE,size = constants.N_PROCEED_VEL_SAMPLES[init_segment])
    goal_yaws = [exit_yaw]*constants.N_PROCEED_VEL_SAMPLES[init_segment]
    max_abs_acc_normal = abs(constants.MAX_LONG_ACC_NORMAL)
    max_abs_acc_aggr = abs(constants.MAX_LONG_ACC_AGGR)
    #max_dec_normal = np.arange(0.1,max_abs_dec_normal,(max_abs_dec_normal-0.1)/6)
    #max_dec_aggr = np.arange(max_abs_dec_normal,max_abs_dec_aggr,(max_abs_dec_aggr-max_abs_dec_normal)/6)
    max_acc_normal = [max_abs_acc_normal]*5
    max_acc_aggr = [max_abs_acc_aggr]*5
    for s_c in proceed_poss:
        for p,v,a,y in zip(s_c,goal_vels_nonzero,goal_acc,goal_yaws):
            for d in max_acc_normal:
                boundary_state_list.append((p[0],p[1],y,v,a,d))
            for d in max_acc_aggr:
                boundary_state_list.append((p[0],p[1],y,v,a,d))
    return boundary_state_list
    
    
    
def get_lane_boundary(boundary_lane_spec):
    lane_name,lane_property = boundary_lane_spec.split('|')
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+lane_name+"' and REGION_PROPERTY='"+lane_property+"' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    lb_xs,lb_ys = [],[]
    for row in res:
        lb_xs = ast.literal_eval(row[0])
        lb_ys = ast.literal_eval(row[1])
    lane_boundary = [lb_xs,lb_ys]
    return lane_boundary
    
    
def show_trajectories(trajectories,lane_boundaries):    
    plt.axis('equal')
    
    for lb in lane_boundaries:
        if lb is not None:
            plt.plot(lb[0],lb[1],'r-')
    for traj_list in trajectories:
        for t in traj_list:
            plt.plot(t[1], t[2],'-')
    plt.show()
        


def generate_trajectory(init_state,l1_actions):
    init_pos_x = init_state.x
    init_pos_y = init_state.y
    init_yaw_rads = init_state.yaw
    init_v = init_state.speed
    init_a = init_state[3].long_acc
    init_a_x = init_a * math.cos(init_yaw_rads)
    init_a_y = init_a * math.sin(init_yaw_rads)
    init_segment = init_state.current_segment
    for l1_action in l1_actions:
        traj_plan = TrajectoryPlan(l1_action)
    boundary_state_list = []
    traj_list = []
    lane_boundary = None
    dt = constants.LP_FREQ
    if init_segment == 'prep-turn_s':
        boundary_state_list += get_wait_states(init_state, init_segment)
        boundary_state_list += get_proceed_states(init_state, init_segment)
        maneuver = 'proceed'
        boundary_lane_spec = constants.LANE_BOUNDARY[init_segment+'|'+maneuver]
        lane_boundary = get_lane_boundary(boundary_lane_spec)
        N = len(boundary_state_list)
        for i,b_s in enumerate(boundary_state_list):
            if math.hypot(init_pos_x-b_s[0], init_pos_y-b_s[1]) < constants.CAR_LENGTH/2:
                print('skipped')
            else:
                res = quintic_polynomials_planner(
                    init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a, b_s[0], b_s[1], b_s[2], b_s[3], b_s[4], b_s[5], abs(constants.MAX_ACC_JERK_AGGR), dt, 
                    lane_boundary)
                if res is not None:
                    time, x, y, yaw, v, a, j, T = res
                    traj_list.append([time, x, y, yaw, v, a, j])
                    print(i,'/',N,T)
                    '''
                    goal_sign = np.sign(utils.distance_numpy([lb_xs[0],lb_ys[0]], [lb_xs[1],lb_ys[1]], [b_s[0], b_s[1]]))
                    dist_to_lane_b = np.sign([utils.distance_numpy([lb_xs[0],lb_ys[0]], [lb_xs[1],lb_ys[1]], [t[0],t[1]]) for t in list(zip(x,y))])
                    within_lane = np.all(dist_to_lane_b == goal_sign)
                    print(within_lane)
                    plt.axis('equal')
                    plt.plot([proceed_pos_ends[0][0]],[proceed_pos_ends[0][1]],'rx')
                    plt.plot([proceed_pos_ends[1][0]],[proceed_pos_ends[1][1]],'bx')
                    plt.plot(lb_xs,lb_ys,'r-')
                    plt.plot(x,y,'-')
                    plt.show()
                    '''
                else:
                    print(i,'/',N,'no path',init_v,b_s[0],b_s[1],b_s[3])
                    continue
    else:
        center_line = utils.get_centerline(constants.EXIT_SEGMENTS[init_segment])
        has_leading_vehicle = True
        if not has_leading_vehicle:
            for tries in [1,2,3]:
                print('try',tries)
                if tries==3:
                    accel_param = 'AGGR'
                else:
                    accel_param = 'NORMAL'
                res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, None, None, None, None, None, None, accel_param, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                if res is not None:
                    time, x, y, yaw, v, a, j, T = res
                    traj_list.append([time, x, y, yaw, v, a, j])
                    print(T)
                    plt.axis('equal')
                    plt.plot(x,y,'-')
                    plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                    plt.show()
                    break
            print('no path found')
        else:
            lvx, lvy, lvyaw, lvv, lvax, lvay = 538839.93,4813997.24,5.3094,utils.kph_to_mps(53.1595),-0.1741,0.0026
            for tries in [1,2,3]:
                print('try',tries)
                if tries==3:
                    accel_param = 'AGGR'
                else:
                    accel_param = 'NORMAL'
                res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_param, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                if res is not None:
                    time, x, y, yaw, v, a, j, T = res
                    traj_list.append([time, x, y, yaw, v, a, j])
                    print(T)
                    plt.axis('equal')
                    plt.plot(x,y,'-')
                    plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                    plt.show()
                    break
            print('no path found')
        
    
    return traj_list,lane_boundary
                
'''            
trajectories,lane_boundaries = [],[]

traj_list,lane_boundary = generate_trajectory(((538842.19,4814000.61),1.658,0.127,0),'prep-turn_s')
trajectories.append(traj_list)
lane_boundaries.append(lane_boundary)

traj_list,lane_boundary = generate_trajectory(((538830.52,4814012.16),5.3031,17.19,0),'int-entry_n')
#trajectories.append(traj_list)
lane_boundaries.append(lane_boundary)
show_trajectories(trajectories, lane_boundaries)
'''