'''
Created on Jan 23, 2020

@author: Atrisha
'''

import constants
import math
import sqlite3
import ast 
import numpy as np
import itertools
from QuinticPolynomialsPlanner.quintic_polynomials_planner import *
from quartic_planner import *
import matplotlib.pyplot as plt
import sys
import utils
from planning_objects import VehicleState

    
    
class TrajectoryPlan:
    
    def generate_trajectory(self,veh_state):
        init_pos_x = float(veh_state.x)
        init_pos_y = float(veh_state.y)
        init_yaw_rads = float(veh_state.yaw)
        init_v = float(veh_state.speed)
        init_a = float(veh_state.long_acc)
        init_a_x = init_a * math.cos(init_yaw_rads)
        init_a_y = init_a * math.sin(init_yaw_rads)
        init_segment = veh_state.current_segment
        
        traj_list = []
        l1_action = self.l1_action
        l2_action = self.l2_action
        lane_boundary = None
        dt = constants.LP_FREQ
        #boundary_state_list += get_wait_states(veh_state, init_segment)
        #boundary_state_list += get_proceed_states(veh_state, init_segment)
        boundary_lane_spec = constants.LANE_BOUNDARY[init_segment+'|'+l1_action] if init_segment+'|'+l1_action in constants.LANE_BOUNDARY else None
        if boundary_lane_spec is not None:
            lane_boundary = get_lane_boundary(boundary_lane_spec)
        
        if self.task == 'LEFT_TURN' and veh_state.current_segment[0:2] != 'ln':
            l3_action_found = False
            boundary_state_list = []
            boundary_state_list = get_boundary_states(veh_state,l1_action,l2_action)
            if boundary_state_list is None or len(boundary_state_list) == 0:
                sys.exit('no boundary state list found for '+str(l1_action))
            N = len(boundary_state_list)
            for i,b_s in enumerate(boundary_state_list):
                #if math.hypot(init_pos_x-b_s[0], init_pos_y-b_s[1]) < constants.CAR_LENGTH/2:
                #    print('vehicle alreaskipped')
                #else:
                res = quintic_polynomials_planner(
                    init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a, b_s[0], b_s[1], b_s[2], b_s[3], b_s[4], b_s[5], abs(constants.MAX_ACC_JERK_AGGR), dt, 
                    lane_boundary)
                if res is not None:
                    l3_action_found = True
                    time, x, y, yaw, v, a, j, T, plan_type = res
                    clipped_res = utils.clip_trajectory_to_viewport(res)
                    #if len(traj_list) == constants.MAX_L3_ACTIONS:
                    #    break
                    if clipped_res is not None:
                        traj_list.append((clipped_res, plan_type))
                    #print(i,'/',N,T)
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
                    #print(i,'/',N,'no path',init_v,b_s[0],b_s[1],b_s[3])
                    continue
            if not l3_action_found:
                print('no path found left turn')
            else:
                l3_action_len = len(traj_list)
        elif self.task == 'STRAIGHT' or (self.task == 'LEFT_TURN' and veh_state.current_segment[0:2] == 'ln'):
            center_line = utils.get_centerline(veh_state.current_lane)
            if center_line is None:
                sys.exit('center line not found for '+str(veh_state.current_lane))
            accel_param = self.l2_action
            
            max_accel_long = constants.MAX_LONG_ACC_NORMAL+.5 if accel_param is 'NORMAL' else constants.MAX_LONG_ACC_AGGR-.5
            max_accel_lat = constants.MAX_LAT_ACC_NORMAL if accel_param is 'NORMAL' else constants.MAX_LAT_ACC_AGGR
            if self.lead_vehicle is None:
                acc_long_vals = np.random.normal(loc=max_accel_long, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_TRACK_SPEED)
                acc_lat_vals = np.random.normal(loc=max_accel_lat, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_TRACK_SPEED)
                acc_vals = list(zip(acc_long_vals,acc_lat_vals))
                for i,accel_val in enumerate(acc_vals):
                    #print('try',tries)
                    
                    res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, None, None, None, None, None, None, accel_val, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                    if res is not None:
                        time, x, y, yaw, v, a, j, T, plan_type = res
                        clipped_res = utils.clip_trajectory_to_viewport(res)
                        if clipped_res is not None:
                            traj_list.append((clipped_res, plan_type))
                        #print(T)
                        #plt.axis('equal')
                        #plt.plot(x,y,'-')
                        #plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                        #plt.show()
                        
                    else:
                        sys.exit('no path found track speed')
                        #print('no path found')
            else:
                lead_vehicle = self.lead_vehicle
                acc_long_vals = np.random.normal(loc=max_accel_long, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_FOLLOW_LEAD)
                acc_lat_vals = np.random.normal(loc=max_accel_lat, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_FOLLOW_LEAD)
                acc_vals = list(zip(acc_long_vals,acc_lat_vals))
                lvx, lvy, lvyaw, lvv, lvax, lvay = lead_vehicle.x, lead_vehicle.y, lead_vehicle.yaw, lead_vehicle.speed, lead_vehicle.tan_acc, lead_vehicle.long_acc
                for i,accel_val in enumerate(acc_vals):
                    #print('try',tries)
                    res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_val, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                    if res is not None:
                        time, x, y, yaw, v, a, j, T, plan_type = res
                        clipped_res = utils.clip_trajectory_to_viewport(res)
                        if clipped_res is not None:
                            traj_list.append((clipped_res, plan_type))
                        #print(T)
                        #plt.axis('equal')
                        #plt.plot(x,y,'-')
                        #plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                        #plt.show()
                        
                    else:
                        sys.exit('no path found follow lead')
                        #print('no path found')
        else:
            sys.exit('task '+self.task+' is not implemented')
            
        
        return np.asarray(traj_list)

    def __init__(self,l1_action,l2_action,task):
        self.l1_action = l1_action
        self.l2_action = l2_action
        self.task = task
    
    def set_l1_action(self,l1_action):
        self.l1_action = l1_action
        
    def set_l2_action(self,l2_action):
        self.l2_action = l2_action
        
    def set_task(self,task):
        self.task = task
        
    def set_lead_vehicle(self,lead_vehicle):
        self.lead_vehicle = lead_vehicle
    

def get_boundary_states(veh_state,l1_action,l2_action):
    if l1_action == 'wait':
        return get_wait_states(veh_state,l2_action)
    elif l1_action == 'proceed':
        return get_proceed_states(veh_state, l2_action)
    
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

def get_wait_states(veh_state,l2_action):
    ''' sample boundary points for WAIT maneuver '''
    init_segment = veh_state.current_segment
    boundary_state_list = []
    init_yaw_rads = veh_state.yaw
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
    
    if l2_action == 'AGGRESSIVE':
        max_dec = [max_abs_dec_aggr]*5
    elif l2_action == 'NORMAL': 
        max_dec = [max_abs_dec_normal]*5
    
    for s_c in stop_poss:
        for p,v,y in zip(s_c,goal_vels_zero,goal_yaws):
            for d in max_dec:
                boundary_state_list.append((p[0],p[1],y,v,0,d))
    
    return boundary_state_list

def get_continue_states(veh_state):
    ''' sample boundary points for CONTINUE maneuver '''
    init_segment = veh_state.current_segment
    boundary_state_list = []
    init_yaw_rads = veh_state.yaw
    curr_vel = veh_state.speed
    curr_acc = veh_state.long_acc
    proceed_pos_ends = list(get_exitline(constants.EXIT_SEGMENTS[init_segment]))
    #proceed_poss = utils.construct_state_grid(proceed_pos_ends[0], proceed_pos_ends[1], constants.N_PROCEED_POS_SAMPLES[init_segment],constants.LATERAL_TOLERANCE_EXITLINE)
    proceed_poss_x = np.random.normal((proceed_pos_ends[1][0]+proceed_pos_ends[0][0])/2,1,constants.N_PROCEED_POS_SAMPLES[init_segment])
    proceed_poss_y = np.random.normal((proceed_pos_ends[1][1]+proceed_pos_ends[0][1])/2,1,constants.N_PROCEED_POS_SAMPLES[init_segment])
    #plt.plot([veh_state[0][0]],[veh_state[0][1]],'go')
    #plt.plot([x[0] for x in proceed_pos_ends],[x[1] for x in proceed_pos_ends],'r-')
    #plt.plot(proceed_poss_x,proceed_poss_y,'bx')
    #plt.show()
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

    

def get_proceed_states(veh_state,l2_action):
    init_segment = veh_state.current_segment
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
    #max_dec_normal = np.arange(0.1,max_abs_dec_normal,(max_abs_dec_normal-0.1)/6)
    #max_dec_aggr = np.arange(max_abs_dec_normal,max_abs_dec_aggr,(max_abs_dec_aggr-max_abs_dec_normal)/6)
    if l2_action == 'AGGRESSIVE':
        max_abs_acc_aggr = abs(constants.MAX_LONG_ACC_AGGR)
        max_acc = [max_abs_acc_aggr]*5
    elif l2_action == 'NORMAL': 
        max_abs_acc_normal = abs(constants.MAX_LONG_ACC_NORMAL)
        max_acc = [max_abs_acc_normal]*5
    
    for s_c in proceed_poss:
        for p,v,a,y in zip(s_c,goal_vels_nonzero,goal_acc,goal_yaws):
            for d in max_acc:
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