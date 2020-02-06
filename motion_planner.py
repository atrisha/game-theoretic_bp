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
import sys


class VehicleState:
    def set_track_info(self,vehicle_track_info):
        self.track_info_set = True
        self.track_id = vehicle_track_info[0,]
        self.x = float(vehicle_track_info[1,])
        self.y = float(vehicle_track_info[2,])
        self.speed = utils.kph_to_mps(float(vehicle_track_info[3,]))
        self.tan_acc = float(vehicle_track_info[4,])
        self.long_acc = float(vehicle_track_info[5,])
        self.time = float(vehicle_track_info[6,])
        self.yaw = float(vehicle_track_info[7,])
        self.traffic_region = vehicle_track_info[8,]
        
    def __init__(self):
        self.track_info_set = True
    
    def set_current_segment(self,segment):
        self.current_segment = segment
    
    def set_current_lane(self,current_lane):
        self.current_lane = current_lane
    
    def set_segment_seq(self,segment_seq):
        self.segment_seq = segment_seq
        
    def set_current_time(self,time):
        if time is not None:
            self.current_time = float(time)
        else:
            self.current_time = None
        
    def set_gates(self,gates):
        self.gates = gates
        
    def set_traffic_light(self,signal):
        self.signal = signal
    
    def set_entry_exit_time(self,time_tuple):
        self.entry_exit_time = time_tuple
        
    def set_id(self,id):
        self.id = id
        
    def set_gate_crossing_times(self,times):
        self.gate_crossing_times = times
        
    def set_dist_to_segment_exit(self,dist):
        self.dist_to_segment_exit= dist
        
    def set_out_of_view(self,oov):
        self.out_of_view = oov
    
    
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
        
        if self.task == 'LEFT_TURN':
            l3_action_found = False
            boundary_state_list = []
            boundary_state_list = get_boundary_states(veh_state,l1_action,l2_action)
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
                    time, x, y, yaw, v, a, j, T = res
                    if len(traj_list) == constants.MAX_L3_ACTIONS:
                        break
                    traj_list.append([time, x, y, yaw, v, a, j])
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
        elif self.task == 'STRAIGHT':
            center_line = utils.get_centerline(veh_state.current_lane)
            if self.lead_vehicle is None:
                for tries in [1,2,3]:
                    #print('try',tries)
                    accel_param = self.l2_action
                    res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, None, None, None, None, None, None, accel_param, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                    if res is not None:
                        time, x, y, yaw, v, a, j, T = res
                        traj_list.append([time, x, y, yaw, v, a, j])
                        #print(T)
                        #plt.axis('equal')
                        #plt.plot(x,y,'-')
                        #plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                        #plt.show()
                        break
                    else:
                        sys.exit('no path found track speed')
                    print('no path found',tries)
            else:
                lead_vehicle = self.lead_vehicle
                lvx, lvy, lvyaw, lvv, lvax, lvay = lead_vehicle.x, lead_vehicle.y, lead_vehicle.yaw, lead_vehicle.speed, lead_vehicle.tan_acc, lead_vehicle.long_acc
                for tries in [1,2,3]:
                    #print('try',tries)
                    accel_param = self.l2_action
                    res = car_following_planner(init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a_x, init_a_y, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_param, abs(constants.MAX_ACC_JERK_AGGR), dt, None, center_line)
                    if res is not None:
                        time, x, y, yaw, v, a, j, T = res
                        traj_list.append([time, x, y, yaw, v, a, j])
                        #print(T)
                        #plt.axis('equal')
                        #plt.plot(x,y,'-')
                        #plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
                        #plt.show()
                        break
                    else:
                        sys.exit('no path found follow lead')
                    print('no path found',tries)
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
        return get_wait_states(veh_state)
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

def get_wait_states(veh_state):
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
    max_dec_normal = [max_abs_dec_normal]*5
    max_dec_aggr = [max_abs_dec_aggr]*5
    for s_c in stop_poss:
        for p,v,y in zip(s_c,goal_vels_zero,goal_yaws):
            for d in max_dec_normal:
                boundary_state_list.append((p[0],p[1],y,v,0,d))
            for d in max_dec_aggr:
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