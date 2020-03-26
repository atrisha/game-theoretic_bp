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
from Planners.quintic_polynomials_planner import *
from Planners.cubic_spline_planner import *
from quartic_planner import *
import matplotlib.pyplot as plt
import sys
import utils
from planning_objects import VehicleState
from scipy.interpolate import CubicSpline
from scipy.stats import halfnorm
from pip._vendor.distlib.util import proceed
from collections import OrderedDict

    
    
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
        '''this tuple is to flag whether trajectories were found or not (in_view_trajectories,clipped_trajectories) '''
        traj_generated = (False,False)
        traj_list = []
        l1_action = self.l1_action
        l2_action = self.l2_action
        veh_state.set_current_l2_action(l2_action)
        veh_state.set_current_l1_action(l1_action)
        lane_boundary = None
        dt = constants.LP_FREQ
        #boundary_state_list += get_wait_states(veh_state, init_segment)
        #boundary_state_list += get_proceed_states(veh_state, init_segment)
        boundary_lane_spec = constants.LANE_BOUNDARY[init_segment+'|'+l1_action] if init_segment+'|'+l1_action in constants.LANE_BOUNDARY else None
        if boundary_lane_spec is not None:
            lane_boundary = get_lane_boundary(boundary_lane_spec)
        planner_type = 'CP'
        if self.l1_action == 'proceed-turn' or self.l1_action == 'wait-for-oncoming':
            boundary_state_list = []
            if self.l1_action == 'wait-for-oncoming':
                res = construct_stationary_trajectory(veh_state)
                traj_list.append((res, 'ST'))
            if self.l1_action == 'proceed-turn' and planner_type == 'CP':
                boundary_states = get_proceed_states_cp(veh_state, l2_action)
                traj_list = cubic_spline_planner(veh_state, boundary_states)
            elif self.l1_action == 'wait-for-oncoming' and planner_type == 'CP':
                boundary_states = get_wait_states_cp(veh_state, l2_action)
                traj_list = traj_list + cubic_spline_planner(veh_state, boundary_states)
            else:
                l3_action_found = False
                boundary_state_list = get_boundary_states(veh_state,l1_action,l2_action)
                if boundary_state_list is None or len(boundary_state_list) == 0:
                    sys.exit('no boundary state list found for '+str(l1_action))
                N = len(boundary_state_list)
                for i,b_s in enumerate(boundary_state_list):
                    if math.hypot(init_pos_x-b_s[0], init_pos_y-b_s[1]) < constants.CAR_LENGTH/2:
                        ''' add waiting at current location to the trajectory '''
                        f = 1
                    else:
                        res = qp_planner(
                            init_pos_x, init_pos_y, init_yaw_rads, init_v, init_a, b_s[0], b_s[1], b_s[2], b_s[3], b_s[4], b_s[5], abs(constants.MAX_ACC_JERK_AGGR), dt, 
                            lane_boundary)
                        if res is not None:
                            l3_action_found = True
                            traj_generated[1] = True
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
        elif self.l1_action == 'track_speed' or self.l1_action == 'follow_lead':
            center_line = utils.get_centerline(veh_state.current_segment)
            if center_line is None:
                sys.exit('center line not found for '+str(veh_state.current_segment))
            accel_param = self.l2_action
            if constants.SEGMENT_MAP[veh_state.current_segment] == 'left-turn-lane':
                target_vel = constants.LEFT_TURN_VEL_START_POS
            else:
                target_vel = constants.TARGET_VEL
            
            max_accel_long = constants.MAX_LONG_ACC_NORMAL if accel_param is 'NORMAL' else constants.MAX_LONG_ACC_AGGR
            max_accel_lat = constants.MAX_LAT_ACC_NORMAL if accel_param is 'NORMAL' else constants.MAX_LAT_ACC_AGGR
            target_vel = target_vel + constants.LEFT_TURN_VEL_START_POS_AGGR_ADDITIVE if self.l2_action == 'AGGRESSIVE' else target_vel
            #if self.lead_vehicle is None:
            acc_long_vals = np.random.normal(loc=max_accel_long, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_TRACK_SPEED)
            acc_lat_vals = np.random.normal(loc=max_accel_lat, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_TRACK_SPEED)
            if accel_param == 'NORMAL':
                if self.lead_vehicle is None:
                    if veh_state.speed <= constants.TARGET_VEL:
                        target_vel_vals = np.random.uniform(low=constants.TARGET_VEL,high=constants.TARGET_VEL+(10/3.6), size=constants.MAX_SAMPLES_TRACK_SPEED)
                    else:
                        target_vel_vals = np.random.uniform(low=veh_state.speed,high=veh_state.speed+2, size=constants.MAX_SAMPLES_TRACK_SPEED)
                else:
                    target_vel_vals = np.random.uniform(low=min(self.lead_vehicle.speed,veh_state.speed),high=max(self.lead_vehicle.speed,veh_state.speed), size=constants.MAX_SAMPLES_TRACK_SPEED)
            else:
                if self.lead_vehicle is None:
                    if veh_state.speed <= constants.TARGET_VEL:
                        target_vel_vals = halfnorm.rvs(loc=constants.TARGET_VEL, scale=constants.TARGET_VEL_SD, size=constants.MAX_SAMPLES_TRACK_SPEED)
                    else:
                        target_vel_vals = halfnorm.rvs(loc=veh_state.speed, scale=veh_state.speed+2, size=constants.MAX_SAMPLES_TRACK_SPEED)
                else:
                    target_vel_vals = halfnorm.rvs(loc=self.lead_vehicle.speed, scale=self.lead_vehicle.speed+2, size=constants.MAX_SAMPLES_TRACK_SPEED)
            target_vel_vals = [x if x<=constants.TRACK_SPEED_VEL_LIMIT else veh_state.speed + (constants.TRACK_SPEED_VEL_LIMIT-veh_state.speed)*np.random.random_sample() for x in target_vel_vals]
            state_vals = list(zip(acc_long_vals,acc_lat_vals,target_vel_vals)) + [(0,0,veh_state.speed)]
            for i,state_val in enumerate(state_vals):
                res = track_speed_planner_cp(veh_state, center_line, state_val)
                '''
                if planner_type != 'CP':
                    if res is not None:
                        time, x, y, yaw, v, a, j, T, plan_type = res
                        traj_generated[1] = True
                        clipped_res = utils.clip_trajectory_to_viewport(res)
                        if clipped_res is not None:
                            traj_list.append((clipped_res, plan_type))
                            
                    else:
                        sys.exit('no path found track speed')
                        #print('no path found')
                else:
                    if res is not None:
                        traj_list = traj_list + res
                '''
                if res is not None:
                        traj_list = traj_list + res
            '''
            for t in traj_list:
                sl = min(len(t[0][0]),len(t[0][4]))
                plt.plot(t[0][0][:sl],t[0][4][:sl])
            plt.show()
            '''
            '''
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
                        traj_generated = (traj_generated[0],True)
                        if constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
                            clipped_res = utils.clip_trajectory_to_viewport(res)
                            if clipped_res is not None:
                                traj_list.append((clipped_res, plan_type))
                        else:
                            traj_list.append((res,plan_type))
                        
                    else:
                        sys.exit('no path found follow lead')
                        #print('no path found')
            '''
        elif self.l1_action == 'decelerate-to-stop' or self.l1_action == 'wait_for_lead_to_cross':
            max_decel_long = constants.MAX_LONG_DEC_NORMAL if self.l2_action == 'NORMAL' else constants.MAX_LONG_DEC_AGGR
            max_decel_lat = constants.MAX_LAT_DEC_NORMAL if self.l2_action is 'NORMAL' else constants.MAX_LAT_DEC_AGGR
            center_line = utils.get_centerline(veh_state.current_lane)
            if self.l1_action != 'wait_for_lead_to_cross' and center_line is None:
                sys.exit('center line not found for '+str(veh_state.current_lane))
            if center_line is not None:
                proceed_line = center_line
            else:
                length = constants.CAR_LENGTH*2
                proceed_line = [(veh_state.x,veh_state.y),(veh_state.x+length*np.cos(veh_state.yaw),veh_state.y+length*np.sin(veh_state.yaw))]
            dec_long_vals = np.random.normal(loc=max_decel_long, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_DEC_TO_STOP)
            dec_lat_vals = np.random.normal(loc=max_decel_lat, scale=constants.PROCEED_ACC_SD, size=constants.MAX_SAMPLES_DEC_TO_STOP)
            stop_line = utils.get_exit_boundary(veh_state.current_segment)
            central_stop_point = (stop_line[0][0]+(stop_line[0][1] - stop_line[0][0])/2, stop_line[1][0] + (stop_line[1][1] - stop_line[1][0])/2)
            vect_to_vehicle = [veh_state.x - central_stop_point[0], veh_state.y - central_stop_point[1]]
            norm_vect_to_vehicle = math.hypot(vect_to_vehicle[0], vect_to_vehicle[1])
            unit_v = [x/norm_vect_to_vehicle for x in vect_to_vehicle]
            stop_poss = []
            stop_point_constructed = False
            if self.lead_vehicle is not None:
                vect_from_lead_to_sv = [veh_state.x-self.lead_vehicle.x, veh_state.y-self.lead_vehicle.y]
                if math.hypot(vect_from_lead_to_sv[0], vect_from_lead_to_sv[1]) - norm_vect_to_vehicle <= constants.CAR_LENGTH:
                    ''' construct stop point based on the lead vehicle position '''
                    central_stop_point = (self.lead_vehicle.x, self.lead_vehicle.y)
                    for d in constants.LATERAL_TOLERANCE_DISTANCE_GAPS:
                        stop_poss.append((central_stop_point[0] + d*unit_v[0], central_stop_point[1]+d*unit_v[1]))
                    stop_point_constructed = True
            if not stop_point_constructed:
                for d in constants.LATERAL_TOLERANCE_STOPLINE:
                    stop_poss.append((central_stop_point[0] + d*unit_v[0], central_stop_point[1]+d*unit_v[1]))
            target_vel_vals = [0]*len(dec_long_vals)
            state_vals = list(zip(dec_long_vals,dec_lat_vals,stop_poss))
            for i,state_val in enumerate(state_vals):
                dist_to_stop = math.hypot(veh_state.x-state_val[2][0], veh_state.y-state_val[2][1])
                res = decelerate_to_stop_planner_cp(veh_state, proceed_line, dist_to_stop)
                if res is not None:
                    traj_list.append((res, 'CP'))
                    
            
            if self.l1_action == 'wait_for_lead_to_cross':
                res = construct_stationary_trajectory(veh_state)
                traj_list.append((res, 'ST'))
                
                
        elif self.l1_action == 'follow_lead_into_intersection':
            traj_list = generate_follow_lead_in_intersection_trajectory(veh_state)
            
        elif self.l1_action == 'wait-on-red':
            res = construct_stationary_trajectory(veh_state)
            traj_list.append((res, 'ST'))
        else:
            sys.exit('task '+self.task+' is not implemented')
            
        if len(traj_list) > 0:
            traj_generated = (True, traj_generated[1])
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

def construct_stationary_trajectory(veh_state):
    time, x, y, yaw, v, a, j, T = [], [], [], [], [], [], [], 0 
    sc_trajectory = utils.get_track(veh_state, veh_state.current_time, True)
    selected_indices = np.arange(0,len(sc_trajectory),constants.DATASET_FPS*constants.LP_FREQ,dtype=int)
    sc_trajectory = sc_trajectory[selected_indices,:]
    rx, ry = [float(x) for x in sc_trajectory[:,1]],[float(y) for y in sc_trajectory[:,2]]
    hpx = [rx[0],rx[len(rx)//3],rx[2*len(rx)//3],rx[-1]]
    hpy = [ry[0],ry[len(ry)//3],ry[2*len(ry)//3],ry[-1]]
    
    try:
        _d = OrderedDict(sorted(list(zip(hpx,hpy)),key=lambda tup: tup[0]))
        hpx,hpy = list(_d.keys()),list(_d.values())
        cs = CubicSpline(hpx, hpy)
    except (ValueError,IndexError,TypeError):
        print(hpx)
        print(hpy)
        raise
    ref_path = [(float(x),cs(float(x))) for x in rx]
    time = [float(t) for t in sc_trajectory[:int(constants.PLAN_FREQ/constants.LP_FREQ),6]]
    if veh_state.l2_action == 'AGGRESSIVE':
        max_dec_lims,max_dec_jerk_lims = constants.MAX_LONG_DEC_AGGR, constants.MAX_DEC_JERK_AGGR
    else:
        max_dec_lims,max_dec_jerk_lims = constants.MAX_LONG_DEC_NORMAL, constants.MAX_DEC_JERK_NORMAL
    
    vel_profile = construct_stationary_velocity_profile(time, veh_state.speed, veh_state.long_acc, max_dec_lims, max_dec_jerk_lims)
    v = vel_profile
    traj = utils.generate_trajectory_from_vel_profile(time, ref_path, vel_profile)
    x = [e[0] for e in traj]
    y = [e[1] for e in traj]
    yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
    a = [veh_state.long_acc] + [e[1]-e[0] for e in zip(vel_profile[:-1],vel_profile[1:])]
    j = [e[1]-e[0] for e in zip(a[:-1],a[1:])]
    j = j + [j[-1]]
    return np.asarray(time), np.asarray(x), np.asarray(y), np.asarray(yaw), np.asarray(v), np.asarray(a), np.asarray(j), len(time)

def construct_stationary_velocity_profile(time,vs,a_s,max_dec_lims,max_dec_jerk_lims):
    vel_profile = [vs]
    a = a_s
    for i in np.arange(1,len(time)):
        delta_t_secs = time[i] - time[i-1]
        if a > max_dec_lims:
            a = max(a - (max_dec_jerk_lims*constants.LP_FREQ), max_dec_lims*constants.LP_FREQ)   
        vel_t = max(0, vel_profile[-1] + a*delta_t_secs) 
        vel_profile.append(vel_t)
    return vel_profile




def get_boundary_states(veh_state,l1_action,l2_action):
    if l1_action == 'wait-for-oncoming':
        return get_wait_states(veh_state,l2_action)
    elif l1_action == 'proceed-turn':
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

def get_exitline_from_direction(direction,veh_state):
    if constants.SEGMENT_MAP[veh_state.current_segment] == 'left-turn-lane':
        exit_segment = 'prep-turn_'+direction[2].lower()
    else:
        exit_segment = 'exec-turn_'+direction[2].lower()
    return get_exitline(exit_segment)

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
                                           constants.LATERAL_TOLERANCE_EXITLINE,'line')
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

def get_wait_states_cp(veh_state,l2_action):
    init_segment = veh_state.current_segment
    boundary_state_list = []
    ''' sample boundary points for PROCEED maneuver '''
    #next_segment = veh_state.segment_seq[veh_state.segment_seq.index(veh_state.current_segment)+1]
    knot_ends = []
    knot_ends.append((get_exitline(veh_state.current_segment),veh_state.current_segment))
    ''' to fix the yaw, we need to add another knot end just after the last'''
    curr_last_knot = knot_ends[-1][0]
    last_knotline_normal = math.atan2(curr_last_knot[1][1]-curr_last_knot[0][1], curr_last_knot[1][0]-curr_last_knot[0][0]) + (math.pi/2)
    last_knot_line = [(x[0]+(1*np.cos(last_knotline_normal)),x[1]+(1*np.sin(last_knotline_normal))) for x in curr_last_knot]
    knot_ends.append((last_knot_line,knot_ends[-1][1]))
    path_knot_grids, vel_knot_grids = [], []
    
    for i,_e in enumerate(knot_ends):
        k_e,seg_name = _e[0], _e[1]
        proceed_poss = utils.construct_state_grid(k_e[0], k_e[1], constants.N_PROCEED_POS_SAMPLES[init_segment],\
                                              constants.LATERAL_TOLERANCE_EXITLINE,'line')
        velocity_samples = np.full(shape=(proceed_poss.shape[0],proceed_poss.shape[1]), fill_value=0.0)
        path_knot_grids.append(proceed_poss)
        vel_knot_grids.append(velocity_samples)
    return (path_knot_grids,vel_knot_grids)


def get_proceed_states_cp(veh_state,l2_action):
    init_segment = veh_state.current_segment
    boundary_state_list = []
    ''' sample boundary points for PROCEED maneuver '''
    next_segment = veh_state.segment_seq[veh_state.segment_seq.index(veh_state.current_segment)+1]
    knot_ends = []
    knot_ends.append((get_exitline(next_segment),next_segment))
    if veh_state.segment_seq.index(next_segment) < len(veh_state.segment_seq)-1:
        next_to_next_segment = next_segment = veh_state.segment_seq[veh_state.segment_seq.index(next_segment)+1]
        knot_ends.append((get_exitline(next_to_next_segment),next_to_next_segment))
    ''' to fix the yaw, we need to add another knot end just after the last'''
    curr_last_knot = knot_ends[-1][0]
    last_knotline_normal = math.atan2(curr_last_knot[1][1]-curr_last_knot[0][1], curr_last_knot[1][0]-curr_last_knot[0][0]) + (math.pi/2)
    last_knot_line = [(x[0]+(constants.CAR_LENGTH*np.cos(last_knotline_normal)),x[1]+(constants.CAR_LENGTH*np.sin(last_knotline_normal))) for x in curr_last_knot]
    knot_ends.append((last_knot_line,knot_ends[-1][1]))
    path_knot_grids, vel_knot_grids = [], []
    
    for i,_e in enumerate(knot_ends):
        k_e,seg_name = _e[0], _e[1]
        proceed_poss = utils.construct_state_grid(k_e[0], k_e[1], constants.N_PROCEED_POS_SAMPLES[init_segment],\
                                              constants.LATERAL_TOLERANCE_EXITLINE,'line')
        if constants.SEGMENT_MAP[seg_name] == 'exec-left-turn':
            vel_mean = constants.PROCEED_VEL_MEAN_EXEC_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXEC_TURN + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
            vel_sd =  constants.PROCEED_VEL_SD_EXEC_TURN
            vel_diff = constants.PROCEED_VEL_MEAN_EXEC_TURN - constants.PROCEED_VEL_MEAN_PREP_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXEC_TURN - constants.PROCEED_VEL_MEAN_PREP_TURN 
        elif constants.SEGMENT_MAP[next_segment] == 'exit-lane':
            vel_mean = constants.PROCEED_VEL_MEAN_EXIT if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXIT + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
            vel_sd =  constants.PROCEED_VEL_SD_EXIT
            vel_diff = constants.PROCEED_VEL_MEAN_EXIT-constants.PROCEED_VEL_MEAN_EXEC_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXIT-constants.PROCEED_VEL_MEAN_EXEC_TURN 
        else:
            ''' segment is prepare turn segment'''
            vel_mean = constants.PROCEED_VEL_MEAN_PREP_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_PREP_TURN + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
            vel_sd =  constants.PROCEED_VEL_SD_PREP_TURN
            vel_diff = vel_mean
        '''
        if i==0:
            velocity_samples = np.random.normal(loc=vel_mean, scale=vel_sd, size=(proceed_poss.shape[0],proceed_poss.shape[1]))
        else:
            velocity_samples = vel_knot_grids[-1] + halfnorm.rvs(loc=vel_diff, scale=vel_sd, size=(proceed_poss.shape[0],proceed_poss.shape[1]))
        '''
        path_knot_grids.append(proceed_poss)
        #vel_knot_grids.append(velocity_samples)
    
    vel_knot_grids = []
    for i,k in enumerate(path_knot_grids):
        if i>0:
            curr_vels = vel_knot_grids[i-1]
            dist_array = np.hypot(path_knot_grids[i][:,:,0]-path_knot_grids[i-1][:,:,0],path_knot_grids[i][:,:,1]-path_knot_grids[i-1][:,:,1])
        else:
            curr_vels = np.full(shape=(path_knot_grids[i].shape[0],path_knot_grids[i].shape[1]),fill_value = veh_state.speed)
            dist_array = np.hypot(path_knot_grids[i][:,:,0]-veh_state.x,path_knot_grids[i][:,:,1]-veh_state.y)
        if veh_state.l2_action == 'NORMAL':
            acc_samples = np.random.uniform(low=0.5,high=1.5,size=(path_knot_grids[i].shape[0],path_knot_grids[i].shape[1]))
        else:
            acc_samples = halfnorm.rvs(loc=1, scale=.5, size=(path_knot_grids[i].shape[0],path_knot_grids[i].shape[1]))
        final_vels = np.sqrt(np.square(curr_vels) + 2 * np.multiply(acc_samples,dist_array))
        vel_knot_grids.append(final_vels)
    
    return (path_knot_grids,vel_knot_grids)
    


def get_proceed_states(veh_state,l2_action):
    init_segment = veh_state.current_segment
    boundary_state_list = []
    ''' sample boundary points for PROCEED maneuver '''
    next_segment = veh_state.segment_seq[veh_state.segment_seq.index(veh_state.current_segment)+1]
    proceed_pos_ends = get_exitline_from_direction(veh_state.direction,veh_state)
    proceed_poss = utils.construct_state_grid(proceed_pos_ends[0], proceed_pos_ends[1], constants.N_PROCEED_POS_SAMPLES[init_segment],\
                                              constants.LATERAL_TOLERANCE_EXITLINE,'line')
    if next_segment[0:4] == 'exec':
        vel_mean = constants.PROCEED_VEL_MEAN_EXEC_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXEC_TURN + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
        vel_sd =  constants.PROCEED_VEL_SD_EXEC_TURN
    elif constants.SEGMENT_MAP[next_segment] == 'exit-lane':
        ''' next segment is ln_*_* (end of segment)'''
        vel_mean = constants.PROCEED_VEL_MEAN_EXIT if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_EXIT + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
        vel_sd =  constants.PROCEED_VEL_SD_EXIT
    else:
        ''' next segment is prepare turn segment'''
        vel_mean = constants.PROCEED_VEL_MEAN_PREP_TURN if l2_action == 'NORMAL' else constants.PROCEED_VEL_MEAN_PREP_TURN + constants.PROCEED_VEL_AGGRESSIVE_ADDITIVE
        vel_sd =  constants.PROCEED_VEL_SD_PREP_TURN
    
    goal_vels_nonzero = np.random.normal(vel_mean,vel_sd,size=constants.N_PROCEED_VEL_SAMPLES[init_segment])
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