'''
Created on Jan 23, 2020

@author: Atrisha
'''

import constants
import math
import sqlite3
import ast 
import os
import numpy as np
import itertools
from motion_planners.quintic_polynomials_planner import *
from motion_planners.cubic_spline_planner import *
from motion_planners.quartic_planner import *
import matplotlib.pyplot as plt
import sys
import all_utils
from all_utils import utils
from motion_planners.planning_objects import VehicleState
from scipy.interpolate import CubicSpline, griddata, interp1d
from scipy.stats import halfnorm
from pip._vendor.distlib.util import proceed
from collections import OrderedDict
from constants import L2_ACTION_CODES
from visualizer import visualizer
from collections import OrderedDict
from all_utils.thread_utils import CustomMPS
from scipy.stats import truncnorm

log = constants.common_logger


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class PathInfo:
    
    def __init__(self,traj_plan):
        self.segments = OrderedDict()
        self.veh_state = traj_plan.veh_state
        self.all_paths = []
        self.traj_plan = traj_plan
    
    def append_segment(self,segment_name,centerline):
        self.segments[segment_name] = dict()
        self.segments[segment_name]['centerline'] = centerline
        
    def add_waypoints_for_segment(self,segment_name,waypts):
        generate_boundary = self.traj_plan.generate_boundary
        if segment_name in self.segments:
            self.segments[segment_name]['waypoints'] = waypts
            lat_tol_check_key = constants.SEGMENT_MAP[segment_name]
            if lat_tol_check_key == 'exit-lane' and self.veh_state.task == 'STRAIGHT':
                lat_tol_check_key = 'DEFAULT'
            if lat_tol_check_key in constants.LATERAL_LATTICE_TOLERANCE:
                lat_tol = constants.LATERAL_LATTICE_TOLERANCE[lat_tol_check_key]
            else:
                lat_tol = constants.LATERAL_LATTICE_TOLERANCE['DEFAULT']
            self.segments[segment_name]['lateral_waypoints'] = [waypts]
            if self.traj_plan.trajectory_type == 'GAUSSIAN':
                self.segments[segment_name]['gaussian_waypoints'] = []
                X = get_truncated_normal(mean=0, sd=lat_tol/2, low=0, upp=lat_tol)
                for d in X.rvs(4):
                    pl1,pl2 = utils.add_parallel(waypts, d)
                    self.segments[segment_name]['gaussian_waypoints'].append(pl1)
                    self.segments[segment_name]['gaussian_waypoints'].append(pl2)
            
            if generate_boundary:
                self.segments[segment_name]['boundary_waypoints'] = []
                pl1,pl2 = utils.add_parallel(waypts, lat_tol)
                self.segments[segment_name]['boundary_waypoints'].append(pl1)
                self.segments[segment_name]['boundary_waypoints'].append(pl2)
           
    
class TrajectoryPlan:
    
    def insert_baseline_trajectory(self,table_name):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
        c = conn.cursor()
        q_string = "select MAX("+table_name+".TRAJECTORY_ID) FROM "+table_name
        c.execute(q_string)
        res = c.fetchone()
        max_traj_id = int(res[0]) if res[0] is not None else 0
        agent_id = self.veh_id
        time_ts = self.veh_state.current_time
        relev_agent = self.relev_veh_id
        l1_action = self.l1_action
        l2_action = self.l2_action
        i_string_data = (int(constants.CURRENT_FILE_ID),agent_id,relev_agent,l1_action,l2_action,time_ts,1)
        #print('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?,?)',i_string_data)
        c.execute("SELECT * FROM GENERATED_TRAJECTORY_INFO WHERE AGENT_ID="+str(i_string_data[1])+" AND RELEV_AGENT_ID="+str(i_string_data[2])+" AND L1_ACTION='"+str(i_string_data[3])+"' AND \
                        L2_ACTION='"+str(i_string_data[4])+"' AND TIME="+str(i_string_data[5]))
        res = c.fetchone()
        if res is not None and len(res) > 0:
            traj_info_id = res[1]
            log.info('deleted '+table_name+' with info id'+str(traj_info_id))
            c.execute('DELETE FROM '+table_name+' WHERE TRAJECTORY_INFO_ID='+str(traj_info_id))
            conn.commit()
        else:
            c.execute('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?)',i_string_data)
            conn.commit()
            traj_info_id = int(c.lastrowid)
            
        
        traj_id = max_traj_id+1
        ins_list = []
        if table_name == 'GENERATED_BOUNDARY_TRAJECTORY':
            traj_category = 'boundary'
        elif table_name == 'GENERATED_GAUSSIAN_TRAJECTORY':
            traj_category = 'gaussian'
        else:
            traj_category = 'baseline'
        for traj_dets in self.trajectories[traj_category]:
            slice_len = min([len(x) for x in traj_dets[0:7]])
            tx,rx,ry,ryaw,rv,ra,rj = traj_dets[0][:slice_len],traj_dets[1][:slice_len],traj_dets[2][:slice_len],traj_dets[3][:slice_len],traj_dets[4][:slice_len],traj_dets[5][:slice_len],traj_dets[6][:slice_len]
            ins_list.extend(list(zip([traj_id]*slice_len,[traj_info_id]*slice_len,[round(x,5) for x in tx],[round(x,5) for x in rx],[round(x,5) for x in ry],[round(x,5) for x in ryaw],[round(x,5) for x in rv],[round(x,5) for x in ra],[round(x,5) for x in rj])))
            traj_id += 1
        i_string = 'INSERT INTO '+table_name+' VALUES (?,?,?,?,?,?,?,?,?)'
        #n=1000
        #chunks = [ins_list[i:i+n] for i in range(0, len(ins_list), n)]
        #for chunk in chunks:
        c.executemany(i_string,ins_list)
        conn.commit()
        conn.close()
        log.info('inserted '+table_name+' with info id'+str(traj_info_id)+ ' and time_ts '+str(i_string_data[5]))
    

    
    def set_generate_boundary(self,flag):
        self.generate_boundary = flag
    
    ''' returns False if the path collides with a vehicle in the current scene'''
    def check_path_collision(self,path):
        vehicles_info = self.veh_state.scene_state.vehicles_info
        segments_to_check = []
        if self.veh_state.direction in constants.PATH_COLLISION_MAP:
            segments_to_check = constants.PATH_COLLISION_MAP[self.veh_state.direction]
        if len(segments_to_check) == 0:
            return True
        for k,v in vehicles_info.items():
            if v[1] in segments_to_check:
                o_v_x,o_v_y = float(v[0][0]),float(v[0][1])
                ''' checking it from reverse might be more efficient '''
                for pt in list(reversed(path)):
                    if math.hypot(pt[0]-o_v_x, pt[1]-o_v_y) <= constants.COLLISION_CHECK_TOLERANCE:
                        return False
        return True
        
    
        
    def generate_velocity_profile(self,all_params):
        if self.veh_state.l1_action in ['wait-for-oncoming'] and self.veh_state.id==55 and self.veh_state.current_time>32:
            brk=1
        tagged_waypts,path_g,path_info,p_type = all_params[0], all_params[1], all_params[2], all_params[3]
        hpx = [x[0][0] for x in tagged_waypts]
        hpy = [x[0][1] for x in tagged_waypts]
        ''' insert a point between the first two waypoints to make it consistent in length with path'''
        #tagged_waypts = [tagged_waypts[0]] + [(((tagged_waypts[0][0][0]+tagged_waypts[1][0][0])/2, (tagged_waypts[1][0][1]+tagged_waypts[1][0][1])/2), tagged_waypts[0][1])] + tagged_waypts[1:]
        segment_tags = [(x[1],None,None) for x in tagged_waypts]
        target_vel_map_in_db = utils.get_target_velocity(self.veh_state)
        target_vel_map_for_path = dict()
        assign_val = None
        ''' get the target velocity for the waypoint segments '''
        for s in self.veh_state.segment_seq:
            if constants.SEGMENT_MAP[s] in target_vel_map_in_db:
                target_vel_map_for_path[s] = target_vel_map_in_db[constants.SEGMENT_MAP[s]]
                assign_val = target_vel_map_in_db[constants.SEGMENT_MAP[s]]
            else:
                target_vel_map_for_path[s] = assign_val
        ''' set the velocity at the starting point '''
        segment_tags[0] = (segment_tags[0][0],(self.veh_state.speed,0),self.veh_state.long_acc)
        ''' assign the target velocity to the last waypoint in each segment'''
        prev_val = self.veh_state.speed
        vel_inc = 1
        for idx,sp in enumerate(zip(segment_tags[:-1],segment_tags[1:])):
            if idx == 0:
                continue
            s1 = sp[0][0]
            s2 = sp[1][0]
            if s1 != s2:
                if prev_val == target_vel_map_for_path[s1][0]:
                    acc = 0
                elif prev_val <= target_vel_map_for_path[s1][0]:
                    acc = 1
                else:
                    acc = -1
                segment_tags[idx] = (segment_tags[idx][0],target_vel_map_for_path[s1],acc)
        exit_vel = None
        for s in reversed(segment_tags):
            if s[1] != None:
                exit_vel = s
                break
        exit_vel = (segment_tags[-1][0],exit_vel[1],0) if (s[0] == segment_tags[-1][0] or constants.SEGMENT_MAP[segment_tags[-1][0]] != 'exit-lane' or self.veh_state.l1_action in constants.WAIT_ACTIONS) \
                     else (segment_tags[-1][0], (exit_vel[1][0]+1,exit_vel[1][1]), 1)
        if self.veh_state.l1_action in constants.WAIT_ACTIONS:
            exit_vel = (segment_tags[-1][0],(0,0),-1) 
            if constants.SEGMENT_MAP[self.veh_state.current_segment] in constants.ENTRY_LANES:
                ''' if vehicle needs to stop in this segment, assign interpolated velocity values to the segments for the trajectory to move '''
                for s_t_idx,s_t in enumerate(segment_tags):
                    if s_t_idx+1 < len(segment_tags) and s_t[0] == self.veh_state.current_segment and s_t[1] is None and segment_tags[s_t_idx+1][1] is not None and segment_tags[s_t_idx+1][1][0]==0:
                        segment_tags[s_t_idx] = (segment_tags[-1][0],(min(self.veh_state.speed,1.5),0),-1)
                        break
        segment_tags[-1] = exit_vel
        ''' if the action is in wait action, assign 0 velocity to all segments after which the vehicle is already stopped '''
        if self.veh_state.l1_action in constants.WAIT_ACTIONS:
            assign_zero = False
            for idx,s in enumerate(segment_tags):
                if s[1]!= None and s[1][0] == 0:
                    assign_zero = True
                if assign_zero:
                    segment_tags[idx] = (segment_tags[idx][0],(0.0,0.0),0)
        ''' if the vehicle is in entry lane then assign current speed till it reaches the scene '''
        if constants.SEGMENT_MAP[self.veh_state.current_segment] in constants.ENTRY_LANES:
            entry_lane_idx = None
            for s_t_idx,s_t in enumerate(segment_tags):
                if s_t[0] == self.veh_state.current_segment and s_t_idx+1 < len(segment_tags)-1 and segment_tags[s_t_idx+1][0] != self.veh_state.current_segment:
                    entry_lane_idx = s_t_idx
                    break
            if entry_lane_idx is not None and entry_lane_idx > 6:
                for i_el_ct in np.arange(1,entry_lane_idx-6,1):
                    segment_tags[i_el_ct] = (segment_tags[i_el_ct][0],(self.veh_state.speed,0.0),0)
                
        indx = [idx for idx,s in enumerate(segment_tags) if s[1]!= None]
        num_tries = 20
        acc_lim = constants.MAX_LONG_ACC_NORMAL if self.veh_state.l2_action == 'NORMAL' else constants.MAX_LONG_ACC_AGGR
        jerk_lim = constants.MAX_TURN_JERK_NORMAL if self.veh_state.l2_action == 'NORMAL' else constants.MAX_TURN_JERK_AGGR
        all_velocity_profiles = dict()
        ''' increment the target velocty by +/- 1 if we are to generate boundary velocity profiles for each path '''
        if self.trajectory_type == 'BASELINE':
            inc_list = [0]
        elif self.trajectory_type == 'GAUSSIAN':
            X = get_truncated_normal(mean=0, sd=vel_inc/2, low=-vel_inc, upp=vel_inc)
            inc_list = X.rvs(3).tolist()
        else:
            inc_list = [-vel_inc,0,vel_inc] if self.veh_state.l1_action not in constants.WAIT_ACTIONS else [0]*3
        for inc in inc_list:
            candidate_profiles = []
            for tr_i in np.arange(num_tries):
                #if self.veh_state.l1_action in constants.WAIT_ACTIONS and not (constants.SEGMENT_MAP[self.veh_state.current_segment] == 'left-turn-lane' and self.veh_state.l1_action=='wait-for-oncoming'):
                #    continue
                v_x = []
                for idx,s in enumerate(segment_tags):
                    if s[1]!= None:
                        if s[1][0] == 0:
                            v_x.append(0)
                        else:
                            v_x.append(np.random.normal(max(s[1][0]+inc,0),s[1][1]/2))
                cs_v = CubicSpline(indx,v_x)
                regen_vx = [max(float(cs_v(x)),0.0) if float(cs_v(x)) > .05 else 0.0 for x in np.arange(len(segment_tags))]
                
                #X = np.arange(0,float('%.1f'%(len(path_g)/10)),.1)
                X = [(x/len(path_g))*len(segment_tags) for x in np.arange(0,len(path_g),1)]
                Y = [max(float(cs_v(x)),0.0) if float(cs_v(x)) > .05 else 0.0 for x in X]
                if self.veh_state.l1_action in constants.WAIT_ACTIONS:
                    zero_index = False
                    for y_val_idx,y_val in enumerate(Y):
                        if y_val == 0:
                            zero_index = True
                        else:
                            if zero_index:
                                Y[y_val_idx] = 0.0 
                cs_d_a = cs_v.derivative()
                Y_prime = [float(cs_d_a(x)) for x in X]
                cs_d2_a = cs_d_a.derivative()
                Y_2prime = [float(cs_d2_a(x)) for x in X]
                #if max(Y_prime) < acc_lim+inc and min(Y_prime) > -acc_lim-inc and max(Y_2prime) < jerk_lim+inc and min(Y_2prime) > -jerk_lim-inc:
                max_coeff = np.max(np.abs(cs_v.c[-1,:]))
                candidate_profiles.append(((hpx, hpy, regen_vx), max_coeff, (max([abs(x) for x in Y_prime]),max([abs(x) for x in Y_2prime]))))
                    #log.info('velocity profile generation candidate found: '+str(tr_i))
                #else:
                #    f=1
                    #log.info('rejected acc lim'+str(max(Y_prime))+','+str(min(Y_prime))+' jerk lim '+str(max(Y_2prime))+','+str(min(Y_2prime)))
            if len(candidate_profiles) > 0:
                candidate_profiles.sort(key=lambda tup: (tup[2][0],tup[2][1],tup[1]))
                all_velocity_profiles[inc] = candidate_profiles
            else:
                all_velocity_profiles[inc] = None
        '''
        plt.plot(X, [y[1] for y in candidate_profiles[0][0]],'blue')
        plt.plot(X, [y[1] for y in candidate_profiles[-1][0]],'red')
        plt.show()
        plt.plot(X,[y[1] for y in candidate_profiles[0][1]],'blue')
        plt.plot(X,[y[1] for y in candidate_profiles[-1][1]],'red')
        plt.show()
        plt.plot(X,[y[1] for y in candidate_profiles[0][2]],'blue')
        plt.plot(X,[y[1] for y in candidate_profiles[-1][2]],'red')
        plt.show()
        '''
        
        ''' generate the trajectories and set them in path_info'''
        all_trajs = []
        for inc_step,all_candidate_profiles in all_velocity_profiles.items():
            if candidate_profiles is None:
                path_f = path_g
                path_X,path_Y = utils.fresnet_to_map(self.veh_state.x, self.veh_state.y, [x[0] for x in path_f], [x[1] for x in path_f], self.veh_state.yaw)
                path_m = list(zip(path_X,path_Y))
                res = self.generate_baseline_trajectory(self.veh_state, path_m, inc_step)
                time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                all_trajs.append(('baseline', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
            else:
                generated_with_lims = False
                if not (self.veh_state.l2_action != 'NORMAL' and self.veh_state.l1_action in constants.WAIT_ACTIONS):
                    for idx,candidate_profiles in enumerate(all_candidate_profiles):
                        if candidate_profiles[2][0] < acc_lim+inc_step and candidate_profiles[2][1] < jerk_lim+inc_step:
                            res = self.regenerate_path(candidate_profiles[0][0],candidate_profiles[0][1],candidate_profiles[0][2])
                            time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                            all_trajs.append(('regenerated', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
                            generated_with_lims = True
                            break
                if not generated_with_lims or (self.veh_state.l2_action != 'NORMAL' and self.veh_state.l1_action in constants.WAIT_ACTIONS):
                    if self.veh_state.l2_action != 'NORMAL' and self.veh_state.l1_action in constants.WAIT_ACTIONS:
                        sorted_profile = sorted(all_candidate_profiles, key=lambda tup: (-tup[2][1],-tup[2][0],tup[1]))
                        if max(sorted_profile[0][0][2]) < 35:
                            log.info('generated_with_lims with limits acc: '+str(sorted_profile[0][2][0])+' jerk:'+str(sorted_profile[0][2][1]))
                            res = self.regenerate_path(sorted_profile[0][0][0],sorted_profile[0][0][1],sorted_profile[0][0][2])
                            time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                            all_trajs.append(('baseline', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
                        else:
                            path_f = path_g
                            path_X,path_Y = utils.fresnet_to_map(self.veh_state.x, self.veh_state.y, [x[0] for x in path_f], [x[1] for x in path_f], self.veh_state.yaw)
                            path_m = list(zip(path_X,path_Y))
                            res = self.generate_baseline_trajectory(self.veh_state, path_m, inc_step)
                            time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                            all_trajs.append(('baseline', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
                    else:
                        if max(all_candidate_profiles[0][0][2]) < 35:
                            res = self.regenerate_path(all_candidate_profiles[0][0][0],all_candidate_profiles[0][0][1],all_candidate_profiles[0][0][2])
                            time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                            all_trajs.append(('regenerated', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
                        else:
                            path_f = path_g
                            path_X,path_Y = utils.fresnet_to_map(self.veh_state.x, self.veh_state.y, [x[0] for x in path_f], [x[1] for x in path_f], self.veh_state.yaw)
                            path_m = list(zip(path_X,path_Y))
                            res = self.generate_baseline_trajectory(self.veh_state, path_m, inc_step)
                            time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
                            all_trajs.append(('baseline', [time_arr,x_arr,y_arr,yaw_profile,v_arr,a_arr,j_arr]))
                    
                    '''
                    plt.figure()
                    plt.title('beyond limits')
                    plt.plot(time_arr,v_arr)
                    plt.show()
                    '''
                    generated_with_lims = True
            
    
        '''
        if self.veh_state.l1_action in constants.WAIT_ACTIONS and self.veh_state.current_time > 30:
            plt.figure()
            for _type, res in all_trajs:
                plt.title(self.veh_state.l1_action + ' ' + str(_type))
                #_x,_y = utils.fresnet_to_map(self.veh_state.x, self.veh_state.y, res[1], res[2], self.veh_state.yaw)
                _x,_y = res[1], res[2]
                visualizer.plot_traffic_regions()
                plt.plot(_x,_y,'black')
            plt.figure()
            for _type, res in all_trajs:
                plt.title(self.veh_state.l1_action + ' ' + str(_type))
                plt.plot(res[0],res[4])
            plt.show()
        '''
        if 'proceed' in self.veh_state.l1_action:
            brk = 1
        
        for _type,res in all_trajs:
            if p_type not in self.trajectories:
                self.trajectories[p_type] = [res]
            else:
                self.trajectories[p_type].append(res)
        return [x[0] for x in all_trajs],p_type
    
    def regenerate_path(self,hpx,hpy,vx):
        if self.veh_state.l1_action in constants.WAIT_ACTIONS and self.veh_state.current_time > 30:
            brk=1
        hpx_f,hpy_f = utils.map_to_fresnet(self.veh_state.x, self.veh_state.y, hpx, hpy, self.veh_state.yaw)
        TX = np.arange(constants.LP_FREQ, constants.OTH_AGENT_L3_ACT_HORIZON, constants.LP_FREQ)
        if vx is None:
            sys.exit('velocity profile for regeneration is None')
        assert len(hpx_f) == len(hpy_f) == len(vx), 'path and velocity sizes do not match'
        if self.veh_state.l1_action in constants.WAIT_ACTIONS:
            hpx_p,hpy_p,vx_p = [],[],[]
            for idx,v in enumerate(vx):
                if v > 0:
                    hpx_p.append(hpx_f[idx])
                    hpy_p.append(hpy_f[idx])
                    vx_p.append(vx[idx])
                else:
                    hpx_p.append(hpx_f[idx])
                    hpy_p.append(hpy_f[idx])
                    vx_p.append(vx[idx])
                    break
            if len(hpx_p) < 2:
                path = [(hpx_p[0],hpy_p[0])]
            else:
                path = [(hpx_p[1],hpy_p[1])]
        else:
            hpx_p,hpy_p,vx_p = hpx_f,hpy_f,vx
        vel_arr = [vx_p[0]]
        acc_arr = [self.veh_state.long_acc]
        jerk_arr = [acc_arr[0]/10]
        yaw_arr = [self.veh_state.yaw]
        time_arr = [0.0]
        path = [(hpx_p[0],hpy_p[0])]
        if len(hpx_p) >= 2:
            #plt.plot(hpx_f,hpy_f,'x')
            #path_prefix = [(hpx_p[0],hpy_p[0])]
            #hpx_p,hpy_p = hpx_p[1:],hpy_p[1:]
            s_x = [0] + [p2-p1 for p1,p2 in list(zip(hpx_p[:-1],hpx_p[1:]))]
            s_y = [0] + [p2-p1 for p1,p2 in list(zip(hpy_p[:-1],hpy_p[1:]))]
            t_x = [0]
            for indx,s in enumerate(zip(s_x,s_y)):
                dist = math.hypot(s[0], s[1])
                if indx > 0:
                    v,u = vx_p[indx], vx_p[indx-1]
                    t = abs((2*dist)/(v+u)) if v+u != 0 else 0.1
                    t_x.append(t_x[-1]+t)
            s_x = [0] + [sum(s_x[:i+1]) for i in np.arange(1,len(s_x))]
            s_y = [0] + [sum(s_y[:i+1]) for i in np.arange(1,len(s_y))]
            indx = np.arange(len(s_x))
            cs_x = CubicSpline(indx,s_x)
            cs_y = CubicSpline(indx,s_y)
            cs_v = CubicSpline(indx,vx_p)
            cs_acc = cs_v.derivative()
            cs_j = cs_acc.derivative()
            cs_t = interp1d(t_x,indx,bounds_error=False,fill_value='extrapolate')
            ref_x = np.arange(indx[1],indx[-1]+.1,.1)
            
            for t in TX:
                indx_val = float(cs_t(t))
                if indx_val > indx[-1]:
                    ''' this has not been fitted '''
                    break
                time_arr.append(round(t,1))
                path.append(((path[0][0]+cs_x(indx_val)), ((path[0][1]+cs_y(indx_val)))))
                v = max(float(cs_v(indx_val)),0.0) if float(cs_v(indx_val)) > .05 else 0.0
                a = float(cs_acc(indx_val))
                j = float(cs_j(indx_val))
                vel_arr.append(v)
                acc_arr.append(a)
                jerk_arr.append(j)
                if len(path) > 1:
                    _y = math.atan2(path[-1][1]-path[-2][1], path[-1][0]-path[-2][0])
                    _y = _y if _y > 0 else (2*math.pi)-abs(_y)
                else:
                    _y = self.veh_state.yaw
                yaw_arr.append(_y)
        if time_arr[-1] < TX[-1] and self.veh_state.l1_action in constants.WAIT_ACTIONS:
            ''' vehicle is stopped, extend the trajectory '''
            for t in np.arange(time_arr[-1]+constants.LP_FREQ,TX[-1]+constants.LP_FREQ,constants.LP_FREQ):
                time_arr.append(round(t,1))
                path.append(path[-1])
                v = 0.0
                a = 0.0
                j = 0.0
                vel_arr.append(v)
                acc_arr.append(a)
                jerk_arr.append(j)
                yaw_arr.append(yaw_arr[-1])
                
            
            '''        
            if self.veh_state.l1_action in constants.WAIT_ACTIONS and self.veh_state.current_time > 30:
                plt.figure()
                plt.plot(hpx_f,hpy_f,'x')
                plt.plot([x[0] for x in path],[x[1] for x in path],'black')
                #plt.show()
            '''
        x_arr, y_arr = utils.fresnet_to_map(self.veh_state.x, self.veh_state.y, [x[0] for x in path],[x[1] for x in path], self.veh_state.yaw)
        time_arr = [x+self.veh_state.current_time for x in time_arr]
        return [time_arr,x_arr,y_arr,yaw_arr,vel_arr,acc_arr,jerk_arr]
        
    
    def generate_path(self,hpx,hpy,collision_check):
        #plt.plot(hpx,hpy,'x')
        path_prefix = utils.split_in_n((hpx[0],hpy[0]), (hpx[1],hpy[1]), 2)
        path = [(hpx[1],hpy[1])]
        hpx,hpy = hpx[1:],hpy[1:]
        max_coeff = 0
        s_x = [0] + [p2-p1 for p1,p2 in list(zip(hpx[:-1],hpx[1:]))]
        s_x = [0] + [sum(s_x[:i+1]) for i in np.arange(1,len(s_x))]
        s_y = [0] + [p2-p1 for p1,p2 in list(zip(hpy[:-1],hpy[1:]))]
        s_y = [0] + [sum(s_y[:i+1]) for i in np.arange(1,len(s_y))]
        indx = np.arange(len(s_x))
        if len(indx) > 1:
            cs_x = CubicSpline(indx,s_x)
            cs_y = CubicSpline(indx,s_y)
            ref_x = np.arange(indx[1],indx[-1]+.1,.1)
            for i_a in ref_x:
                path.append(((path[0][0]+cs_x(i_a)), ((path[0][1]+cs_y(i_a)))))
        #for i_a in ref_x:
        #    path.append(((path[-1][0]+cs_x(i_a)), ((path[-1][1]+cs_y(i_a)))))
            max_coeff = max(np.max(np.abs(cs_x.c[-1,:])), np.max(np.abs(cs_y.c[-1,:])))
            path = path_prefix + path
        else:
            path = path_prefix
        path = list(OrderedDict.fromkeys(path))
        
        #plt.plot([x[0] for x in path],[x[1] for x in path])
        #plt.show()
        if collision_check:
            if self.check_path_collision(path):
                return path,max_coeff
            else:
                return None,None
        else:
            return path,max_coeff
        
        
    
    def construct_waypoints(self,p_segment,veh_state,seg_info,hpx,hpy,seg_hpx,seg_hpy):
        if constants.SEGMENT_MAP[p_segment] == 'exit-lane' and veh_state.task != 'STRAIGHT':
            if p_segment == veh_state.current_segment:
                centerline_l = utils.get_forward_line((veh_state.x,veh_state.y), veh_state.yaw, utils.get_centerline(p_segment[:-2]+'-1'))
                centerline_r = utils.get_forward_line((veh_state.x,veh_state.y), veh_state.yaw, utils.get_centerline(p_segment[:-2]+'-2'))
                if len(centerline_l) == len(centerline_r):
                    centerline = [((x1[0]+x2[0])/2, (x1[1]+x2[1])/2) for x1,x2 in zip(centerline_l,centerline_r)]
                else:
                    centerline = utils.get_forward_line((veh_state.x,veh_state.y), veh_state.yaw, utils.get_centerline(p_segment))
            else:
                centerline = utils.get_centerline(p_segment)
        else:
            if p_segment == veh_state.current_segment:
                centerline = utils.get_forward_line((veh_state.x,veh_state.y), veh_state.yaw, utils.get_centerline(p_segment))
            else:
                centerline = utils.get_centerline(p_segment)
        ''' add lattice points along the centerline '''
        seg_info.append_segment(p_segment, centerline)
        for cl_pt in centerline:
            min_hpx_dist = constants.CAR_LENGTH/2 if veh_state.task == 'RIGHT_TURN' else constants.CAR_LENGTH
            if math.hypot(cl_pt[0]-hpx[-1],cl_pt[1]-hpy[-1]) > min_hpx_dist:
                if len(hpx) > 1:
                    angle_bet_lines = utils.angle_between_lines_2pi([(hpx[-2],hpy[-2]),(hpx[-1],hpy[-1])], [(hpx[-1],hpy[-1]),cl_pt])
                ''' add only if it is ahead in the path'''
                if (len(hpx) > 1 and (angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2)) or len(hpx) <= 1:
                    if math.hypot(cl_pt[0]-hpx[-1],cl_pt[1]-hpy[-1]) > 1*constants.CAR_LENGTH:
                        seg_l = math.hypot(cl_pt[0]-hpx[-1],cl_pt[1]-hpy[-1])
                        seg_pts = utils.split_in_n((hpx[-1],hpy[-1]), cl_pt, int(seg_l//constants.CAR_LENGTH))
                        for s_p in seg_pts:
                            ''' check again since dist might be less after splitting '''
                            if math.hypot(s_p[0]-hpx[-1],s_p[1]-hpy[-1]) > min_hpx_dist:
                                hpx.append(s_p[0])
                                hpy.append(s_p[1])
                                seg_hpx.append(s_p[0])
                                seg_hpy.append(s_p[1])
                hpx.append(cl_pt[0])
                hpy.append(cl_pt[1])
                seg_hpx.append(cl_pt[0])
                seg_hpy.append(cl_pt[1])
            else:
                seg_hpx.append(cl_pt[0])
                seg_hpy.append(cl_pt[1])
        return seg_info
    
    
    def construct_trajectories(self,veh_state):
        l1_action = veh_state.l1_action
        yaw = veh_state.yaw
        hpx,hpy = [veh_state.x,veh_state.x+((constants.CAR_LENGTH/2)*np.cos(yaw))],[veh_state.y,veh_state.y+((constants.CAR_LENGTH/2)*np.sin(yaw))]
        path = None
        if veh_state.current_time==32.866167 and veh_state.id==44:
            brk=1
        centerline_based = False
        path_info = PathInfo(self)
        if l1_action == 'proceed-turn' or l1_action == 'wait-for-oncoming' or l1_action == 'wait-for-pedestrian' \
            or l1_action == 'track_speed' or l1_action == 'follow_lead' or l1_action == 'decelerate-to-stop' or l1_action == 'wait-on-red' or l1_action == 'yield-to-merging'\
                or l1_action == 'follow_lead_into_intersection' or l1_action == 'wait_for_lead_to_cross':
            centerline_based = True
            path_info = PathInfo(self)
            segments_in_path = veh_state.segment_seq[veh_state.segment_seq.index(veh_state.current_segment):]
            for p_segment in segments_in_path:
                seg_hpx,seg_hpy = [],[]
                '''
                if veh_state.l1_action in constants.WAIT_ACTIONS:
                    if veh_state.l1_action in constants.STOP_SEGMENT:
                        if p_segment != veh_state.current_segment and constants.SEGMENT_MAP[p_segment] in constants.STOP_SEGMENT[veh_state.l1_action]:
                            break
                '''
                if veh_state.l1_action == 'follow_lead_into_intersection' or veh_state.l1_action == 'wait_for_lead_to_cross':
                    ''' only add the waypoints for the current segment '''
                    lead_veh_segment = veh_state.leading_vehicle.current_segment
                    if p_segment != veh_state.current_segment or (lead_veh_segment == veh_state.current_segment or math.hypot(veh_state.leading_vehicle.x-veh_state.x, veh_state.leading_vehicle.y-veh_state.y) < (constants.CAR_LENGTH*2)):
                        break
                path_info = self.construct_waypoints(p_segment,veh_state,path_info,hpx,hpy,seg_hpx,seg_hpy)
                path_info.add_waypoints_for_segment(p_segment, list(zip(seg_hpx,seg_hpy)))
            if l1_action == 'follow_lead_into_intersection' or l1_action == 'wait_for_lead_to_cross':
                ''' now add the lead vehicle positions '''
                end_pos_but_last = (veh_state.leading_vehicle.x, veh_state.leading_vehicle.y)
                end_pos = (end_pos_but_last[0] + ((constants.CAR_LENGTH/2) * np.cos(veh_state.leading_vehicle.yaw)), end_pos_but_last[1] + ((constants.CAR_LENGTH/2) * np.sin(veh_state.leading_vehicle.yaw)))
                hpx.append(end_pos_but_last[0])
                seg_hpx.append(end_pos_but_last[0])
                hpy.append(end_pos_but_last[1])
                seg_hpy.append(end_pos_but_last[1])
                if veh_state.leading_vehicle.current_segment not in path_info.segments:
                    path_info.append_segment(veh_state.leading_vehicle.current_segment, [end_pos_but_last,end_pos])
                path_info.add_waypoints_for_segment(veh_state.leading_vehicle.current_segment, list(zip(seg_hpx,seg_hpy)))
        elif l1_action == 'cut-in':
            proceed_pos_ends = get_exitline_from_direction(veh_state.direction,veh_state)
            proceed_pos_center = ((proceed_pos_ends[0][0]+proceed_pos_ends[1][0])/2, (proceed_pos_ends[0][1]+proceed_pos_ends[1][1])/2) 
            if math.hypot(proceed_pos_center[0]-veh_state.x, proceed_pos_center[1]-veh_state.y) < constants.CAR_LENGTH/2:
                next_segment = veh_state.segment_seq[min(veh_state.segment_seq.index(veh_state.current_segment)+1, len(veh_state.segment_seq)-1)]
                proceed_pos_ends = get_exitline(next_segment)
            procced_line_len = math.hypot(proceed_pos_ends[0][0]-proceed_pos_ends[1][0], proceed_pos_ends[0][1]-proceed_pos_ends[1][1])
            if procced_line_len > constants.LANE_WIDTH*1.5:
                if veh_state.segment_seq[-1][-2:] == '-1':
                    proceed_pos_ends = [(proceed_pos_ends[0][0],proceed_pos_ends[0][1]), (proceed_pos_ends[0][0]+proceed_pos_center[0]/2, proceed_pos_ends[0][1]+proceed_pos_center[1]/2)]
                else:
                    [(proceed_pos_ends[1][0]+proceed_pos_center[0]/2, proceed_pos_ends[1][1]+proceed_pos_center[1]/2), (proceed_pos_ends[1][0],proceed_pos_ends[1][1]) ]
                proceed_pos_center = ((proceed_pos_ends[0][0]+proceed_pos_ends[1][0])/2, (proceed_pos_ends[0][1]+proceed_pos_ends[1][1])/2)
            proceed_pos_yaw = math.atan2(proceed_pos_ends[1][0]-proceed_pos_ends[0][0], proceed_pos_ends[1][1]-proceed_pos_ends[1][1]) + (math.pi/2)
            end_pos = proceed_pos_center
            end_pos_but_last = (end_pos[0]-((constants.CAR_LENGTH/2)*np.cos(proceed_pos_yaw)) , end_pos[1]-((constants.CAR_LENGTH/2)*np.sin(proceed_pos_yaw)))
            hpx.append(end_pos_but_last[0])
            hpx.append(end_pos[0])
            hpy.append(end_pos_but_last[1])
            hpy.append(end_pos[1])
        else:
            raise ValueError(l1_action+" l1_action not implemented")
        path_list = {'baseline':[], 'boundary':[], 'gaussian':[]}
        if centerline_based:
            selected_path = ([],np.inf)
            ''' forward centerline without the current vehicle location'''
            if len(hpx) < 3:
                hpx_f,hpy_f = utils.map_to_fresnet(veh_state.x, veh_state.y, hpx, hpy, veh_state.yaw)
                path_f = utils.split_in_n((hpx_f[0],hpy_f[0]), (hpx_f[1],hpy_f[1]), 2)
                x_m,y_m = utils.fresnet_to_map(veh_state.x, veh_state.y, [x[0] for x in path_f], [x[1] for x in path_f], veh_state.yaw)
                path_m = list(zip(x_m,y_m))
                tagged_wp = [(x,self.veh_state.current_segment) for x in path_m]
                path_list['baseline'].append((path_f,tagged_wp))
                if self.generate_boundary:
                    path_list['boundary'].append((path_f,tagged_wp))
                    path_list['boundary'].append((path_f,tagged_wp))
                elif self.trajectory_type == 'GAUSSIAN':
                    path_list['gaussian'].append((path_f,tagged_wp))
                    path_list['gaussian'].append((path_f,tagged_wp))
            else:
                if self.generate_boundary:
                    centerline_list = {'lateral_waypoints':[], 'boundary_waypoints':[]} 
                elif self.trajectory_type == 'GAUSSIAN':
                    centerline_list = {'gaussian_waypoints':[]}
                else:
                    centerline_list = {'lateral_waypoints':[]} 
                for w_type, cl_list in centerline_list.items():
                    ''' lateral_waypoints already contains the baseline path waypoints'''
                    for k,v in path_info.segments.items():
                        lwp = v[w_type]
                        for idx,wp in enumerate(lwp):
                            if len(cl_list) <= idx:
                                cl_list.append([(x,k) for x in wp])
                            else:
                                cl_list[idx].extend([(x,k) for x in wp])
                    ''' remove the lattice points that are either too close or behind'''
                    for cl_idx,cl in enumerate(cl_list):
                        cleaned_list = [((hpx[0],hpy[0]),veh_state.current_segment), ((hpx[1],hpy[1]),veh_state.current_segment)]
                        min_hpx_dist = constants.CAR_LENGTH/2 if (veh_state.task == 'RIGHT_TURN' or veh_state.direction=='L_W_N') else constants.CAR_LENGTH
                        for cl_pt,seg in cl:
                            angle_bet_lines = utils.angle_between_lines_2pi([cleaned_list[-2][0],cleaned_list[-1][0]], [cleaned_list[-1][0],cl_pt])
                            if math.hypot(cl_pt[0]-cleaned_list[-1][0][0],cl_pt[1]-cleaned_list[-1][0][1]) > min_hpx_dist and (angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2):
                                    cleaned_list.append((cl_pt,seg))
                        cl_list[cl_idx] = cleaned_list
                        
                    
                ''' generate paths - either only baseline or along with boundary paths'''
                
                for w_type, cl_list in centerline_list.items():
                    if w_type == 'lateral_waypoints':
                        p_type = 'baseline' 
                    elif w_type == 'gaussian_waypoints':
                        p_type = 'gaussian'
                    else:
                        p_type = 'boundary'
                    for m_idx,l in enumerate(cl_list):
                        hpx_l,hpy_l = utils.map_to_fresnet(veh_state.x, veh_state.y, [x[0][0] for x in l], [x[0][1] for x in l], veh_state.yaw)
                        cl_mx,cl_my = [x[0][0] for x in l], [x[0][1] for x in l]
                        '''
                        if True:
                            plt.plot(cl_mx,cl_my,'x')
                        '''
                        path_g,max_coeff = self.generate_path(hpx_l, hpy_l, True)
                        ''' we select the path that has the least 3rd order coefficient '''
                        if p_type == 'baseline':
                            if path_g is not None and max_coeff is not None and max_coeff < selected_path[1]:
                                selected_path = (path_g,max_coeff,l)
                        else:
                            if path_g is not None:
                                path_list[p_type].append((path_g,l))
                        
                    ''' baseline paths have not been added to the path list. Add it now.'''
                    if p_type == 'baseline':
                        path_f,path_max_coeff,tagged_wp = selected_path[0],selected_path[1],selected_path[2]
                        path_list[p_type].append((path_f,tagged_wp))
                    
        else:
            hpx_f,hpy_f = utils.map_to_fresnet(veh_state.x, veh_state.y, hpx, hpy, veh_state.yaw)
            path_f,max_coeff = self.generate_path(hpx_f, hpy_f, False)
            x_m,y_m = utils.fresnet_to_map(veh_state.x, veh_state.y, [x[0] for x in path_f], [x[1] for x in path_f], veh_state.yaw)
            path_m = list(zip(x_m,y_m))
            tagged_wp = [(x,self.veh_state.current_segment) for x in path_m]
            path_list['baseline'].append((path_f,tagged_wp))
            if self.generate_boundary:
                path_list['boundary'].append((path_f,tagged_wp))
                path_list['boundary'].append((path_f,tagged_wp))
            elif self.trajectory_type == 'GAUSSIAN':
                path_list['gaussian'].append((path_f,tagged_wp))
                path_list['gaussian'].append((path_f,tagged_wp))
        path_info.all_paths = path_list
        all_params = []
        for p_type,path_l in path_list.items():
            for path_f,tagged_wp in path_l:
                all_params.append([tagged_wp,path_f,path_info,p_type])
        cmp = CustomMPS()
        def callback(res,p_type):
            if p_type not in self.trajectories:
                self.trajectories[p_type] = [r for r in res]
            else:
                self.trajectories[p_type].extend(res)
            
        #cmp.execute_with_callback(self.generate_velocity_profile, all_params, callback)
        for par in all_params:
            #x_m,y_m = utils.fresnet_to_map(veh_state.x, veh_state.y, [x[0] for x in par[1]], [x[1] for x in par[1]], veh_state.yaw)
            #path_m = list(zip(x_m,y_m))
            #plt.plot([x[0][0] for x in par[0]], [x[0][1] for x in par[0]], 'x')
            #plt.plot(x_m,y_m)
            self.generate_velocity_profile(par)
        #plt.show()
        #self.generate_velocity_profile(tagged_wp,path_f,path_info,p_type)
        
        
        return self.trajectories
    
    
    def generate_baseline_trajectory(self,veh_state,baseline_path,inc):
        dt = constants.LP_FREQ
        l1_action = self.l1_action
        l2_action = self.l2_action
        if l2_action == 'AGGRESSIVE':
            max_acc = ((constants.MAX_LONG_ACC_AGGR - constants.MAX_LONG_ACC_NORMAL)/2) + inc 
            max_dec = ((constants.MAX_LONG_DEC_AGGR + constants.MAX_LONG_DEC_NORMAL)/2) - inc
            max_acc_jerk = constants.MAX_ACC_JERK_AGGR + inc
            max_dec_jerk = constants.MAX_DEC_JERK_AGGR - inc
        else:
            max_acc, max_dec, max_acc_jerk, max_dec_jerk = constants.MAX_LONG_ACC_NORMAL + inc, (constants.MAX_LONG_DEC_NORMAL/2) - inc, constants.MAX_ACC_JERK_NORMAL + inc, constants.MAX_DEC_JERK_NORMAL-inc
        time_tx = np.arange(veh_state.current_time,veh_state.current_time+constants.OTH_AGENT_L3_ACT_HORIZON,constants.LP_FREQ)
        if l1_action == 'decelerate-to-stop' or l1_action == 'wait-on-red' or l1_action == 'wait_for_lead_to_cross' or l1_action == 'wait-for-oncoming' or l1_action == 'yield-to-merging' \
            or l1_action == 'wait-for-pedestrian':
            time_tx,V, path = utils.generate_baseline_trajectory(time_tx,baseline_path,veh_state.speed,veh_state.long_acc,max_dec,max_dec_jerk,0.0,constants.LP_FREQ,False)
        elif l1_action == 'proceed-turn' or l1_action == 'track_speed' or l1_action == 'follow_lead' or l1_action == 'follow_lead_into_intersection' or l1_action == 'cut-in':
            if l1_action == 'proceed-turn':
                if veh_state.task == 'RIGHT_TURN':
                    v_g = 7 + inc
                else:
                    v_g = 8.8 + inc
            elif l1_action == 'follow_lead' or l1_action == 'follow_lead_into_intersection':
                v_g = veh_state.leading_vehicle.speed
            elif l1_action == 'cut-in':
                v_g = min(8.8, math.sqrt(veh_state.speed**2 + 2*max((veh_state.long_acc+max_acc)/2,0)*math.hypot(veh_state.x-baseline_path[-1][0], veh_state.y-baseline_path[-1][1])))
            else:
                v_g = max(16,veh_state.speed)
            time_tx, V, path = utils.generate_baseline_trajectory(time_tx,baseline_path,veh_state.speed,veh_state.long_acc,max_acc,max_acc_jerk,v_g,constants.LP_FREQ,True)
        else:
            raise ValueError(l1_action+" l1 action not implemented")
        yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(path[:-1],path[1:])]
        rv = V
        ra = [abs(x[1]-x[0])/dt for x in zip(rv[:-1],rv[1:])]
        ra = ra + [ra[-1]]
        rj = [abs(x[1]-x[0])/dt for x in zip(ra[:-1],ra[1:])]
        rj = rj +[rj[-1]]
        res = [time_tx, [x[0] for x in path], [x[1] for x in path],yaw,rv,ra,rj]
        return res  
    
    def generate(self):
        veh_state = self.veh_state
        if utils.is_out_of_view((veh_state.x,veh_state.y)) and constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
            ''' vehicle has exited the scene so there is no need to generate the trajectory'''
            return None
        l1_action = self.l1_action
        l2_action = self.l2_action
        veh_state.set_current_l1_action(l1_action)
        veh_state.set_current_l2_action(l2_action)
        if l1_action == 'wait-for-pedestrian' and veh_state.id == 9:
            brk=1
        
        self.construct_trajectories(veh_state)
        '''
        clipped_res = utils.clip_trajectory_to_viewport(res)
        if clipped_res is not None:
            res =  clipped_res
        '''
        '''
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point'"
        c.execute(q_string)
        q_res = c.fetchall()
        plt.axis("equal")
        for row in q_res:
            plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        plt.plot(res[1], res[2])
        plt.show()
        _sl = min(len(time_tx),len(rv))
        plt.plot(time_tx[:_sl],rv[:_sl])
        plt.show()
        '''
    '''
    def generate_trajectory(self):
        veh_state = self.veh_state
        this tuple is to flag whether trajectories were found or not (in_view_trajectories,clipped_trajectories)
        #boundary_state_list += get_wait_states(veh_state, init_segment)
        #boundary_state_list += get_proceed_states(veh_state, init_segment)
        res = self.generate()
    ''' 
        
    
    def generate_trajectory_deprecated(self,veh_state):
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
        traj_list = []
        if self.baseline_only:
            res = self.generate_baseline(veh_state)
            if res is not None:
                traj_list.append((res, 'BASELINE'))
        else:
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
                sys.exit('action '+self.l1_action+' is not implemented')
                
            if len(traj_list) > 0:
                traj_generated = (True, traj_generated[1])
        return np.asarray(traj_list)

    def __init__(self,l1_action,l2_action,task,trajectory_type,veh_state,veh_id,relev_veh_id):
        self.l1_action = l1_action
        self.l2_action = l2_action
        self.task = task
        self.generate_boundary = True if trajectory_type == 'BOUNDARY' else False
        self.trajectory_type = trajectory_type
        self.veh_state = veh_state
        self.trajectories = {'baseline':[], 'boundary':[], 'gaussian':[]}
        self.veh_id = veh_id
        self.relev_veh_id = relev_veh_id
        
    
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

def get_exitsegment_from_direction(direction,veh_state):
    if constants.SEGMENT_MAP[veh_state.current_segment] == 'left-turn-lane':
        exit_segment = 'prep-turn_'+direction[2].lower()
    elif constants.SEGMENT_MAP[veh_state.current_segment] == 'right-turn-lane':
        exit_segment = 'rt_exec-turn_'+direction[2].lower()
    else:
        '''
        if constants.SEGMENT_MAP[veh_state.current_segment] != 'exit-lane':
            if constants.TASK_MAP[direction] == 'RIGHT_TURN':
                exit_segment = 'rt_exec-turn_'+direction[2].lower()
            else:
                exit_segment = 'exec-turn_'+direction[2].lower()
        else:
            exit_segment = veh_state.current_segment
        '''
        ''' next segment '''
        if constants.SEGMENT_MAP[veh_state.current_segment] == 'prep-left-turn':
            exit_segment = 'exec-turn_'+direction[2].lower()
        elif constants.SEGMENT_MAP[veh_state.current_segment] == 'prep-right-turn':
            exit_segment = 'rt_exec-turn_'+direction[2].lower()
        else:    
            exit_segment = veh_state.current_segment
    return exit_segment

def get_exitline_from_direction(direction,veh_state):
    exit_segment = get_exitsegment_from_direction(direction, veh_state)
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