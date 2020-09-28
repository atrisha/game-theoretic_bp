'''
Created on Mar 11, 2020

@author: authorA
'''

import numpy as np
from all_utils import utils
import constants
from scipy.interpolate import CubicSpline,interp1d,splev, splrep, BSpline, UnivariateSpline
import math
import matplotlib.pyplot as plt
import sys
from tkinter.ttk import _val_or_dict
from collections import OrderedDict


def decelerate_to_stop_planner_cp(veh_state, center_line, dist_to_stop):
    max_acc_lon, max_acc_lat, target_vel = constants.MAX_LONG_ACC_NORMAL if veh_state.l2_action=='NORMAL' else constants.MAX_LONG_ACC_AGGR, constants.MAX_LAT_ACC_NORMAL if veh_state.l2_action=='NORMAL' else constants.MAX_LAT_ACC_AGGR, 0.0
    dt = constants.LP_FREQ
    sx, sy, syaw, sv, sax, say = veh_state.x, veh_state.y, veh_state.yaw, veh_state.speed, veh_state.long_acc, veh_state.tan_acc
    if veh_state.l2_action == 'NORMAL':
        max_acc_lims,max_acc_jerk_lims = constants.MAX_LONG_ACC_NORMAL,constants.MAX_ACC_JERK_NORMAL
    else:
        max_acc_lims,max_acc_jerk_lims = constants.MAX_LONG_ACC_AGGR,constants.MAX_ACC_JERK_AGGR 
    center_line_angle = math.atan2((center_line[1][1]-center_line[0][1]),(center_line[1][0]-center_line[0][0]))
    dist_to_centerline = utils.distance_numpy([center_line[0][0],center_line[0][1]],[center_line[1][0],center_line[1][1]],[sx,sy])
    sv_angle_with_cl = syaw - center_line_angle
    sv_angle_in_map = math.atan2(sy,sx)
    fresnet_origin = (sx - dist_to_centerline*math.cos(sv_angle_in_map), sy - dist_to_centerline*math.sin(sv_angle_in_map))
    sv_x_fresnet,sv_y_fresnet = 0,dist_to_centerline
    hpx,hpy = [sv_x_fresnet,sv_y_fresnet]
    centerline_merge_dists = dist_to_stop/2
    
    HPX = [sv_x_fresnet,centerline_merge_dists,dist_to_stop]
    HPY = [sv_y_fresnet,0,0]
    if veh_state.l2_action == 'NORMAL':
        mean_acc = max_acc_lon/np.random.randint(low=2,high=5)
    else:
        mean_acc = max_acc_lon/(2+np.random.random_sample())
    time_to_target_speed = abs(target_vel-veh_state.speed)/abs(mean_acc)
    if time_to_target_speed < constants.PLAN_FREQ*2:
        tx_knots = [0,time_to_target_speed/5,time_to_target_speed,constants.PLAN_FREQ*2]
        vxh_knots = [veh_state.speed,veh_state.speed+(target_vel-veh_state.speed)/5,target_vel,target_vel]
    else:
        tx_knots = [0,time_to_target_speed/5,time_to_target_speed]
        vxh_knots = [veh_state.speed,veh_state.speed+(target_vel-veh_state.speed)/5,target_vel]
    #print(vxh_knots)
    try:
        #cs_v = UnivariateSpline(tx_knots, np.sqrt(vxh_knots), k=2)
        #cs_v.set_smoothing_factor(0.5)
        _d = OrderedDict(sorted(list(zip(tx_knots,vxh_knots)),key=lambda tup: tup[0]))
        tx_knots, vxh_knots = list(_d.keys()),list(_d.values())
        cs_v = CubicSpline(tx_knots, vxh_knots,bc_type='clamped')
    except ValueError:
        print(tx_knots,vxh_knots)
        raise
    #cs_v = CubicSpline(tx_knots, np.sqrt(vxh_knots), bc_type='clamped')
    tx = np.arange(0,tx_knots[-1]+dt,dt)
    #vel_profile = [(x,cs_v(x)**2) for x in tx]
    vel_profile = []
    stop_reached = False
    for t in tx:
        if not stop_reached:
            _vel_val = float(cs_v(t))
            if _vel_val <=0.01 :
                stop_reached = True
            vel_profile.append((t,max(0,_vel_val)))
        else:
            vel_profile.append((t,0))
    num_within_lane = 0
    
    #plt.plot([sv_x_fresnet],[sv_y_fresnet],'x')
    #plt.plot([x[0] for x in center_line],[x[1] for x in center_line])
    
    cs_p = CubicSpline(HPX, HPY,bc_type='clamped')
    ref_path = [(x,float(cs_p(x))) for x in np.arange(0,HPX[-1]+.5,.5)]
    #plt.plot([x[0] for x in ref_path], [x[1] for x in ref_path])
    MX,MY = utils.fresnet_to_map(fresnet_origin[0], fresnet_origin[1], [x[0] for x in ref_path], [x[1] for x in ref_path], center_line_angle)
    ref_path_map = list(zip(MX,MY))
    traj = utils.generate_trajectory_from_vel_profile(tx, ref_path_map, [v[1] for v in vel_profile])
    traj_outside_lane = False
    traj_dist_from_centerline = max([abs(utils.distance_numpy([center_line[0][0],center_line[0][1]],[center_line[1][0],center_line[1][1]],[t[0],t[1]])) for t in traj])
    if traj_dist_from_centerline > constants.LANE_WIDTH:
        traj_outside_lane = True
    if traj_outside_lane:
        return None
        #else:
        #    print('outside lane boundary',traj_dist_from_centerline)
    #print('number of trajectories found within lane',num_within_lane)
    
    #plt.show()
    num_acc,num_rej = 0,0
    
    tx = [veh_state.current_time+x for x in tx]
    rv = [x[1] for x in vel_profile]
    ra = [abs(x[1]-x[0])/dt for x in zip(rv[:-1],rv[1:])]
    ra = ra + [ra[-1]]
    rj = [abs(x[1]-x[0])/dt for x in zip(ra[:-1],ra[1:])]
    rj = rj +[rj[-1]]
    min_acc,max_acc,min_jerk,max_jerk = min(ra),max(ra),min(rj),max(rj)
        
    if (max_acc <= max_acc_lims and max_jerk <= max_acc_jerk_lims) and not (veh_state.l1_action[0:4]=='wait' and rv[-1] > 0.01):
        yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
        res = np.asarray([tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj)])
        if constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
            clipped_res = utils.clip_trajectory_to_viewport(res)
            if clipped_res is not None:
                return clipped_res
        else:
            return res
        #print('accepted',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
        
    else:
        ''' we are adding all trajectories since we would analyze the cost later'''
        #print('rejected',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
        
        yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
        res = np.asarray([tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj)])
        if constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
            clipped_res = utils.clip_trajectory_to_viewport(res)
            if clipped_res is not None:
                return clipped_res
        else:
            return res


def track_speed_planner_cp(veh_state, center_line, state_vals):
    max_acc_lon, max_acc_lat, target_vel = state_vals[0], state_vals[1], state_vals[2]
    dt = constants.LP_FREQ
    traj_list = []
    sx, sy, syaw, sv, sax, say = veh_state.x, veh_state.y, veh_state.yaw, veh_state.speed, veh_state.long_acc, veh_state.tan_acc
    if veh_state.l2_action == 'NORMAL':
        max_acc_lims,max_acc_jerk_lims = constants.MAX_LONG_ACC_NORMAL,constants.MAX_ACC_JERK_NORMAL
    else:
        max_acc_lims,max_acc_jerk_lims = constants.MAX_LONG_ACC_AGGR,constants.MAX_ACC_JERK_AGGR 
    center_line_angle = math.atan2((center_line[1][1]-center_line[0][1]),(center_line[1][0]-center_line[0][0]))
    dist_to_centerline = utils.distance_numpy([center_line[0][0],center_line[0][1]],[center_line[1][0],center_line[1][1]],[sx,sy])
    sv_angle_with_cl = syaw - center_line_angle
    sv_angle_in_map = math.atan2(sy,sx)
    fresnet_origin = (sx - dist_to_centerline*math.cos(sv_angle_in_map), sy - dist_to_centerline*math.sin(sv_angle_in_map))
    sv_x_fresnet,sv_y_fresnet = 0,dist_to_centerline
    hpx,hpy = [sv_x_fresnet,sv_y_fresnet]
    centerline_merge_dists = np.arange(1,50)
    final_knots_on_centerline = centerline_merge_dists + [constants.CAR_LENGTH]*len(centerline_merge_dists)
    HPX = [sv_x_fresnet]*len(centerline_merge_dists) + centerline_merge_dists + final_knots_on_centerline
    HPY = [sv_y_fresnet]*len(centerline_merge_dists) + [0]*(len(centerline_merge_dists)+1)
    HPX,HPY = [[sv_x_fresnet,50,100]],[[sv_y_fresnet,sv_y_fresnet,sv_y_fresnet]]
    num_samples = 100
    for y_len in np.arange(sv_y_fresnet-1,sv_y_fresnet+1.05,0.05):
        HPY.append([sv_y_fresnet,(y_len+sv_y_fresnet)/2,y_len])
        HPX.append([sv_x_fresnet,50,100])
    if veh_state.l2_action == 'NORMAL':
        mean_acc = max_acc_lon/4
    else:
        mean_acc = max_acc_lon/2
    if mean_acc != 0:
        time_to_target_speed = abs(target_vel-veh_state.speed)/abs(mean_acc)
        if time_to_target_speed < constants.PLAN_FREQ*2:
            tx_knots = [0,time_to_target_speed/5,time_to_target_speed,constants.PLAN_FREQ*2]
            vxh_knots = [veh_state.speed,veh_state.speed+(target_vel-veh_state.speed)/5+np.random.random_sample(),target_vel,target_vel]
        else:
            tx_knots = [0,time_to_target_speed/5,time_to_target_speed]
            vxh_knots = [veh_state.speed,veh_state.speed+(target_vel-veh_state.speed)/5+np.random.random_sample(),target_vel]
        #print(vxh_knots)
        try:
            cs_v = UnivariateSpline(tx_knots, np.sqrt(vxh_knots), k=2)
            cs_v.set_smoothing_factor(0.5)
        except ValueError:
            print(tx_knots,vxh_knots)
            raise
        #cs_v = CubicSpline(tx_knots, np.sqrt(vxh_knots), bc_type='clamped')
        tx = np.arange(0,tx_knots[-1]+dt,dt)
        vel_profile = [(x,cs_v(x)**2) for x in tx]
    else:
        tx = np.arange(0,(constants.PLAN_FREQ*2)+dt,dt)
        vel_profile = [(x,veh_state.speed) for x in tx]
    num_within_lane = 0
    all_path_samples,all_vel_samples,all_time_samples = [],[],[]
    #plt.plot([sv_x_fresnet],[sv_y_fresnet],'x')
    #plt.plot([x[0] for x in center_line],[x[1] for x in center_line])
    for path_knots in zip(HPX,HPY):
        cs_p = CubicSpline(path_knots[0], path_knots[1],bc_type='clamped')
        ref_path = [(x,float(cs_p(x))) for x in np.arange(0,path_knots[0][-1],.5)]
        #plt.plot([x[0] for x in ref_path], [x[1] for x in ref_path])
        MX,MY = utils.fresnet_to_map(fresnet_origin[0], fresnet_origin[1], [x[0] for x in ref_path], [x[1] for x in ref_path], center_line_angle)
        ref_path_map = list(zip(MX,MY))
        traj = utils.generate_trajectory_from_vel_profile(tx, ref_path_map, [v[1] for v in vel_profile])
        traj_outside_lane = False
        traj_dist_from_centerline = max([abs(utils.distance_numpy([center_line[0][0],center_line[0][1]],[center_line[1][0],center_line[1][1]],[t[0],t[1]])) for t in traj])
        if traj_dist_from_centerline > constants.LANE_WIDTH:
            traj_outside_lane = True
        if not traj_outside_lane:
            all_vel_samples.append(vel_profile)    
            all_path_samples.append(traj)
            all_time_samples.append(tx)
            num_within_lane +=1
        #else:
        #    print('outside lane boundary',traj_dist_from_centerline)
    #print('number of trajectories found within lane',num_within_lane)
    
    #plt.show()
    num_acc,num_rej = 0,0
    for T,P,V in zip(all_time_samples,all_path_samples,all_vel_samples):
        tx = [veh_state.current_time+x for x in T]
        rv = [x[1] for x in V]
        rx = [x[0] for x in P]
        traj = P
        ra = [abs(x[1]-x[0])/dt for x in zip(rv[:-1],rv[1:])]
        ra = ra + [ra[-1]]
        rj = [abs(x[1]-x[0])/dt for x in zip(ra[:-1],ra[1:])]
        rj = rj +[rj[-1]]
        min_acc,max_acc,min_jerk,max_jerk = min(ra),max(ra),min(rj),max(rj)
            
        if (max_acc <= max_acc_lims and max_jerk <= max_acc_jerk_lims) and not (veh_state.l1_action[0:4]=='wait' and rv[-1] > 0.01):
            yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
            res = np.asarray([tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj), len(T), 'CP'])
            if constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
                clipped_res = utils.clip_trajectory_to_viewport(res)
                if clipped_res is not None:
                    traj_list.append((clipped_res, 'CP'))
            else:
                traj_list.append((res,'CP'))
            #print('accepted',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
            num_acc += 1
        else:
            ''' we are adding all trajectories since we would analyze the cost later'''
            #print('rejected',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
            
            yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
            res = np.asarray([tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj), len(T), 'CP'])
            if constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
                clipped_res = utils.clip_trajectory_to_viewport(res)
                if clipped_res is not None:
                    traj_list.append((clipped_res, 'CP'))
            else:
                traj_list.append((res,'CP'))
            
            num_rej += 1
    #print(veh_state.id,veh_state.l1_action,veh_state.l2_action,target_vel,round(np.mean([x[-1] for x in all_time_samples]),2),'acc|rej',num_acc,num_rej)
    
    '''
    plt.figure()
    plt.plot([veh_state.x],[veh_state.y],'x')
    for t in traj_list:
        plt.plot(t[0][1],t[0][2])
    plt.axis('equal')
    plt.title('path')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][4])
    plt.title('vel')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][5])
    plt.title('acc')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][6])
    plt.title('jerk')
    plt.show()
    f=1
    '''
    return traj_list
    
def cubic_spline_planner(veh_state,knot_grids):
    max_acc = (constants.MAX_LONG_ACC_AGGR+constants.MAX_LAT_ACC_AGGR)/2 if veh_state.l2_action == 'AGGRESSIVE' else (constants.MAX_LONG_ACC_NORMAL+constants.MAX_LAT_ACC_NORMAL)/2
    path_knot_grids, vel_knot_grids = knot_grids[0], knot_grids[1]
    HPX = [np.full(shape=(path_knot_grids[-1].shape[0],path_knot_grids[-1].shape[1]),fill_value = veh_state.x)]
    for p_k in path_knot_grids:
        HPX.append(p_k[:,:,0])
    HPY = [np.full(shape=(path_knot_grids[-1].shape[0],path_knot_grids[-1].shape[1]),fill_value = veh_state.y)]
    for p_k in path_knot_grids:
        HPY.append(p_k[:,:,1])
    VP = [np.full(shape=(vel_knot_grids[-1].shape[0],vel_knot_grids[-1].shape[1]),fill_value = veh_state.speed)]
    for v_k in vel_knot_grids:
        VP.append(v_k)
    HPX = np.stack(HPX, axis = -1)
    HPY = np.stack(HPY, axis = -1)
    VP = np.stack(VP, axis = -1)
    if len(HPX.shape) != 3:
        sys.exit('malformed knot arrays')
    all_path_samples,all_time_ts,ders,all_vel_x_profiles,all_vel_y_profiles,all_vel_h_profiles,all_vel_profiles,all_path_splines,\
    all_vel_splines = [],[],[],[],[],[],[],[],[]
    dt = constants.LP_FREQ
    for i in np.arange(HPX.shape[0]):
        for j in np.arange(HPX.shape[1]):
            hpx_vals = HPX[i,j,:].tolist()
            hpy_vals = HPY[i,j,:].tolist()
            vpx_vals = VP[i,j,:].tolist()
            path_lengths = [math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(zip(hpx_vals[:-1], hpy_vals[:-1]), zip(hpx_vals[1:], hpy_vals[1:]))]
            total_time_to_cross = sum(path_lengths) / (max_acc + (-1 * np.random.sample()))
            #tx_vals = [0] + [((vpx_vals[i] - vpx_vals[0]) / max_acc) + 0.25 for i in np.arange(1,len(vpx_vals))]
            tx_vals = [0] + [sum(path_lengths[:i])/sum(path_lengths)*total_time_to_cross for i in np.arange(1,len(path_lengths)+1)]
            if veh_state.l1_action[0:4] == 'wait':
                step = -0.1 if hpx_vals[0] >= hpx_vals[-1] else 0.1
            else:
                step = -0.5 if hpx_vals[0] >= hpx_vals[-1] else 0.5
            x_d1 = 1 if hpx_vals[0] >= hpx_vals[-1] else -1
            rx = list(np.arange(float(hpx_vals[0]),float(hpx_vals[-1]+step),step))
            if len(rx) < 5:
                continue
            reversed = False
            if hpx_vals[-1] < hpx_vals[0]:
                hpx_vals.reverse()
                hpy_vals.reverse()
                vpx_vals.reverse()
                reversed = True
            _l = list(zip(hpx_vals,hpy_vals))
            _l.sort(key=lambda tup: tup[0])
            hpx_vals,hpy_vals = [x[0] for x in _l], [x[1] for x in _l]
            try:
                cs = CubicSpline(hpx_vals, hpy_vals)
            except ValueError:
                print(hpx_vals)
                print(hpy_vals)
                raise
            cs_d2 = cs.derivative(2)
            cs_d1 = cs.derivative(1)
            path_sample = [(float(x),float(cs(float(x)))) for x in rx]
            all_path_splines.append(cs)
            #ry = [x[1] for x in path_sample]
            path_yaws = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(path_sample[:-1],path_sample[1:])]
            
            k = [abs(x_d1*cs_d2(x)) / ((x_d1**2 + cs_d1(x)**2)**1.5) for x in rx]
            max_k = max(k)
            max_k_index = k.index(max(k))
            knot_indices_in_path = [utils.find_nearest_in_array(rx,x) for x in hpx_vals]
            path_angle_at_knots = [path_yaws[i] for i in knot_indices_in_path]
            vpx_x_vals = [x[0]*np.cos(x[1]) for x in zip(vpx_vals,path_angle_at_knots)]
            vpx_y_vals = [x[0]*np.sin(x[1]) for x in zip(vpx_vals,path_angle_at_knots)]
            step_lengths = [0]+[math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(path_sample[:-1],path_sample[1:])]
            ders.append(k)
            cs_v = CubicSpline(hpx_vals,vpx_vals)
            cs_v_x = CubicSpline(hpx_vals,vpx_x_vals)
            cs_v_y = CubicSpline(hpx_vals,vpx_y_vals)
            #vel_profile = [(x,float(cs_v(x))) for x in rx]
            vel_profile_x = [(x,float(cs_v_x(x))) for x in rx]
            vel_profile_y = [(x,float(cs_v_y(x))) for x in rx]
            #time_ts = np.arange(tx_vals[0],tx_vals[-1]+constants.LP_FREQ,constants.LP_FREQ)
            '''
            if True:
                constant_vel_arc_length,indices,arc_sum,_i = 1,[max_k_index],0,1
                while arc_sum < constant_vel_arc_length:
                    arc_sum += step_lengths[max_k_index]
                    arc_sum += step_lengths[max_k_index-_i]
                    arc_sum += step_lengths[max_k_index+_i]
                    indices.append(max_k_index-_i)
                    indices.append(max_k_index+_i)
                    _i += 1
                const_vel_val = min([vel_profile[x][1] for x in indices])
                new_knot_1 = [(vel_profile[max_k_index-_i][0],const_vel_val),(vel_profile[max_k_index+_i][0],const_vel_val)]
                new_vel_knots = list(zip(hpx_vals,vpx_vals)) + new_knot_1
                new_vel_knots.sort(key=lambda tup: tup[0])
                cs_v = CubicSpline([x[0] for x in new_vel_knots],[x[1] for x in new_vel_knots])
                #vel_profile = [(x,float(cs_v(x))) for x in rx]
            '''
            '''
            time_ts = np.arange(tx_vals[0],tx_vals[-1]+.1,.1)
            step_lengths = [0]+[math.hypot(x[1][0]-x[0][0],x[1][1]-x[0][1]) for x in zip(path_sample[:-1],path_sample[1:])]
            time_steps = [(2*step_lengths[i]) / (vel_profile[i][1]+vel_profile[i-1][1]) for i in np.arange(len(rx))]
            mean_ts = np.mean(time_steps)
            #vel_profile = [(float(x)+veh_state.current_time,float(cs_v(float(x)))) for x in time_ts]
            total_time_to_cross = sum(time_steps)
            acc_curve = cs_v.derivative(1)
            acc_at_max_k = acc_curve(hpx_vals[0]+(max_k_index*step))
            vel_at_max_k = vel_profile[max_k_index][1]
            time_to_max_k = sum(time_steps[:max_k_index])
            ders.append([vel_profile[i][1]-vel_profile[i-1][1] for i in np.arange(1,len(rx))])
            '''        
            
            vel_profile = [(x[0][0],abs(math.hypot(x[0][1],x[1][1])*np.cos(x[2]))) for x in zip(vel_profile_x,vel_profile_y,path_yaws)]
            
            time_steps = [(2*step_lengths[i]) / (vel_profile[i][1]+vel_profile[i-1][1]) for i in np.arange(len(rx))]
            total_time_to_cross = sum(time_steps)
            mean_vel = np.mean([math.hypot(x[0][1],x[1][1]) for x in zip(vel_profile_x,vel_profile_y)])
            path_length = sum(step_lengths)
            
            tx = np.arange(0,path_length/mean_vel+dt,dt)
            acc_knots = [veh_state.long_acc,3.6,1+np.random.sample(),np.random.sample()]
            acc_tx_knots = [0,abs(3.6-veh_state.long_acc),tx[-1]-0.5,tx[-1]]
            #acc_cs = splrep(x=acc_tx_knots,y=acc_knots,k=3,s=0.5)
            #v_cs = acc_cs.antiderivative(1)
            #acc_profile = splev(tx, acc_cs)
            #vel_profile = [(t,0) for t in tx]
            #acc_profile = [veh_state.long_acc] + [vel_profile[i][1]-vel_profile[i-1][1] for i in np.arange(1,len(vel_profile))]
            #all_vel_h_profiles.append((vel_profile,acc_profile))
            
            tx_knots = [0]
            vxx_knots = [vel_profile_x[0][1]]
            vxy_knots = [vel_profile_y[0][1]]
            vxh_knots = [veh_state.speed]
            if reversed:
                knot_indices_in_path.reverse()
            #  or k_i == len(knot_indices_in_path)-2
            knot_indices_in_path = [0,len(rx)//2,len(rx)-2,len(rx)-1]
            for k_i,k in enumerate(knot_indices_in_path):
                if k==0:
                    continue
                mean_vel = np.mean([math.hypot(x[0][1],x[1][1]) for x in zip(vel_profile_x[:k],vel_profile_y[:k])])
                path_length = sum(step_lengths[:k])
                tx_knot_val = path_length/mean_vel
                if math.isnan(tx_knot_val):
                    brk = 1
                tx_knots.append(path_length/mean_vel)
                #vxx_knots.append(vel_profile_x[k][1])
                #vxy_knots.append(vel_profile_y[k][1])
                vxh_knots.append(vel_profile[k][1])
            ''' add the first knot to start accelerating '''
            tx_knots = [tx_knots[0]] + [abs(tx_knots[1]-tx_knots[0])/2] + tx_knots[1:]
            vxh_knots = [vxh_knots[0]] + [vxh_knots[0] + (vxh_knots[1]-vxh_knots[0])/2] + vxh_knots[1:]
            #cs_v_x = CubicSpline(tx_knots,vxx_knots)
            #cs_v_y = CubicSpline(tx_knots,vxy_knots)
            #cs_p_x = CubicSpline(tx_knots,hpx_vals)
            #cs_p_y = CubicSpline(tx_knots,hpy_vals)
            #cs_v = CubicSpline(tx_knots,np.sqrt(vxh_knots))
            if veh_state.l1_action[0:4] == 'wait':
                first_der = -.1
                #vxh_knots = [x if x >= 0.01 else 0 for x in vxh_knots]
                
                try:
                    #cs_v = CubicSpline([x for i,x in enumerate(tx_knots) if i!=len(tx_knots)-2],[x for i,x in enumerate(vxh_knots) if i!=len(vxh_knots)-2],bc_type='clamped')
                    _l = list(zip(tx_knots,vxh_knots))
                    dup_idx = []
                    for i in np.arange(1,len(_l)):
                        if _l[i-1][0] == _l[i][0]:
                            dup_idx.append(i)
                    if len(dup_idx) > 0:
                        _l = [_l[i] for i in np.arange(len(_l)) if i not in dup_idx]
                    #cs_v = UnivariateSpline([x[0] for x in _l], [x[1] for x in _l], k=2)
                    if len(tx_knots)-2 in dup_idx:
                        cs_v = CubicSpline([x[0] for x in _l], [x[1] for x in _l],bc_type='clamped')
                    else:
                        cs_v = CubicSpline([x[0] for i,x in enumerate(_l) if i!=len(_l)-2],[x[1] for i,x in enumerate(_l) if i!=len(_l)-2],bc_type='clamped')
                except ValueError:
                    print([x[0] for x in _l])
                    print([x[1] for x in _l])
                    print(tx_knots)
                    print(vxh_knots)
                    raise
                vel_profile = []
                stop_reached = False
                for t in tx:
                    if not stop_reached:
                        _vel_val = float(cs_v(t))
                        if _vel_val <=0.01 :
                            stop_reached = True
                        vel_profile.append((t,max(0,_vel_val)))
                    else:
                        vel_profile.append((t,0))
                
            else:
                try:
                    _l = list(zip(tx_knots,vxh_knots))
                    _l.sort(key=lambda tup: tup[0])
                    tx_knots,vxh_knots = [x[0] for x in _l], [x[1] for x in _l]
                    cs_v = UnivariateSpline(tx_knots, np.sqrt(vxh_knots), k=2)
                except ValueError:
                    print(tx_knots)
                    print(vxh_knots)
                    raise
                vel_profile = [(t,float(cs_v(t))**2) for t in tx]
            #all_vel_splines.append(cs_v)
            #cs_yaw = CubicSpline(tx_knots,path_angle_at_knots)
            #csd_acc = cs_v.derivative(1)
            #csd_jerk = cs_v.derivative(2)
            #vel_profile_x = [(t,float(cs_v_x(t))) for t in tx]
            #vel_profile_y = [(t,float(cs_v_y(t))) for t in tx]
            #traj_profile_x = [(t,float(cs_p_x(t))) for t in tx]
            #traj_profile_y = [(t,float(cs_p_y(t))) for t in tx]
            #yaw_profile = [float(cs_yaw(t)) for t in tx]
            #vel_4_poly = splrep(x=tx_knots,y=vxh_knots,k=3,s=1)
            #vel_profile = list(zip(tx,splev(tx, vel_4_poly)))
            
            #acc_profile = [float(csd_acc(t)) for t in tx]
            #jerk_profile = [float(csd_jerk(t)) for t in tx]
            #vel_profile_m = [(x[0][0],math.hypot(x[0][1],x[1][1])) for x in zip(vel_profile_x,vel_profile_y,yaw_profile)]
            #vel_profile_yaws = [(x[0][0],math.hypot(x[0][1],x[1][1])) for x in zip(vel_profile_x,vel_profile_y)]
            #path_sample = [(x[0][1],x[1][1]) for x in zip(traj_profile_x,traj_profile_y)]
            #all_vel_profiles.append(vel_profile)
            acc_profile = [veh_state.long_acc] + [abs(vel_profile[i][1]-vel_profile[i-1][1]) for i in np.arange(1,len(vel_profile))]
            jerk_profile = [abs(acc_profile[i]-acc_profile[i-1]) for i in np.arange(1,len(acc_profile))]
            jerk_profile = jerk_profile + [jerk_profile[-1]]
            all_vel_profiles.append(vel_profile)
            all_time_ts.append(tx)
            all_vel_x_profiles.append(vel_profile_x)
            all_vel_y_profiles.append(vel_profile_y)
            #all_vel_m_profiles.append(vel_profile_m)
            #print(total_time_to_cross,tx[-1])
            #print('time to x',total_time_to_cross)
            #plt.plot(tx,[x[1] for x in vel_profile])
            
            all_path_samples.append(path_sample)
    '''
    plt.figure()
    plt.axis('equal')
    for p in all_path_samples:
        plt.plot([x[0] for x in p],[x[1] for x in p])
    plt.title('paths')
    
    '''
    '''
    plt.figure()
    for p in ders:
        plt.plot(np.arange(len(p)),p)
    plt.title('curvature')
    
    plt.figure()
    for d in all_vel_x_profiles:
        plt.plot([x[0] for x in d],[x[1] for x in d])
    plt.title('vel x')
    
    plt.figure()
    for d in all_vel_y_profiles:
        plt.plot([x[0] for x in d],[x[1] for x in d])
    plt.title('vel y')
    
    plt.figure()
    for d in all_vel_y_profiles:
        plt.plot([x[0] for x in d],[x[1] for x in d])
    plt.title('vel m')
    '''
    '''
    plt.figure()
    for v in all_vel_profiles:
        plt.plot([x[0] for x in v],[x[1] for x in v])
    plt.title('vel')
    '''
    '''
    plt.figure()
    for d in all_vel_h_profiles:
        v,a = d[0],d[1]
        plt.plot([x[0] for x in v],a)
    plt.title('acc')
    
    plt.figure()
    for d in all_vel_h_profiles:
        v,a,j = d[0],d[1],d[2]
        plt.plot([x[0] for x in v],j)
    plt.title('jerk')
    
    plt.show()
    '''
    traj_list = []
    if veh_state.l2_action == 'AGGRESSIVE':
        max_acc_lims,max_dec_lims,max_acc_jerk_lims,max_dec_jerk_lims = constants.MAX_TURN_ACC_AGGR, -constants.MAX_TURN_ACC_AGGR, constants.MAX_TURN_JERK, -constants.MAX_TURN_JERK
    else:
        max_acc_lims,max_dec_lims,max_acc_jerk_lims,max_dec_jerk_lims = constants.MAX_TURN_ACC_NORMAL, -constants.MAX_TURN_ACC_NORMAL, constants.MAX_TURN_JERK, -constants.MAX_TURN_JERK
    num_acc,num_rej = 0,0
    for T,P,V,CS in zip(all_time_ts,all_path_samples,all_vel_profiles,all_path_splines):
        tx = [veh_state.current_time+x for x in T]
        rv = [x[1] for x in V]
        rx = [x[0] for x in P]
        rx_time = [x[0] for x in utils.split_in_n((rx[0],0), (rx[-1],0), len(tx))]
        traj = [(x,float(CS(x))) for x in rx_time]
        #traj = all_utils.generate_trajectory_from_vel_profile(tx, P, rv)
        ra = [abs(x[1]-x[0])/.1 for x in zip(rv[:-1],rv[1:])]
        ra = ra + [ra[-1]]
        rj = [abs(x[1]-x[0])/.1 for x in zip(ra[:-1],ra[1:])]
        rj = rj +[rj[-1]]
        min_acc,max_acc,min_jerk,max_jerk = min(ra),max(ra),min(rj),max(rj)
            
        if (max_acc <= max_acc_lims and max_jerk <= max_acc_jerk_lims) and not (veh_state.l1_action[0:4]=='wait' and rv[-1] > 0.01):
            yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
            res = tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj), len(T)
            traj_list.append((res,'CP'))
            #print('accepted',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
            num_acc += 1
        else:
            yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
            res = tx, np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj), len(T)
            traj_list.append((res,'CP'))
            #print('rejected',veh_state.l2_action,'trajectory.max,min acc,jerk',(max_acc),(max_jerk))
            num_rej += 1
    print(veh_state.l1_action,veh_state.l2_action,'acc|rej',num_acc,num_rej)
    '''
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][1],t[0][2])
    plt.title('path')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][4])
    plt.title('vel')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][5])
    plt.title('acc')
    plt.figure()
    for t in traj_list:
        plt.plot(t[0][0],t[0][6])
    plt.title('jerk')
    plt.show()
    '''
    return traj_list
    

def generate_follow_lead_in_intersection_trajectory(veh_state):
    traj_list = []
    
    sc_trajectory = utils.get_track(veh_state, veh_state.current_time, True)
    lv_trajectory = utils.get_track(veh_state.leading_vehicle, veh_state.current_time, True)
    selected_indices = np.arange(0,min(len(sc_trajectory),len(lv_trajectory)),constants.DATASET_FPS*constants.LP_FREQ,dtype=int)
    sc_trajectory = sc_trajectory[selected_indices,:]
    lv_trajectory = lv_trajectory[selected_indices,:]
    rx, ry = [float(x) for x in sc_trajectory[:,1]],[float(y) for y in sc_trajectory[:,2]]
    rx_slice,ry_slice = rx[:int(constants.PLAN_FREQ/constants.LP_FREQ)],ry[:int(constants.PLAN_FREQ/constants.LP_FREQ)]
    lvv = np.true_divide(lv_trajectory[:,3].astype(float),3.6)
    lv_slice = lvv[:int(constants.PLAN_FREQ/constants.LP_FREQ)]
    lv_path = list(zip(lv_trajectory[:,1],lv_trajectory[:,2]))
    lv_path_slice = [(float(x[0]),float(x[1])) for x in lv_path[:int(constants.PLAN_FREQ/constants.LP_FREQ)]]
    hpx = [rx[0],rx[len(rx)//3],rx[2*len(rx)//3],rx[-1]]
    hpy = [ry[0],ry[len(ry)//3],ry[2*len(ry)//3],ry[-1]]
    '''
    if hpx[-1] < hpx[0]:
        hpx.reverse()
        hpy.reverse()
    cs = CubicSpline(hpx, hpy)
    path = [(float(x),cs(float(x))) for x in rx]
    '''
    lv_x_pos, lv_y_pos = veh_state.leading_vehicle.x, veh_state.leading_vehicle.y
    ''' construct a line lattice wrt lead vehicle's position '''
    path_y_samples_fresnet = np.arange(-2,2.25,.25)
    path_knot_1_samples = utils.fresnet_to_map(lv_x_pos, lv_y_pos, [0]*len(path_y_samples_fresnet), path_y_samples_fresnet, veh_state.leading_vehicle.yaw)
    all_path_samples = []
    for pt_idx in np.arange(len(path_knot_1_samples[0])):
        hpx = [rx[0],path_knot_1_samples[0][pt_idx],rx[-1]]
        hpy = [ry[0],path_knot_1_samples[1][pt_idx],ry[-1]]
        try:
            _d = OrderedDict(sorted(list(zip(hpx,hpy)),key=lambda tup: tup[0]))
            hpx,hpy = list(_d.keys()),list(_d.values())
            cs = CubicSpline(hpx, hpy)
        except ValueError:
            print(hpx)
            print(hpy)
            raise
        all_path_samples.append([(float(x),float(cs(float(x)))) for x in rx])
    time = sc_trajectory[:int(constants.PLAN_FREQ/constants.LP_FREQ),6]
    vpx = [time[0],time[len(time)//3],time[2*len(time)//3],time[-1]]
    knot_1_samples = np.random.normal(loc=lv_slice[len(lv_slice)//3], scale=0.25, size=50)
    knot_2_samples = np.random.normal(loc=lv_slice[2*len(lv_slice)//3], scale=0.25, size=50)
    knot_3_samples = np.random.normal(loc=lv_slice[-1], scale=0.25, size=50)
    knot_states = zip(knot_1_samples,knot_2_samples,knot_3_samples)
    traj_list = []
    okct = 0
    if veh_state.l2_action == 'NORMAL':
        max_acc_lims,max_dec_lims,max_acc_jerk_lims,max_dec_jerk_lims = constants.MAX_LONG_ACC_AGGR, constants.MAX_LONG_DEC_AGGR, constants.MAX_ACC_JERK_AGGR, constants.MAX_DEC_JERK_AGGR
    else:
        max_acc_lims,max_dec_lims,max_acc_jerk_lims,max_dec_jerk_lims = constants.MAX_LONG_ACC_NORMAL, constants.MAX_LONG_DEC_NORMAL, constants.MAX_ACC_JERK_NORMAL, constants.MAX_DEC_JERK_NORMAL
    for ref_p in all_path_samples:
        for ks in knot_states:
            if ks[0] <=0 or ks[1] <=0 or ks[2] <=0 :
                continue
            vpy = [lv_slice[0],ks[0],ks[1],ks[2]]
            v_cs = CubicSpline(vpx, vpy)
            rv = [max(0,v_cs(t)) for t in time]
            traj = utils.generate_trajectory_from_vel_profile(time, ref_p, [max(0,v_cs(t)) for t in time])
            dist_gaps = []
            ra = [(x[1]-x[0])/constants.LP_FREQ for x in zip(rv[:-1],rv[1:])]
            rj = [(x[1]-x[0])/constants.LP_FREQ for x in zip(ra[:-1],ra[1:])]
            for i,t in enumerate(time):
                dist_gaps.append(math.hypot(lv_path_slice[i][0]-traj[i][0], lv_path_slice[i][1]-traj[i][1]))
                
            #print('min dist gap', min(dist_gaps))
            min_acc,max_acc,min_jerk,max_jerk = min(ra),max(ra),min(rj),max(rj)
            
            #if max_acc <= max_acc_lims and min_acc > max_dec_lims and max_jerk <= max_acc_jerk_lims and min_jerk > max_dec_jerk_lims:
            yaw = [veh_state.yaw] + [math.atan2(e[1][1]-e[0][1], e[1][0]-e[0][0]) for e in zip(traj[:-1],traj[1:])]
            res = np.asarray(time), np.asarray([x[0] for x in traj]), np.asarray([x[1] for x in traj]), np.asarray(yaw), np.asarray(rv), np.asarray(ra), np.asarray(rj), len(time)
            traj_list.append((res,'CP'))
            #print('OK')
            okct += 1
            
        #print()
        
        
        '''
        plt.plot(time,[max(0,v_cs(t)) for t in time])
    plt.plot(time,lvv[:int(constants.PLAN_FREQ/constants.LP_FREQ)])
    plt.show()
    '''
    return traj_list
            