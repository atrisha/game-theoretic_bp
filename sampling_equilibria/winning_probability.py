'''
Created on Aug 20, 2020

@author: Atrisha
'''

from all_utils import utils
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast
from agent_classes import AgentTote
import constants
import os
import math
from operator import itemgetter
from scipy.interpolate import CubicSpline, griddata, interp1d, splrep, splev
from collections import OrderedDict
from planning_objects import SceneState
from motion_planners import motion_planner
from visualizer import visualizer


tau = 1

def generate_path(hpx,hpy):
    path_prefix = utils.split_in_n((hpx[0],hpy[0]), (hpx[1],hpy[1]), 2)
    #all_paths = []
    #for i in np.arange(1):
    path = [(hpx[1],hpy[1])]
    hpx_s,hpy_s = hpx[1:],hpy[1:]
    '''
    if i == 0:
        hpx_s,hpy_s = hpx[1:],hpy[1:]
    else:
        if len(hpx) < 2:
            hpx_s,hpy_s = [np.random.normal(loc=x, scale=1) for x in hpx[1:]],[np.random.normal(loc=x, scale=1) for x in hpy[1:]]
        else:
            lats = utils.add_parallel(list(zip(hpx[2:],hpy[2:])), np.random.normal(0,1))
            lat = lats[np.random.choice([0,1])]
            hpx_s,hpy_s = [hpx[1]]+[x[0] for x in lat],[hpy[1]]+[x[1] for x in lat]
    '''
    max_coeff = 0
    s_x = [0] + [p2-p1 for p1,p2 in list(zip(hpx_s[:-1],hpx_s[1:]))]
    s_x = [0] + [sum(s_x[:i+1]) for i in np.arange(1,len(s_x))]
    s_y = [0] + [p2-p1 for p1,p2 in list(zip(hpy_s[:-1],hpy_s[1:]))]
    s_y = [0] + [sum(s_y[:i+1]) for i in np.arange(1,len(s_y))]
    indx = np.arange(len(s_x))
    if len(indx) > 1:
        cs_x = CubicSpline(indx,s_x)
        cs_y = CubicSpline(indx,s_y)
        ref_x = np.arange(indx[1],indx[-1]+.1,.1)
        for i_a in ref_x:
            path.append(((path[0][0]+cs_x(i_a)), ((path[0][1]+cs_y(i_a)))))
        path = path_prefix + path
    else:
        path = path_prefix
    path = list(OrderedDict.fromkeys(path))
    #all_paths.append(path)
    return path


def find_closest(clx,cly,pt):
    dists = [np.inf]*len(clx)
    if math.hypot(clx[0]-pt[0], cly[0]-pt[1]) < math.hypot(clx[-1]-pt[0], cly[-1]-pt[1]): 
        for idx in np.arange(len(clx)):
            dist = math.hypot(clx[idx]-pt[0],cly[idx]-pt[1])
            dists[idx] = dist  
            if dist < constants.CAR_LENGTH/2:
                return idx
    else:
        for idx in np.arange(len(clx)-1,-1,-1):
            dist = math.hypot(clx[idx]-pt[0],cly[idx]-pt[1])
            dists[idx] = dist  
            if dist < constants.CAR_LENGTH/2:
                return idx
    return min(enumerate(dists), key=itemgetter(1))[0] 

def find_point_along_cl(clx,cly,pt,dist):
    path = [pt] + list(zip(clx,cly))
    dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
    dist_from_origin = [sum(dist_from_origin[:i]) for i in np.arange(1,len(dist_from_origin))]
    path_idx = utils.find_index_in_list(dist, dist_from_origin)
    if path_idx is None:
        print(dist)
        print(dist_from_origin)
        raise IndexError("path_idx is None")
    if path_idx >= len(path)-1:
        path_idx = path_idx - 1
    overflow = dist - dist_from_origin[path_idx]
    r = overflow/math.hypot(path[path_idx+1][0]-path[path_idx][0], path[path_idx+1][1]-path[path_idx][1]) if overflow != 0 else 0
    point_x = path[path_idx][0] + r*(path[path_idx+1][0] - path[path_idx][0])
    point_y = path[path_idx][1] + r*(path[path_idx+1][1] - path[path_idx][1])
    point = (point_x,point_y) 
    return point

def final_path_pt(pos,clx,cly,dist):
    outofbounds = False
    if pos[0] < min(clx) or pos[0] > max(clx) or pos[1] < min(cly) or pos[1] > max(cly):
        outofbounds = True
    if not outofbounds:
        closest_idx = find_closest(clx, cly, pos)
        if closest_idx == len(clx)-1:
            outofbounds = True
    if not outofbounds:
        f_pt = find_point_along_cl(clx[closest_idx:],cly[closest_idx:],pos,dist)
    else:
        if math.hypot(clx[0]-pos[0], cly[0]-pos[1]) < math.hypot(clx[-1]-pos[0], cly[-1]-pos[1]): 
            v = (clx[0]-pos[0], cly[0]-pos[1])
        else:
            v = (pos[0]-clx[-1], pos[1]-cly[-1])
        u = (v[0]/math.hypot(v[0], v[1]), v[1]/math.hypot(v[0], v[1]))
        f_pt = (pos[0] + dist*u[0], pos[1] + dist*u[1])
    f_pt = (round(f_pt[0]*2)/2, round(f_pt[1]*2)/2)
    path = generate_path(clx,cly)
    
    if constants.show_plots:
        #plt.plot([f_pt[0]],[f_pt[1]],'x',color='red')    
        #plt.plot([pos[0]],[pos[1]],'s',color='green')
        #plt.plot(clx,cly,'-.',fillstyle='none',color='black')
        plt.plot([x[0] for x in path],[x[1] for x in path],'-.',color='black')
        f=1
    return f_pt
        

def get_baseline_cl(as_lattice, veh_state, centerline_map, seg_cls):
    veh_state.nose_x = veh_state.x + (constants.CAR_LENGTH/2)*np.cos(veh_state.yaw)
    veh_state.nose_y = veh_state.y + (constants.CAR_LENGTH/2)*np.sin(veh_state.yaw)
    start_seg = veh_state.current_segment
    baseline_cl = dict()
    for k,v in centerline_map.items():
        if start_seg not in k:
            continue
        segment_seq = list(k)
        cl_X, cl_Y = [], []
        cl_map = dict()
        for s_idx,seg in enumerate(segment_seq):
            if segment_seq.index(start_seg) > s_idx:
                continue
            X_pts, Y_pts = seg_cls[seg][0], seg_cls[seg][1]
            del_idx = []
            if 'turn' in seg:
                if segment_seq[-1][-1] == '1':
                    lat1,lat2 = utils.add_parallel(list(zip(X_pts,Y_pts)), 2, 0)
                else:
                    if 'rt' in seg:
                        lat1,lat2 = utils.add_parallel(list(zip(X_pts,Y_pts)), -1, 0)
                    else:
                        lat1,lat2 = utils.add_parallel(list(zip(X_pts,Y_pts)), -2, 0)
                X_pts = [x[0] for x in lat1]
                Y_pts = [x[1] for x in lat1]
            actions = utils.get_actions(veh_state)
            cl_map[seg] = [X_pts,Y_pts,{x:None for x in actions}]
            '''
            for action in actions:
                trajectory_plan = motion_planner.TrajectoryPlan(action,'NORMAL',veh_state.task,'BASELINE',veh_state,veh_state.id,0)
                trajectory_plan.set_lead_vehicle(veh_state.leading_vehicle)
                trajectory_plan.generate()
                f=1
            ''' 
            if segment_seq.index(start_seg) == s_idx:
                for i in np.arange(len(X_pts)-1):
                    angle_bet_veh_and_pts = utils.angle_between_lines_2pi([(veh_state.nose_x,veh_state.nose_y),(veh_state.x,veh_state.y)], [(veh_state.nose_x,veh_state.nose_y),(X_pts[i],Y_pts[i])])
                    dist_from_veh = math.hypot(veh_state.nose_x-X_pts[i], veh_state.nose_y-Y_pts[i])
                    
                    if dist_from_veh < 3 or (angle_bet_veh_and_pts < math.pi/2 or 2*math.pi-angle_bet_veh_and_pts < math.pi/2):
                        del_idx.append(i)
                    
                X_pts = [veh_state.x,veh_state.nose_x] + [x for (i,x) in enumerate(X_pts) if i not in del_idx]
                Y_pts = [veh_state.y,veh_state.nose_y] + [x for (i,x) in enumerate(Y_pts) if i not in del_idx]
            cl_X += X_pts
            cl_Y += Y_pts
        dup_indcs = []
        fcl_X,fcl_Y,cl_V = [],[],dict()
        '''
        plt.plot([veh_state.x,veh_state.nose_x],[veh_state.y,veh_state.nose_y],'-')
        for s_idx,seg in enumerate(list(k)):
            if segment_seq.index(start_seg) > s_idx:
                continue
            else:
                print(list(cl_map.keys()))
                plt.plot(cl_map[seg][0],cl_map[seg][1],'x')
        plt.show()
        '''
        for s_idx,seg in enumerate(list(k)):
            if segment_seq.index(start_seg) > s_idx:
                continue
            elif segment_seq.index(start_seg) == s_idx:
                scl = cl_map[seg]
                fcl_X = [veh_state.x,veh_state.nose_x]
                fcl_Y = [veh_state.y,veh_state.nose_y]
                for cl_pt in zip(scl[0],scl[1]):
                    angle_bet_lines = utils.angle_between_lines_2pi([(veh_state.x,veh_state.y),(veh_state.nose_x,veh_state.nose_y)], [(veh_state.nose_x,veh_state.nose_y),(cl_pt[0],cl_pt[1])])
                    dist_to_pt = math.hypot(veh_state.nose_x-cl_pt[0], veh_state.nose_y-cl_pt[1])
                    if dist_to_pt < 3 or not (angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2):
                        continue
                    else:
                        fcl_X.append(cl_pt[0])
                        fcl_Y.append(cl_pt[1])
                for a_k, a_v in scl[2].items():
                    if a_k not in cl_V:
                        cl_V[a_k] = []
                    cl_V[a_k] += [a_v]*len(scl[0])
            else:
                scl = cl_map[seg]
                for cl_pt in zip(scl[0],scl[1]):
                    angle_bet_lines = utils.angle_between_lines_2pi([(fcl_X[-2],fcl_Y[-2]),(fcl_X[-1],fcl_Y[-1])], [(fcl_X[-1],fcl_Y[-1]),(cl_pt[0],cl_pt[1])])
                    dist_to_pt = math.hypot(fcl_X[-1]-cl_pt[0], fcl_Y[-1]-cl_pt[1])
                    if dist_to_pt < 3 or not (angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2):
                        continue
                    else:
                        fcl_X.append(cl_pt[0])
                        fcl_Y.append(cl_pt[1])
                for a_k, a_v in scl[2].items():
                    if a_k not in cl_V:
                        cl_V[a_k] = []
                    cl_V[a_k] += [a_v]*len(scl[0])
        '''
        for i in np.arange(len(cl_X)-1):
            dist = math.hypot(cl_X[i]-cl_X[i+1], cl_Y[i]-cl_Y[i+1])
            if i < len(cl_X)-2:
                if i in dup_indcs:
                    is_backward = False
                    continue
                angle_bet_lines = utils.angle_between_lines_2pi([(cl_X[i],cl_Y[i]),(cl_X[i+1],cl_Y[i+1])], [(cl_X[i+1],cl_Y[i+1]),(cl_X[i+2],cl_Y[i+2])])
                if angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2:
                    is_backward = False
                else:
                    is_backward = True
            else:
                is_backward = False
            
            if (dist < 2.5 or is_backward) and (i+1) > 1:
                dup_indcs.append(i+1)
        cl_X = [x for (i,x) in enumerate(cl_X) if i not in dup_indcs]
        cl_Y = [x for (i,x) in enumerate(cl_Y) if i not in dup_indcs]
        '''
        cl_X,cl_Y = fcl_X,fcl_Y
        gridpts_X = []
        gridpts_Y = []
        if as_lattice:
            for i in np.arange(len(cl_X)-1):
                dist = math.hypot(cl_X[i]-cl_X[i+1], cl_Y[i]-cl_Y[i+1])
                gridpts_X.append(cl_X[i])
                gridpts_Y.append(cl_Y[i])
                if dist > 1:
                    mid_pts = utils.split_in_n((cl_X[i],cl_Y[i]), (cl_X[i+1],cl_Y[i+1]), math.ceil(dist))
                    for mp in mid_pts[1:-1]:
                        gridpts_X.append(mp[0])
                        gridpts_Y.append(mp[1])
            gridpts_X.append(cl_X[-1])
            gridpts_Y.append(cl_Y[-1])
            baseline_cl[k] = [([round(x*2)/2 for x in gridpts_X],[round(x*2)/2 for x in gridpts_Y])]
        else:
            gridpts_X = cl_X
            gridpts_Y = cl_Y
            baseline_cl[k] = [(gridpts_X,gridpts_Y)]
    return baseline_cl

    

def construct_centerline(as_lattice):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = "select distinct TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ from TRAJECTORY_MOVEMENTS"
    c.execute(q_string)
    res = c.fetchall()
    centerline_map = {tuple(ast.literal_eval(row[0])):None for row in res if row[0] is not None and 'l_e_n' not in row[0] and 'l_n_w' not in row[0]}
    for seg in list(centerline_map.keys()):
        seg_task = 'L_'+seg[0][3].upper()+'_'+seg[-1][-4].upper()
        if constants.TASK_MAP[seg_task] == 'LEFT_TURN' or constants.TASK_MAP[seg_task] == 'RIGHT_TURN':
            end_seg_tag = list(seg)[-1][-1]
            alt_exit_segment = list(seg)[:-1] + [list(seg)[-1][:-1] + '1'] if end_seg_tag == '2' else list(seg)[:-1] + [list(seg)[-1][:-1] + '2']  
            if tuple(alt_exit_segment) not in centerline_map:
                centerline_map[tuple(alt_exit_segment)] = None
    q_string = "select * from TRAFFIC_REGIONS_DEF WHERE TRAFFIC_REGIONS_DEF.REGION_PROPERTY='center_line'"
    c.execute(q_string)
    res = c.fetchall()
    all_centerlines = {row[0]:(ast.literal_eval(row[4]),ast.literal_eval(row[5])) for row in res}
    
    for k,v in centerline_map.items():
        segment_seq = list(k)
        cl_X, cl_Y = [], []
        for seg in segment_seq:
            cl_X += all_centerlines[seg][0]
            cl_Y += all_centerlines[seg][1]
        dup_indcs = []
        for i in np.arange(len(cl_X)-1):
            dist = math.hypot(cl_X[i]-cl_X[i+1], cl_Y[i]-cl_Y[i+1])
            if i < len(cl_X)-2:
                angle_bet_lines = utils.angle_between_lines_2pi([(cl_X[i],cl_Y[i]),(cl_X[i+1],cl_Y[i+1])], [(cl_X[i+1],cl_Y[i+1]),(cl_X[i+2],cl_Y[i+2])])
                if angle_bet_lines < math.pi/2 or 2*math.pi-angle_bet_lines < math.pi/2:
                    is_backward = False
                else:
                    is_backward = True
            else:
                is_backward = False
            if dist < 1 or is_backward:
                dup_indcs.append(i+1)
        cl_X = [x for (i,x) in enumerate(cl_X) if i not in dup_indcs]
        cl_Y = [x for (i,x) in enumerate(cl_Y) if i not in dup_indcs]
        '''
        gridpts_X = []
        gridpts_Y = []
        for i in np.arange(len(cl_X)-1):
            dist = math.hypot(cl_X[i]-cl_X[i+1], cl_Y[i]-cl_Y[i+1])
            gridpts_X.append(cl_X[i])
            gridpts_Y.append(cl_Y[i])
            if dist > 1:
                mid_pts = utils.split_in_n((cl_X[i],cl_Y[i]), (cl_X[i+1],cl_Y[i+1]), math.ceil(dist))
                for mp in mid_pts[1:-1]:
                    gridpts_X.append(mp[0])
                    gridpts_Y.append(mp[1])
        gridpts_X.append(cl_X[-1])
        gridpts_Y.append(cl_Y[-1])
        centerline_map[k] = ([round(x*2)/2 for x in gridpts_X],[round(x*2)/2 for x in gridpts_Y])
        '''
        gridpts_X = []
        gridpts_Y = []
        if as_lattice:
            for i in np.arange(len(cl_X)-1):
                dist = math.hypot(cl_X[i]-cl_X[i+1], cl_Y[i]-cl_Y[i+1])
                gridpts_X.append(cl_X[i])
                gridpts_Y.append(cl_Y[i])
                if dist > 1:
                    mid_pts = utils.split_in_n((cl_X[i],cl_Y[i]), (cl_X[i+1],cl_Y[i+1]), math.ceil(dist))
                    for mp in mid_pts[1:-1]:
                        gridpts_X.append(mp[0])
                        gridpts_Y.append(mp[1])
            gridpts_X.append(cl_X[-1])
            gridpts_Y.append(cl_Y[-1])
            centerline_map[k] = ([round(x*2)/2 for x in gridpts_X],[round(x*2)/2 for x in gridpts_Y])
        else:
            gridpts_X = cl_X
            gridpts_Y = cl_Y
            centerline_map[k] = [(gridpts_X,gridpts_Y)]
            
    return centerline_map
                

def populate_all_agent_info(agent_tote,pedestrian_info,scene_state):
    ra_ids = list(agent_tote.relev_agents.keys())
    time_ts = agent_tote.time_ts
    file_id = constants.CURRENT_FILE_ID
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
    c = conn.cursor()
    for ra_id in ra_ids:
        q_string = "select * FROM L1_ACTIONS where L1_ACTIONS.TRACK_ID="+str(ra_id)+" and time = "+str(time_ts)+" order by time"
        c.execute(q_string)
        res = c.fetchone()
        emp_action = res[0]
        ra_file_key = os.path.join(constants.L3_ACTION_CACHE, str(agent_tote.agent_id)+'-'+str(ra_id)+'_'+str(time_ts).replace('.', ','))
        if os.path.exists(ra_file_key):
            ag_info = utils.pickle_load(ra_file_key)
        else:
            ag_info = utils.setup_vehicle_state(ra_id,time_ts)
        if not hasattr(ag_info, 'relev_crosswalks'):
            relev_crosswalks = utils.get_relevant_crosswalks(ag_info)
            ag_info.set_relev_crosswalks(relev_crosswalks)
        if not hasattr(ag_info, 'scene_state'):
            ag_info.set_scene_state(scene_state)
        if not hasattr(ag_info, 'relev_pedestrians'):
            relev_pedestrians = utils.get_relevant_pedestrians(ag_info, pedestrian_info)
            ag_info.set_relev_pedestrians(relev_pedestrians)
        
            
        agent_tote.relev_agents[ra_id] = ag_info
    ag_file_key = os.path.join(constants.L3_ACTION_CACHE, str(agent_tote.agent_id)+'-0_'+str(time_ts).replace('.', ','))
    if os.path.exists(ag_file_key):
        ag_info = utils.pickle_load(ag_file_key)
    else:
        ag_info = utils.setup_vehicle_state(agent_tote.agent_id,time_ts)
    if not hasattr(ag_info, 'relev_crosswalks'):
        relev_crosswalks = utils.get_relevant_crosswalks(ag_info)
        ag_info.set_relev_crosswalks(relev_crosswalks)
    if not hasattr(ag_info, 'scene_state'):
            ag_info.set_scene_state(scene_state)
        
    if not hasattr(ag_info, 'relev_pedestrians'):
        relev_pedestrians = utils.get_relevant_pedestrians(ag_info, pedestrian_info)
        ag_info.set_relev_pedestrians(relev_pedestrians)
    agent_tote.ag_state = ag_info
 
def generate_s1_pt(ag_tote,centerlines,seg_cls):
    relev_agents = ag_tote.relev_agents
    ag_tote.r_ag_pts = dict()
    for ra_id,ra_state in relev_agents.items():
        print('ra_id',ra_id,ra_state.time,round(ra_state.speed/3.6,2))
        segment_seq = tuple(ra_state.segment_seq)
        baseline_cl = get_baseline_cl(False, ra_state, centerlines, seg_cls)
        ra_cls = [v for k,v in baseline_cl.items() if ra_state.current_segment in k]
        for ra_cl in ra_cls:
            dist_moved = max((ra_state.speed*tau) + (0.5*ra_state.tan_acc*(tau**2)), 0)    
            f_pt = final_path_pt((ra_state.x,ra_state.y), ra_cl[0][0], ra_cl[0][1], dist_moved)
            actions = utils.get_actions(ra_state)
            for action in actions:
                trajectory_plan = motion_planner.TrajectoryPlan(action,'NORMAL',ra_state.task,'BASELINE',ra_state,ag_tote.agent_id,ra_state.id)
                trajectory_plan.set_lead_vehicle(ra_state.leading_vehicle)
                trajectory_plan.generate()
                visualizer.plot_traffic_regions()
                plt.plot(trajectory_plan.trajectories['baseline'][0][1],trajectory_plan.trajectories['baseline'][0][2],'-.', color='blue')
                plt.show()
                f=1
            if ra_id not in ag_tote.r_ag_pts:
                ag_tote.r_ag_pts[ra_id] = []
            ag_tote.r_ag_pts[ra_id].append(f_pt)
       
    ag_state = ag_tote.ag_state
    baseline_cl = get_baseline_cl(False, ag_state, centerlines, seg_cls)
    ag_cls = [v for k,v in baseline_cl.items() if ag_state.current_segment in k]
    ag_tote.ag_pts = []
    actions = utils.get_actions(ag_tote.ag_state)
    for action in actions:
        trajectory_plan = motion_planner.TrajectoryPlan(action,'NORMAL',ag_tote.ag_state.task,'BASELINE',ag_tote.ag_state,ag_tote.agent_id,0)
        trajectory_plan.set_lead_vehicle(ag_tote.ag_state.leading_vehicle)
        trajectory_plan.generate()
            
    for ag_cl in ag_cls:
        dist_moved = max((ag_state.speed*tau) + (0.5*ag_state.tan_acc*(tau**2)), 0)    
        f_pt = final_path_pt((ag_state.x,ag_state.y), ag_cl[0][0], ag_cl[0][1], dist_moved)
        ag_tote.ag_pts.append(f_pt)
    
    
def get_trajectory(agent_id, emp_trajs):
    emp_trajs[agent_id] = dict()
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" where TRAJECTORIES_0"+constants.CURRENT_FILE_ID+".TRACK_ID="+str(agent_id)+" ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    emp_trajs[agent_id] = (res[0][6],[(row[6],row[1],row[2],row[3]/3.6,row[4]) for row in res])
    return emp_trajs

def main():
    emp_trajs = dict()
    constants.CURRENT_FILE_ID = '769'
    constants.show_plots = True
    parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'game_theoretic_planner_cache'))
    constants.L3_ACTION_CACHE = os.path.join(parentDirectory,'l3_action_trajectories_'+constants.CURRENT_FILE_ID)
    ''' get the agents who have greater than 1 relevent agent '''
    file_id = constants.CURRENT_FILE_ID
    centerlines = construct_centerline(False)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
    c = conn.cursor()
    q_string = "select * from TRAFFIC_REGIONS_DEF WHERE TRAFFIC_REGIONS_DEF.REGION_PROPERTY='center_line'"
    c.execute(q_string)
    res = c.fetchall()
    seg_cls = {row[0]:(ast.literal_eval(row[4]),ast.literal_eval(row[5])) for row in res}
    q_string = "select * from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.TASK='W_S' AND EQUILIBRIUM_ACTIONS.RELEV_AGENT_IDS != '[]' order by track_id,time"
    c.execute(q_string)
    res = c.fetchall()
    track_time_map = dict()
    for row in res:
        emp_action = ast.literal_eval(row[15])[0] if len(ast.literal_eval(row[15])) > 0 else None
        ag_tote = AgentTote(row[3],row[4],ast.literal_eval(row[12]),emp_action)
        if ag_tote.agent_id not in emp_trajs:
            get_trajectory(ag_tote.agent_id, emp_trajs)
        if (row[3],row[4]) not in track_time_map:
            time_ts = row[4]
            print("populating",row[3],row[4],round(row[8]/3.6,2))
            pedestrian_info = utils.setup_pedestrian_info(time_ts)
            vehicles_info = utils.get_vehicles_info(time_ts)
            scene_state = SceneState(pedestrian_info,vehicles_info)
            populate_all_agent_info(ag_tote,pedestrian_info,scene_state)
            generate_s1_pt(ag_tote, centerlines, seg_cls)
            track_time_map[(row[3],row[4])] = ag_tote
            if constants.show_plots:
                for ra_id,ra_agent in ag_tote.relev_agents.items():
                    if ra_agent.id not in emp_trajs:
                        get_trajectory(ra_agent.id, emp_trajs)
                    '''
                    if row[4] >= emp_trajs[ra_agent.id][0] and row[4] <= emp_trajs[ra_agent.id][1][-1][0]:
                        idx = int(round((row[4] - emp_trajs[ra_agent.id][0])*constants.DATASET_FPS)) 
                        if idx+30 < len(emp_trajs[ra_agent.id][1]):
                            plt.plot([emp_trajs[ra_agent.id][1][idx+30][1]],[emp_trajs[ra_agent.id][1][idx][2]],'x',color='blue')
                    '''
                    plt.plot([x[1] for x in emp_trajs[ra_agent.id][1]],[x[2] for x in emp_trajs[ra_agent.id][1]],'-',color='blue')
                
                '''
                if row[4] >= emp_trajs[ag_tote.agent_id][0] and row[4] <= emp_trajs[ag_tote.agent_id][1][-1][0]:
                    idx = int(round((row[4] - emp_trajs[ag_tote.agent_id][0])*constants.DATASET_FPS))
                    if idx+30 < len(emp_trajs[ag_tote.agent_id][1]):
                        plt.plot([emp_trajs[ag_tote.agent_id][1][idx+30][1]],[emp_trajs[ag_tote.agent_id][1][idx][2]],'x',color='blue')
                '''
                plt.plot([x[1] for x in emp_trajs[ag_tote.agent_id][1]],[x[2] for x in emp_trajs[ag_tote.agent_id][1]],'-',color='blue')
                #plot_traffic_regions()
                plt.xlim(538780, 538890)
                plt.ylim(4813970, 4814055)
                img = plt.imread("D:\\behavior modeling\\background.jpg")
                plt.title("PNE-QE:MAXMIN:BOUNDARY")
                plt.imshow(img, extent=[538780, 538890, 4813970, 4814055])
                mng = plt.get_current_fig_manager()
                mng.window.state("zoomed")
                plt.show()
    conn.close()    
    
if __name__ == '__main__':
    main()
    