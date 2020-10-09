'''
Created on Aug 29, 2020

@author: Atrisha
'''
import sqlite3
import numpy as np
import constants
import math
import ast
from all_utils import utils
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, griddata, interp1d, splrep, splev
from collections import OrderedDict
import itertools
from visualizer import visualizer
import spline_utils
import emp_residuals


def generate_path2(hpx,hpy):
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
        cs_x = spline_utils.get_natural_cubic_spline_model(indx,s_x, minval=0, maxval=len(s_x), n_knots=3)
        cs_y = spline_utils.get_natural_cubic_spline_model(indx,s_y, minval=0, maxval=len(s_x),n_knots=3)
        ref_x = np.arange(indx[1],indx[-1]+.1,.1)
        for i_a in ref_x:
            path.append(((path[0][0]+cs_x.predict(i_a)[0]), ((path[0][1]+cs_y.predict(i_a)[0]))))
        path = path_prefix + path
    else:
        path = path_prefix
    path = list(OrderedDict.fromkeys(path))
    #all_paths.append(path)
    return path

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

   

def construct_centerline(as_lattice,veh_state):
    start_seg = veh_state.current_segment
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
        if start_seg not in k:
            continue
        segment_seq = list(k)
        cl_X, cl_Y = [], []
        cl_map = dict()
        for s_idx,seg in enumerate(segment_seq):
            if segment_seq.index(start_seg) > s_idx:
                continue
            X_pts, Y_pts = all_centerlines[seg][0], all_centerlines[seg][1]
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
            cl_map[seg] = [X_pts,Y_pts]
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
        fcl_X,fcl_Y = [],[]
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
            centerline_map[k] = [([round(x*2)/2 for x in gridpts_X],[round(x*2)/2 for x in gridpts_Y])]
        else:
            gridpts_X = cl_X
            gridpts_Y = cl_Y
            centerline_map[k] = [(gridpts_X,gridpts_Y)]
    return centerline_map

def add_parallel_cls(centerline_map):
    for k,v in centerline_map.items():
        if v is not None:
            centerline_map[k] = v
            if k[-1] == '1':
                tol_list = list(itertools.product([1,2],[1,2,3,4]))
            else:
                tol_list = list(itertools.product([1,2,3,4],[1,2]))
            for d in tol_list:
                lat1, lat2 = utils.add_parallel(list(zip(v[0][0][1:],v[0][1][1:])),d[0],d[1])
                centerline_map[k].append([[v[0][0][0]]+[x[0] for x in lat1],[v[0][1][0]]+[x[1] for x in lat1]])
                centerline_map[k].append([[v[0][0][0]]+[x[0] for x in lat2],[v[0][1][0]]+[x[1] for x in lat2]])
    return centerline_map

def get_trajectory(agent_id,file_id):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = "select * from TRAJECTORIES_0"+file_id+" where TRAJECTORIES_0"+file_id+".TRACK_ID="+agent_id+" ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    traj = [(row[6],row[1],row[2],row[3]/3.6,row[4]) for row in res]
    return traj
    
def main():
    as_lattice = False
    agent_id = '56'
    time_ts = 50.05
    file_id = '769'
    constants.CURRENT_FILE_ID = file_id
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    q_string = "select distinct TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ from TRAJECTORY_MOVEMENTS where TRAJECTORY_MOVEMENTS.TRACK_ID="+str(agent_id)
    c.execute(q_string)
    res = c. fetchone()
    seg_seq = res[0]
    parentDirectory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'game_theoretic_planner_cache'))
    constants.L3_ACTION_CACHE = os.path.join(parentDirectory,'l3_action_trajectories_'+constants.CURRENT_FILE_ID)
    ''' get the agents who have greater than 1 relevent agent '''
    ra_file_key = os.path.join(constants.L3_ACTION_CACHE, str(agent_id)+'-'+str(0)+'_'+str(time_ts).replace('.', ','))
    if os.path.exists(ra_file_key):
        ag_info = utils.pickle_load(ra_file_key)
    else:
        ag_info = utils.setup_vehicle_state(agent_id,time_ts)
    ag_info.nose_x = ag_info.x + (constants.CAR_LENGTH/2)*np.cos(ag_info.yaw)
    ag_info.nose_y = ag_info.y + (constants.CAR_LENGTH/2)*np.sin(ag_info.yaw)
    centerlines = construct_centerline(as_lattice,ag_info)
    #centerlines = add_parallel_cls(centerlines)
    this_cls = [v for k,v in centerlines.items() if ag_info.current_segment in k]
    emp_traj = get_trajectory(agent_id,file_id)
    emp_path = generate_path([emp_traj[i][1] for i in np.arange(0,len(emp_traj),3)], [emp_traj[i][2] for i in np.arange(0,len(emp_traj),3)])
    #plt.plot([x[0] for x in emp_path],[x[1] for x in emp_path],'c-')
    for this_cl in this_cls:
        for lines in this_cl:
            #this_cl = centerlines[tuple(ast.literal_eval(seg_seq))]
            #cl_len = len(this_cl[0])
            #knot_indxs = [0,1*cl_len//3,2*cl_len//3,cl_len-1]
            knot_pts = (lines[0],lines[1])
            path = generate_path(knot_pts[0], knot_pts[1])
            plt.plot(lines[0][1:],lines[1][1:],'bx')
            plt.plot(lines[0][:2],lines[1][:2],'bo')
            #for path in all_paths:
            plt.plot([x[0] for x in path],[x[1] for x in path],'r-')
        
    plt.plot([x[1] for x in emp_traj],[x[2] for x in emp_traj],'g-')
    #visualizer.plot_traffic_regions()
    plt.axis("equal")
    conn_traj = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_generated_trajectories_'+file_id+'.db')
    c_traj = conn_traj.cursor()
    res_gen = emp_residuals.ResidualGeneration(file_id)
    for k,v in res_gen.traj_info_dict.items():
        if k==round(time_ts) or k==round(time_ts)+1:
            for ag,ag_traj in v.items():
                if ag[0] == int(agent_id) and ag[1] == 0:
                    plt.plot([x[0] for x in ag_traj], [x[1] for x in ag_traj],c='black')
                    break
    f=1
    plt.show()
    #plt.plot([x[0] for x in emp_traj],[x[3] for x in emp_traj],'g-')
    #plt.show()
    f=1
    
def plot_all_lwn():
    traj = []
    all_files = [769,770,771]+np.arange(775,780).tolist()
    for file_id in all_files:
        print(file_id)
        constants.CURRENT_FILE_ID = str(file_id)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "select * from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" where TRAJECTORIES_0"+constants.CURRENT_FILE_ID+".TRACK_ID in (select track_id from  TRAJECTORY_MOVEMENTS WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ LIKE '%ln_s_1%') ORDER BY track_id,TIME"
        c.execute(q_string)
        res = c.fetchall()
        for i in np.arange(0,len(res),30):
            traj.append((res[i][1],res[i][2]))
    plt.plot([x[0] for x in traj], [x[1] for x in traj], '.')
    plt.show()
if __name__ == '__main__':
    #visualizer.plot_traffic_regions()
    main()
    #plot_all_lwn()
    
    