'''
Created on Jan 16, 2020

@author: Atrisha
'''
import sqlite3
import math
import numpy as np
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
#import matplotlib.pyplot as plt
import ast
import sys
import constants
#from matplotlib import cm
import pickle
from motion_planners.planning_objects import VehicleState,PedestrianState,TrajectoryDef
import os,shutil
from matplotlib import path
from decimal import Decimal
from collections import OrderedDict
import pandas as pd
from builtins import isinstance
import statistics
from scipy.interpolate import CubicSpline
from itertools import islice
import logging
import planning_objects
logging.basicConfig(format='%(levelname)-8s %(filename)s: %(message)s',level=logging.INFO)
log = constants.common_logger



PI = math.pi
PI_BY_2 = math.pi/2

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_numpy(A, B, P):
    A,B,P = np.asarray(A),np.asarray(B),np.asarray(P)
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    angle_with_A = arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A)))
    angle_with_B = arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2
    if angle_with_A > pi / 2:
        return norm(P - A) * math.sin(pi - angle_with_A)
    if angle_with_B > pi / 2:
        return norm(P - B) * math.sin(pi - angle_with_B)
    #return norm(cross(A-B, A-P))/norm(B-A),
    return cross(A-B, A-P)/norm(B-A)

#from https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self.inverse.setdefault(tuple(value.tolist()),[]).append(key)
            else:
                self.inverse.setdefault(tuple(value),[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def split_traj_metadata_by_agents(m_data):
    agents = dict()
    for k,v in m_data.items():
        if k == 'raw_data':
            for rdk,rdv in v.items():
                (ag_id,relev_agent) = [int(s) for s in rdk.split('-')]
                if ag_id not in agents:
                    agents[ag_id] = [relev_agent]
                else:
                    agents[ag_id].append(relev_agent)
    split_dict = {k:dict() for k in agents.keys()}
    for ag_id,ag_v in agents.items():
        for k,v in m_data.items():
            if k == 'raw_data':
                split_dict[ag_id][k] = dict()
                for rd_k,rd_v in v.items():
                    if int(rd_k.split('-')[0]) == ag_id:
                        split_dict[ag_id][k][rd_k] = rd_v
            elif k == 'relev_agents':
                split_dict[ag_id][k] = dict()
                for ra_k,ra_v in v.items():
                    if ra_k in ag_v:
                        split_dict[ag_id][k][ra_k] = ra_v
            else:
                if k == ag_id:
                    split_dict[ag_id][k] = v
    return split_dict

def angle_between_lines(A,B):
    unit_vector_1 = A / np.linalg.norm(A)
    unit_vector_2 = B / np.linalg.norm(B)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def eval_complexity(traj,strat_str):
    if isinstance(traj, pd.DataFrame):
        rv = traj['v'].to_numpy()
        rx = traj['x'].to_numpy()
        ry = traj['y'].to_numpy()
        ra = traj['a'].to_numpy()
        time = traj['time'].to_numpy()
    else:
        assert(traj.shape[1]==9)
        rv,rx,ry,ra,time = traj[:,6],traj[:,3],traj[:,4],traj[:,7],traj[:,2]
    dt = constants.LP_FREQ
    traj_def = TrajectoryDef(strat_str)
    '''
    plt.plot(time,rv,'g')
    plt.plot(time,ra,'r')
    plt.show()
    #linear_plan_x = all_utils.linear_planner(sx, vxs, axs, gx, vxg, axg, max_accel, max_jerk, dt)
    '''
    hpx = [rx[0],rx[len(rx)//3],rx[2*len(rx)//3],rx[-1]]
    hpy = [ry[0],ry[len(ry)//3],ry[2*len(ry)//3],ry[-1]]
    _d = OrderedDict(sorted(list(zip(hpx,hpy)),key=lambda tup: tup[0]))
    hpx,hpy = list(_d.keys()),list(_d.values())
    try:
        cs = CubicSpline(hpx, hpy)
    except ValueError:
        print(hpx,hpy)
        raise
    path_residuals = sum([abs(cs(x)-ry[_i]) for _i,x in enumerate(rx)])
    new_vels = generate_baseline_trajectory(time,[(x,cs(x)) for x in rx],rv[0],ra[0],traj_def.max_acc_long/2,traj_def.max_jerk/2,rv[-1],dt,traj_def.acc)
    slice_len = min(len(new_vels),len(rv))
    vel_residuals = sum([abs(x[0]-x[1]) for x in zip(rv[:slice_len],  new_vels[:slice_len])])
    compl = math.hypot(path_residuals, vel_residuals)
    return compl
'''
def line_intersection(line_1,line_2):
    x1,y1,x2,y2,x3,y3,x4,y4 = line_1[0][0], line_1[1][0], line_1[0][1], line_1[1][1], line_2[0][0], line_2[1][0], line_2[0][1], line_2[1][1]
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return (px, py)
'''

def get_forward_line(pt,yaw,line):
    forward_line = []
    for l_pt in line:
        angle_to_line_pt = math.atan2(l_pt[1]-pt[1], l_pt[0]-pt[0])
        yaw = yaw % (2*math.pi)
        angle_to_line_pt = angle_to_line_pt if angle_to_line_pt > 0 else (2*math.pi)-abs(angle_to_line_pt)
        if abs(angle_to_line_pt-yaw) < math.pi/2 or 2*math.pi-abs(angle_to_line_pt-yaw) < math.pi/2:
            forward_line.append(l_pt)
    return forward_line

def angle_between_lines_2pi(line_1,line_2):
    '''x2-x1'''
    line_1_angle = math.atan2(line_1[1][1]-line_1[0][1], line_1[1][0]-line_1[0][0])
    line_1_angle = line_1_angle if line_1_angle > 0 else (2*math.pi)-abs(line_1_angle)
    line_2_angle = math.atan2(line_2[1][1]-line_2[0][1], line_2[1][0]-line_2[0][0])
    line_2_angle = line_2_angle if line_2_angle > 0 else (2*math.pi)-abs(line_2_angle)
    diff = abs(line_1_angle-line_2_angle)
    return diff

def get_target_velocity(veh_state):
    action = veh_state.l1_action
    q_string = "select * from TARGET_VELOCITIES where action='"+str(action)+"'"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    c.execute(q_string)
    res = c.fetchall()
    target_vels = {row[0]:(row[3],row[4]) if row[3] != -1 else (veh_state.leading_vehicle.speed,1) for row in res}
    return target_vels

def get_centerline(lane_segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+lane_segment+"' and REGION_PROPERTY='center_line'"
    c.execute(q_string)
    res = c.fetchall()
    center_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        center_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    if center_coordinates is None:
        sys.exit("no centerline found for "+lane_segment)
    return center_coordinates

def get_mean_yaws_for_segments(segments):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    if isinstance(segments, str):
        segments = "('"+segments+"')"
        q_string = "SELECT gate_id, AVG(YAW) FROM GATE_CROSSING_EVENTS WHERE GATE_CROSSING_EVENTS.GATE_ID in "+segments+" group by gate_id"
    else:
        q_string = "SELECT gate_id, AVG(YAW) FROM GATE_CROSSING_EVENTS WHERE GATE_CROSSING_EVENTS.GATE_ID in "+str(tuple(segments))+" group by gate_id"
    c.execute(q_string)
    res = c.fetchall()
    yaws = {r[0]:r[1] for r in res}
    return yaws
    

def get_leading_vehicles(veh_state):
    path = veh_state.segment_seq
    time_ts = veh_state.current_time
    current_segment = veh_state.current_segment
    next_segment_idx = veh_state.segment_seq.index(current_segment)+1
    next_segment = veh_state.segment_seq[next_segment_idx] if next_segment_idx < len(veh_state.segment_seq) else veh_state.segment_seq[-1]
    veh_id = veh_state.id
    veh_pos_x = float(veh_state.x)
    veh_pos_y = float(veh_state.y)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    ''' find the exit boundaries of the current and next segment. This will help calculate which vehicles are ahead.'''
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE (NAME like '"+current_segment+"%' OR NAME like '"+next_segment+"') and REGION_PROPERTY = 'exit_boundary'"
    ex_b_positions = dict()
    c.execute(q_string)
    res_exit_b = c.fetchall()
    for ex_b in res_exit_b:
        ex_b_positions[ex_b[0]] = (ast.literal_eval(ex_b[4]),ast.literal_eval(ex_b[5]))
    #print(list(ex_b_positions.keys()))
    #veh_dist_to_segment_exit = (math.hypot(ex_b_positions[current_segment][0][0] - veh_pos_x, ex_b_positions[current_segment][1][0] - veh_pos_y) + \
    #                            math.hypot(ex_b_positions[current_segment][0][1] - veh_pos_x, ex_b_positions[current_segment][1][1] - veh_pos_y))/2
    veh_vect_to_segment_exit = [((ex_b_positions[current_segment][0][0] - veh_pos_x) + (ex_b_positions[current_segment][0][1] - veh_pos_x))/2,\
                                ((ex_b_positions[current_segment][1][0] - veh_pos_y) + (ex_b_positions[current_segment][1][1] - veh_pos_y))/2]
    ''' find the vehicles that are in the current segment or the next and appears within the window of the subject vehicle '''
    if next_segment[-2] == '-':
        q_string = "SELECT T.TRACK_ID FROM TRAJECTORY_MOVEMENTS T, v_TIMES V WHERE (T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+current_segment+"''%' OR T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+next_segment[:-1]+"%') AND T.TRACK_ID = V.TRACK_ID AND (V.ENTRY_TIME <= "+str(time_ts)+" AND V.EXIT_TIME >= "+str(time_ts)+") AND T.TRACK_ID <> "+str(veh_id)
    else:
        q_string = "SELECT T.TRACK_ID FROM TRAJECTORY_MOVEMENTS T, v_TIMES V WHERE (T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+current_segment+"''%' OR T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+next_segment+"''%') AND T.TRACK_ID = V.TRACK_ID AND (V.ENTRY_TIME <= "+str(time_ts)+" AND V.EXIT_TIME >= "+str(time_ts)+") AND T.TRACK_ID <> "+str(veh_id)
    c.execute(q_string)
    res = c.fetchall()
    potential_lead_vehicles = []
    if len(res) > 0:
        for row in res:
            leading_vehicle_id = row[0]
            ''' find the position of the potential lead vehicle in the current time '''
            q_string = "select * from trajectories_0769,trajectories_0769_ext where trajectories_0769.track_id=trajectories_0769_ext.track_id and trajectories_0769.time=trajectories_0769_ext.time and trajectories_0769.track_id="+str(leading_vehicle_id)+" and trajectories_0769.time = "+str(time_ts)
            c.execute(q_string)
            pt_res = c.fetchone()
            l_v_state = VehicleState()
            if pt_res is None:
                ''' this means that there is no entry for this vehicle in trajectories_0769_ext yet'''
                continue
            l_v_state.set_id(pt_res[0])
            l_v_state.set_current_time(time_ts)
            l_v_track = get_track(l_v_state,time_ts)
            l_v_state.set_track_info(l_v_track[0,])
            l_v_track_segment_seq = get_track_segment_seq(l_v_state.id)
            l_v_state.set_segment_seq(l_v_track_segment_seq)
            l_v_current_segment = pt_res[11]
            l_v_state.set_current_segment(l_v_current_segment)
            l_v_state.set_current_l1_action(pt_res[12])
            if l_v_current_segment not in ex_b_positions.keys():
                ''' potential lead vehicle is not in the path (current or the next segment), so ignore '''
                continue
            else:
                lead_vehicle_pos = (float(pt_res[1]),float(pt_res[2]))
                l_v_segment_ex_b = ex_b_positions[l_v_current_segment]
                #l_v_vect_to_segment_exit = (math.hypot(ex_b_positions[l_v_current_segment][0][0] - lead_vehicle_pos[0], ex_b_positions[l_v_current_segment][1][0] - lead_vehicle_pos[1]) + \
                #                    math.hypot(ex_b_positions[l_v_current_segment][0][1] - lead_vehicle_pos[0], ex_b_positions[l_v_current_segment][1][1] - lead_vehicle_pos[1]))/2
                                    
                l_v_vect_to_segment_exit = [((ex_b_positions[l_v_current_segment][0][0] - lead_vehicle_pos[0]) + (ex_b_positions[l_v_current_segment][0][1] - lead_vehicle_pos[0]))/2,\
                                ((ex_b_positions[l_v_current_segment][1][0] - lead_vehicle_pos[1]) + (ex_b_positions[l_v_current_segment][1][1] - lead_vehicle_pos[1]))/2]
                l_v_state.set_vect_to_segment_exit(l_v_vect_to_segment_exit)
                if l_v_current_segment == current_segment and np.linalg.norm(l_v_vect_to_segment_exit) > np.linalg.norm(veh_vect_to_segment_exit):
                    ''' this vehicle is behind the subject vehicle '''
                    continue
                elif math.hypot(lead_vehicle_pos[0]-veh_pos_x,lead_vehicle_pos[1]-veh_pos_y) > constants.LEAD_VEH_DIST_THRESH:
                    ''' this vehicle is too far '''
                    continue
                else:
                    potential_lead_vehicles.append(l_v_state)
            
        if len(potential_lead_vehicles) > 1:
            #sys.exit('need to resolve multiple potential lead vehicles ')
            lv_idx,min_dist = 0,np.inf
            for idx, lv in enumerate(potential_lead_vehicles):
                dist_from_subject = math.hypot(float(lv.x)-veh_pos_x, float(lv.y)-veh_pos_y)
                if dist_from_subject < min_dist:
                    lv_idx = idx
            return potential_lead_vehicles[lv_idx]
                    
                
        else:
            return potential_lead_vehicles[0] if len(potential_lead_vehicles) ==1 else None
    else:
        return None    

def pickle_load(file_key):
    l3_actions = pickle.load( open( file_key, "rb" ) )
    return l3_actions

def pickle_dump(cache,file_key,l3_actions):
    file_key = os.path.join(constants.ROOT_DIR,cache,file_key)
    directory = os.path.dirname(file_key)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump( l3_actions, open( file_key, "wb" ) )
    
def dist_along_yaw(pt1,pt2,yaw,pos):
    ''' r*cos(yaw-slope + 90) = d '''
    d = abs(distance_numpy([pt1[0],pt1[1]], [pt2[0],pt2[1]], [pos[0],pos[1]]))
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    r = d / (math.cos(yaw - slope + (.5*math.pi)))
    return abs(r)

def find_nearest_in_array(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def kph_to_mps(kph):
    return kph/3.6

def clear_cache(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def fresnet_to_map(o_x,o_y,X,Y,centerline_angle):
    M_X,M_Y = [],[]
    a = centerline_angle
    h,k = o_x,o_y
    ''' we had our points in right hand rule, so we need to reflect the y'''
    Y = [-y for y in Y]
    ''' from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    rotation_and_translation_matrix = np.asarray([[np.cos(a), -np.sin(a), h],\
                                                 [np.sin(a), np.cos(a), k],\
                                                 [0, 0, 1]])
    '''
    translation_matrix = np.asarray([[1, 0, h],\
                                    [0, 1, k],\
                                    [0, 0, 1]])
    
    rotation_matrix = np.asarray([[np.cos(a), -np.sin(a), 0],\
                                 [np.sin(a), np.cos(a), 0],\
                                 [0, 0, 1]])
    
    for x,y in zip(X,Y):
        point = np.asarray([x, y, 1]).T
        rotated_point = np.matmul(rotation_matrix, point)
        new_point = np.matmul(translation_matrix,rotated_point)
        M_X.append(new_point[0])
        M_Y.append(new_point[1])
    return M_X,M_Y

def map_to_fresnet(o_x,o_y,X,Y,yaw):
    F_X,F_Y = [],[]
    a = yaw
    h,k = o_x,o_y
    ''' from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    rotation_and_translation_matrix = np.asarray([[np.cos(a), -np.sin(a), h],\
                                                 [np.sin(a), np.cos(a), k],\
                                                 [0, 0, 1]])
    '''
    translation_matrix = np.asarray([[1, 0, -h],\
                                    [0, 1, -k],\
                                    [0, 0, 1]])
    
    rotation_matrix = np.asarray([[np.cos(a), np.sin(a), 0],\
                                 [-np.sin(a), np.cos(a), 0],\
                                 [0, 0, 1]])
    
    for x,y in zip(X,Y):
        point = np.asarray([x, y, 1]).T
        translated_point = np.matmul(translation_matrix,point)
        new_point = np.matmul(rotation_matrix, translated_point)
        F_X.append(new_point[0])
        F_Y.append(-new_point[1])
    return F_X,F_Y
    
        
def remove_files(cache_dir):
    path = os.path.join(constants.ROOT_DIR,cache_dir)
    filelist = [ f for f in os.listdir(path)]
    for f in filelist:
        os.remove(os.path.join(path, f))      
        
def get_processed_files(cache_dir):
    path = os.path.join(constants.ROOT_DIR,cache_dir)
    filelist = [ f for f in os.listdir(path)]
    return filelist

def split_in_n(pt1,pt2,N):
    step = (pt2[0] - pt1[0])/N
    x_coords = [pt1[0] + (step*i) for i in np.arange(N)]
    x_coords = x_coords + [pt2[0]]
    step = (pt2[1] - pt1[1])/N
    y_coords = [pt1[1] + (step*i) for i in np.arange(N)]
    y_coords = y_coords + [pt2[1]]
    return list(zip(x_coords,y_coords))

def construct_state_grid(pt1,pt2,N,tol,grid_type):
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    slope_comp = (math.pi/2) - slope
    
    if grid_type == 'line':
        central_coords = split_in_n(pt1,pt2,N)
        grid = [central_coords]
    else:
        ''' grid type is point. return stop positions along a line '''
        central_coords = [(pt1[0]+(pt2[0]-pt1[0])/2,pt1[1](pt2[1]-pt1[1])/2)]
        grid = [central_coords]
    for r in tol:
        grid.append([(x[0]-(r*math.cos(slope_comp)),x[1]+(r*math.sin(slope_comp))) for x in central_coords])
    return np.asarray(grid)


def reduce_relev_agents(agent_id,time_ts,relev_agents):
    ag_file_key = os.path.join(constants.ROOT_DIR,constants.L3_ACTION_CACHE,str(agent_id)+'-0_'+str(time_ts).replace('.', ','))
    found = False
    if os.path.exists(ag_file_key):
        veh_state = pickle_load(ag_file_key)
        found = True
    else:
        veh_state = setup_vehicle_state(agent_id, time_ts)
    veh_state.set_dist_to_sub_agent(0.0)
    ra_states = []
    for r_a_id in relev_agents:
        found = False
        ag_file_key = os.path.join(constants.ROOT_DIR,constants.L3_ACTION_CACHE,str(veh_state.id)+'-'+str(r_a_id)+'_'+str(time_ts).replace('.', ','))
        if os.path.exists(ag_file_key):
            r_a_state = pickle_load(ag_file_key)
            found = True
        else:
            r_a_state = setup_vehicle_state(r_a_id, time_ts)
        #log.info('trying to reduce'+str(r_a_id)+' '+str(found))
        dist_to_sv = math.hypot(veh_state.x-r_a_state.x,veh_state.y-r_a_state.y)
        r_a_state.set_dist_to_sub_agent(dist_to_sv)
        ra_states.append(r_a_state)
    ra_map = dict()
    for k,v in constants.RELEV_REDUCTION_MAP.items():
        for relev_agent in ra_states:
            if constants.TASK_MAP[veh_state.direction] != k:
                break
            else:
                if k not in ra_map:
                    ra_map[k] = dict()
                for incl_desc in v:
                    if constants.SEGMENT_MAP[relev_agent.current_segment] == incl_desc[0]:
                        if incl_desc[1] is not None:
                            if relev_agent.current_segment[-1] == incl_desc[1]:
                                if incl_desc not in ra_map[k]:
                                    ra_map[k][incl_desc] = [relev_agent]
                                else:
                                    ra_map[k][incl_desc].append(relev_agent)
                        else:
                            if incl_desc not in ra_map[k]:
                                ra_map[k][incl_desc] = [relev_agent]
                            else:
                                ra_map[k][incl_desc].append(relev_agent)
    incl_list = [veh_state.leading_vehicle.id] if veh_state.leading_vehicle is not None else []
    for dir,v_dir in ra_map.items():
        for incl_desc,ras in v_dir.items():
            ras.sort(key=lambda x: x.dist_to_sv)
            if incl_desc[3] is not None:
                ras = [x for x in ras if x.dist_to_sv<=incl_desc[3]]
            if len(ras) > 0:
                if incl_desc[2] == 1:
                    incl_list.append(ras[0].id)
                else:
                    incl_list.append(ras[0].id)
                    if len(ras) > 1:
                        incl_list.append(ras[-1].id)
            ra_map[dir][incl_desc] = incl_list
    return incl_list
                    
                                
                      
        
    

''' add two parallel lines to line at a lateral distance on either side of line'''
def add_parallel(line, dist):
    #if len(line) < 2:
        #logging.warn("attemped to add parallel to line size less than 2")
        
    pointline = True if len(line) < 2 else False
    p_line_lat1, p_line_lat2 = [],[]
    if not pointline:
        for l_idx in np.arange(1,len(line)):
            pt1 = line[l_idx-1]
            pt2 = line[l_idx]
            cl_angle = math.atan2(pt2[1]-pt1[1], pt2[0]-pt1[0])
            cl_angle = cl_angle if cl_angle > 0 else (2*math.pi) - abs(cl_angle)
            cl_normal = (cl_angle + (math.pi/2))%(2*math.pi)
            ''' add the point normal to pt1 and pt2 on both sides '''
            pt1_lat1 = (pt1[0] + dist*np.cos(cl_normal), pt1[1] + dist*np.sin(cl_normal))
            pt1_lat2 = (pt1[0] - dist*np.cos(cl_normal), pt1[1] - dist*np.sin(cl_normal))
            
            pt2_lat1 = (pt2[0] + dist*np.cos(cl_normal), pt2[1] + dist*np.sin(cl_normal))
            pt2_lat2 = (pt2[0] - dist*np.cos(cl_normal), pt2[1] - dist*np.sin(cl_normal))
            if l_idx == 1:
                p_line_lat1.append(pt1_lat1)
                p_line_lat2.append(pt1_lat2)
            p_line_lat1.append(pt2_lat1)
            p_line_lat2.append(pt2_lat2)
            
    return p_line_lat1, p_line_lat2
        
         
    
    
    
'''
plt.plot([0,20],[0,30],'ro')
split_pts = split_in_n((0,0), (20,30), 10)
plt.plot([x[0] for x in split_pts],[x[1] for x in split_pts],'bx')
plt.show()
'''

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

def get_track_segment_seq(track_id):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRAFFIC_SEGMENT_SEQ FROM TRAJECTORY_MOVEMENTS WHERE TRACK_ID = "+str(track_id)
    c.execute(q_string)
    res = c.fetchall()
    seq = []
    for row in res:
        seq = row[0]
    conn.close()
    return ast.literal_eval(seq) if seq is not None else None


def load_traj_ids_for_traj_info_id(traj_info_id,baseline_only):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    if not baseline_only:
        q_string = "SELECT DISTINCT TRAJECTORY_ID FROM GENERATED_TRAJECTORY WHERE TRAJECTORY_INFO_ID = "+str(traj_info_id)
    else:
        q_string = "SELECT DISTINCT TRAJECTORY_ID FROM GENERATED_BASELINE_TRAJECTORY WHERE TRAJECTORY_INFO_ID = "+str(traj_info_id)
    c.execute(q_string)
    res = c.fetchall()
    traj_ids = [row[0] for row in res]
    return traj_ids


def dict_chunks(data,SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def load_trajs_for_traj_info_id(traj_info_id,baseline_only=True):
    traj_info_id = [int(e) for e in traj_info_id]
    import struct
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    traj_dict = dict()
    if not baseline_only:
        if len(traj_info_id) > 1:        
            q_string = "select * from GENERATED_BOUNDARY_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(traj_info_id)) + ' UNION ' + \
                     "select * from GENERATED_BASELINE_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(traj_info_id))+" order by trajectory_id,time"
        else:
            q_string = "select * from GENERATED_BOUNDARY_TRAJECTORY where TRAJECTORY_INFO_ID = "+str(traj_info_id[0]) + ' UNION ' + \
                     "select * from GENERATED_BASELINE_TRAJECTORY where TRAJECTORY_INFO_ID = "+str(traj_info_id[0])+" order by trajectory_id,time"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[1] not in traj_dict:
                traj_dict[row[1]] = dict()
                traj_dict[row[1]][row[0]] = [list(row[1:])]
            else:
                if row[0] not in traj_dict[row[1]]:
                    traj_dict[row[1]][row[0]] = [list(row[1:])]
                else:
                    traj_dict[row[1]][row[0]].append(list(row[1:]))
        for k,v in traj_dict.items():
            for k1,v1 in v.items():
                traj_dict[k][k1] = np.vstack(v1)
        return traj_dict
    else:
        info_id_dict = dict()
        if len(traj_info_id) > 1:
            q_string = "select distinct(TRAJECTORY_ID),TRAJECTORY_INFO_ID FROM GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID in "+str(tuple(traj_info_id))+" ORDER BY TRAJECTORY_INFO_ID,TRAJECTORY_ID"
        else:
            q_string = "select distinct(TRAJECTORY_ID),TRAJECTORY_INFO_ID FROM GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID = "+str(traj_info_id[0])+" ORDER BY TRAJECTORY_ID"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[1] in info_id_dict:
                info_id_dict[row[1]].append(row[0])
            else:
                info_id_dict[row[1]] = [row[0]]
        selected_traj_ids = []
        for info_id,traj_ids in info_id_dict.items():
            if len(traj_ids) > 3:
                selected_traj_ids.append(traj_ids[1])
            else:
                selected_traj_ids.append(traj_ids[0])
        if len(selected_traj_ids) > 1:        
            q_string = "SELECT * FROM GENERATED_BASELINE_TRAJECTORY WHERE TRAJECTORY_ID IN "+str(tuple(selected_traj_ids))
        else:
            q_string = "SELECT * FROM GENERATED_BASELINE_TRAJECTORY WHERE TRAJECTORY_ID = "+str(selected_traj_ids[0])
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[1] not in traj_dict:
                traj_dict[row[1]] = [[struct.unpack('f', x)[0] if isinstance(x,bytes) else x for x in list(row[1:])]]
            else:
                traj_dict[row[1]].append([struct.unpack('f', x)[0] if isinstance(x,bytes) else x for x in list(row[1:])])
        for k,v in traj_dict.items():
            traj_dict[k] = np.vstack(v)
    return traj_dict


def get_merging_vehicle(veh_state):
    return None

def setup_vehicle_state(veh_id,time_ts):
    r_a_state = VehicleState()
    r_a_state.set_id(veh_id)
    r_a_state.set_current_time(time_ts)
    r_a_track = get_track(r_a_state,time_ts)
    r_a_track_segment_seq = get_track_segment_seq(veh_id)
    r_a_state.set_segment_seq(r_a_track_segment_seq)
    r_a_state.action_plans = dict()
    r_a_state.set_current_time(time_ts)
    entry_exit_time = get_entry_exit_time(r_a_state.id)
    r_a_state.set_entry_exit_time(entry_exit_time)
    if len(r_a_track) == 0:
        ''' this agent is out of the view currently'''
        r_a_state.set_out_of_view(True)
        r_a_track = None
    else:
        r_a_state.set_out_of_view(False)
        r_a_state.set_track_info(r_a_track[0,])
        r_a_track = r_a_track[0,]
    
    if r_a_state.out_of_view or r_a_track[11] is None:
        r_a_track_info = guess_track_info(r_a_state,r_a_track)
        if r_a_track_info[1,] is None:
            brk = 1
        r_a_state.set_track_info(r_a_track_info)
        r_a_track_region = r_a_track_info[8,]
        if r_a_track_region is None:
            sys.exit('need to guess traffic region for relev agent')
        r_a_current_segment = get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,time_ts)
    else:
        r_a_current_segment = r_a_track[11]
    
        
    r_a_state.set_current_segment(r_a_current_segment)
    ''' 
    r_a_current_segment = r_a_track[0,11]
    r_a_state.set_current_segment(r_a_current_segment)
    #for now we will only take into account the leading vehicles of the subject agent's relevant vehicles when constructing the possible actions.'''
    lead_vehicle = get_leading_vehicles(r_a_state)
    r_a_state.set_leading_vehicle(lead_vehicle)
    merging_vehicle = get_merging_vehicle(r_a_state)
    r_a_state.set_merging_vehicle(merging_vehicle)
    r_a_direction = 'L_'+r_a_track_segment_seq[0][3].upper()+'_'+r_a_track_segment_seq[-1][3].upper()
    r_a_task = constants.TASK_MAP[r_a_direction]
    r_a_state.set_task(r_a_task)
    r_a_traffic_light = get_traffic_signal(time_ts, r_a_direction)
    r_a_state.set_traffic_light(r_a_traffic_light)
    r_a_state.set_direction(r_a_direction)
    next_signal_change = get_time_to_next_signal(time_ts, r_a_direction, r_a_traffic_light)
    r_a_state.set_time_to_next_signal(next_signal_change)
    return r_a_state

def calc_traj_len(traj):
    if len(traj) > 3:
        traj_s = [traj[i] for i in np.arange(0,len(traj),3)]
    else:
        traj_s = traj
    return sum([math.hypot(x2[0]-x1[0], x2[1]-x1[1]) for x1,x2 in zip(traj_s[:-1],traj_s[1:])])

def load_traj_from_db(all_traj_id_list,baseline_only):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    traj_dict_list = []
    for traj_id_list in all_traj_id_list:
        if not baseline_only:
            q_string = "SELECT * FROM GENERATED_TRAJECTORY WHERE trajectory_id in "+str(tuple(traj_id_list))+" order by trajectory_id,time"
        else:
            q_string = "SELECT * FROM GENERATED_BASELINE_TRAJECTORY WHERE trajectory_id = "+str(traj_id_list[0])+" order by trajectory_id,time"
        c.execute(q_string)
        res = c.fetchall()
        trajs = dict()
        for row in res:
            if row[0] not in trajs:
                trajs[row[0]] = [list(row[2:])]
            else:
                trajs[row[0]].append(list(row[2:]))
        trajs = {k:np.vstack(v) for k,v in trajs.items()}
        traj_dict_list.append(trajs)
    return traj_dict_list 
    

def load_traj_from_str(file_str):
    traj = pickle_load(file_str)
    p_d_list = []
    data_index = ['time', 'x', 'y', 'yaw', 'v', 'a', 'j']
    for i in np.arange(traj.shape[0]):
        _t_data,_t_type = traj[i][0],traj[i][1]
        if not isinstance(_t_data,np.ndarray):
            _t_data = list(_t_data)[:len(data_index)]
            for t_i,t in enumerate(_t_data):
                if isinstance(t,int):
                    _t_data[t_i] = [None]*len(_t_data[0])
            min_len = min([len(x) for x in _t_data])
            _t_data = [x[:min_len] for x in _t_data]
            _t_data = np.vstack(_t_data)
        s = pd.DataFrame(_t_data, index=data_index, dtype = np.float).T
        s = s.round(5)
        p_d_list.append(s)
    return p_d_list

def get_track(veh_state,curr_time,from_current=None):
    agent_id = veh_state.id
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    if curr_time is not None:
        if from_current is None:
            q_string = "select * from trajectories_0769,trajectories_0769_ext where trajectories_0769.track_id=trajectories_0769_ext.track_id and trajectories_0769.time=trajectories_0769_ext.time and trajectories_0769.track_id="+str(agent_id)+" and trajectories_0769.time="+str(curr_time)+" order by trajectories_0769.time"
        else:
            q_string = "select * from trajectories_0769,trajectories_0769_ext where trajectories_0769.track_id=trajectories_0769_ext.track_id and trajectories_0769.time=trajectories_0769_ext.time and trajectories_0769.track_id="+str(agent_id)+" and trajectories_0769.time >="+str(curr_time)+" order by trajectories_0769.time"
    else:
        q_string = "select * from trajectories_0769,trajectories_0769_ext where trajectories_0769.track_id=trajectories_0769_ext.track_id and trajectories_0769.time=trajectories_0769_ext.time and trajectories_0769.track_id="+str(agent_id)+" order by trajectories_0769.time"
    c.execute(q_string)
    res = c.fetchall()
    l = []
    '''
    if len(res) == 0:
        if time_ts is not None:
            if from_current is None:
                q_string = "select * from trajectories_0769 where trajectories_0769.track_id="+str(agent_id)+" and trajectories_0769.time="+str(time_ts)+" order by trajectories_0769.time"
            else:
                q_string = "select * from trajectories_0769 where trajectories_0769.track_id="+str(agent_id)+" and trajectories_0769.time >="+str(time_ts)+" order by trajectories_0769.time"
        else:
            q_string = "select * from trajectories_0769 where trajectories_0769.track_id="+str(agent_id)+" order by trajectories_0769.time"
        c.execute(q_string)
        res = c.fetchall()
    '''    
    for row in res:
        l.append(row)
    conn.close()
    return np.asarray(l)

def get_current_segment_by_veh_id(veh_id,time_ts):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT ASSIGNED_SEGMENT FROM TRAJECTORIES_0769_EXT WHERE TRACK_ID="+str(veh_id)+" AND TIME="+str(time_ts)
    c.execute(q_string)
    res = c.fetchone()
    curr_segment = res[0]
    return curr_segment
    
    

def get_pedestrian_track(p_state,q_time):
    p_id = p_state.p_id
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID="+str(p_id)+" AND ROUND(TIME,2)=ROUND("+str(q_time)+",2) ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchone()
    return res
    
    
def get_path(veh_id):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select X,Y from TRAJECTORIES_0769 WHERE TRAJECTORIES_0769.TRACK_ID="+str(veh_id)+" ORDER BY TIME"
    c.execute(q_string)
    res = c.fetchall()
    path = [(row[0],row[1]) for row in res]
    return path
    
    
    
def get_vehicles_info(time_ts):
    vehicles_info = dict()
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select TRAJECTORIES_0769.TRACK_ID,TRAJECTORIES_0769.TIME,TRAJECTORIES_0769.X,TRAJECTORIES_0769.Y,TRAJECTORIES_0769_EXT.ASSIGNED_SEGMENT \
                     from TRAJECTORIES_0769 LEFT JOIN TRAJECTORIES_0769_EXT ON TRAJECTORIES_0769.TRACK_ID=TRAJECTORIES_0769_EXT.TRACK_ID AND TRAJECTORIES_0769.TIME=TRAJECTORIES_0769_EXT.TIME \
                          WHERE TRAJECTORIES_0769.TIME="+str(time_ts)
    c.execute(q_string)
    res = c.fetchall()
    for row in res:
        if row[0] not in vehicles_info:
            vehicles_info[row[0]] = ((row[2],row[3]), row[4])
    return vehicles_info
    
    
def gate_crossing_times(veh_state):
    entry_gate,exit_gate = veh_state.gates[0],veh_state.gates[1]
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    entry_time,exit_time = None,None
    if entry_gate is not None:
        q_string = "select TIME from GATE_CROSSING_EVENTS WHERE GATE_ID = "+str(entry_gate)+" AND TRACK_ID = "+str(veh_state.id)
        c.execute(q_string)
        res = c.fetchone()
        entry_time = res[0] if res is not None else None
    if exit_gate is not None:
        q_string = "select TIME from GATE_CROSSING_EVENTS WHERE GATE_ID = "+str(exit_gate)+" AND TRACK_ID = "+str(veh_state.id)
        c.execute(q_string)
        res = c.fetchone()
        exit_time = res[0] if res is not None else None
    return (entry_time,exit_time)
    
    

def get_path_gates_direction(agent_track,agent_id):
    path = ['NA','NA']
    if agent_track is not None and len(agent_track) > 1:
        i = 0
        while len(agent_track[i]) == 0:
            i += 1
        entry_region = agent_track[i].replace(' ','').strip(',')
        j = -1
        while len(agent_track[j]) == 0:
            j += -1
        exit_region = agent_track[j].replace(' ','').strip(',')
        ''' some tracks start in the middle '''
        origin_lane_map = {'s_w':1,'w_n':1,'e_s':1,'n_e':1}
        if 'ln_' not in entry_region:
            entry_regions = entry_region.split(',')
            possible_entry_paths = []
            for r in entry_regions:
                r = r.strip()
                if len(r) < 1:
                    brk = 6
                if r[0] == 'l':
                    possible_entry_paths.append(r)
            for p in possible_entry_paths:
                ends_in = p[4]
                if 'ln_'+ends_in in exit_region:
                    if len(p) < 7:
                        lane_num = origin_lane_map[p[-3:]]
                    else:
                        lane_num = -1 if p[6] == 'l' else -2
                    entry_region = 'ln_'+p[2]+'_' + str(lane_num)
                    path[0] = entry_region
                    break
        else:
            entry_regions = entry_region.split(',')
            for e in entry_regions:
                if 'ln_' in e:
                    entry_region = e
        ''' some tracks end in the middle'''
        exit_lane_map = {''}
        if 'ln_' not in exit_region:
            exit_regions = exit_region.split(',')
            possible_exit_paths = []
            for r in exit_regions:
                if len(r) > 1:
                    r = r.strip()
                    if r[0] == 'l':
                        possible_exit_paths.append(r)
            for p in possible_exit_paths:
                starts_in = p[2]
                if 'ln_'+starts_in in entry_region:
                    if len(p) < 7:
                        lane_num = -1
                    else:
                        lane_num = -1 if p[6] == 'l' else -2 
                    exit_region = 'ln_'+p[4]+'_' + str(lane_num)
                    path[1] = exit_region
                    break
        else:
            exit_regions = exit_region.split(',')
            for e in exit_regions:
                if 'ln_' in e:
                    exit_region = e
        path = [entry_region,exit_region]
        if 'ln_' not in entry_region and 'ln_' not in exit_region:
            lane_mapping = {'l_n_s_r':['ln_n_3','ln_s_-2'],
                            'l_n_s_l':['ln_n_2','ln_s_-1'],
                            'l_s_n_r':['ln_s_3','ln_n_-2'],
                            'l_s_n_l':['ln_s_2','ln_n_-1'],
                            'l_e_w_r':['ln_e_3','ln_w_-2'],
                            'l_e_w_l':['ln_e_2','ln_w_-1'],
                            'l_w_e_r':['ln_w_3','ln_e_-2'],
                            'l_w_e_l':['ln_w_2','ln_e_-1'],
                            'rt_exec-turn_w':['ln_w_4','ln_s_-2']}
            for r in entry_region.split(','):
                if r in exit_region.split(','):
                    path = lane_mapping[r] if r[0] == 'l' else ['NA','NA']
                    break
                path = ['NA','NA']
        elif 'ln_' not in entry_region:
            entry_region = 'NA'
            path = [entry_region,exit_region]
        elif 'ln_' not in exit_region:
            exit_region = 'NA'
            path = [entry_region,exit_region]
        
    
    '''return the gates '''
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    gates = [None,None]
    o,d = None,None    
    if path[0] != 'NA' and path[1] != 'NA':
        o,d = path[0][3],path[1][3]
    q_string = "SELECT ENTRY_GATE,EXIT_GATE FROM TRAJECTORY_MOVEMENTS WHERE TRACK_ID = "+str(agent_id)
    c.execute(q_string)
    res = c.fetchall()
    e_g,ex_g = [],[]
    for row in res:
        if row[0] is not None:
            e_g = [int(row[0])]
            for k,v in constants.gate_map.items():
                if int(row[0]) in v:
                    o = k[0]
        if row[1] is not None:
            ex_g = [int(row[1])]
            for k,v in constants.gate_map.items():
                if int(row[1]) in v:
                    d = k[0]
    ''' if only one gate present, try to acquire it from the path information '''
    if e_g is None and path[0] != 'NA':
        _dir = path[0][3]
        _sign = '_entry' if path[0][-2] == '-' else '_exit'
        _key = _dir + _sign
        e_g = constants.gate_map[_key]
    if ex_g is None and path[1] != 'NA':
        _dir = path[1][3]
        _sign = '_entry' if path[1][-2] == '-' else '_exit'
        _key = _dir + _sign
        ex_g = constants.gate_map[_key]
    for _eg in e_g:
        for _exg in ex_g:
            gates[0]=_eg
            gates[1]=_exg  
    
    if o is not None and d is not None:
        direction = 'L_'+o.upper()+'_'+d.upper()
    else:
        direction = None
    return path,gates,direction
def get_traffic_segment_from_gates(gates):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    segment_seq = []
    q_string = "SELECT SEGMENT_SEQ FROM SEGMENT_SEQ_MAP WHERE ENTRY_GATE = "+str(gates[0])+" AND EXIT_GATE = "+str(gates[1])
    c.execute(q_string)
    res = c.fetchall()
    if res is None:
        sys.exit('unknown gate sequence. update segment_seq table')
    for row in res:
        segment_seq.append(ast.literal_eval(row[0]))
    conn.close()
    return segment_seq


def get_l1_action_string(code):
    for k,v in constants.L1_ACTION_CODES.items():
        if v==code:
            return k

def get_l2_action_string(code):
    for k,v in constants.L2_ACTION_CODES.items():
        if v==code:
            return k  
def print_readable(eq):
    readable_eq = []
    if isinstance(eq, str):
        s = eq
        return s[3:6]+'_'+s[6:9]+'_'+get_l1_action_string(int(s[9:11]))+'_'+get_l2_action_string(int(s[11:13]))
    else:
        for s in eq:
            readable_eq.append(s[3:6]+'_'+s[6:9]+'_'+get_l1_action_string(int(s[9:11]))+'_'+get_l2_action_string(int(s[11:13])))
        return readable_eq
    
def query_agent(conflict,subject_path,veh_state):
    vehicles = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    curr_time = float(veh_state.current_time)
    other_agent_gates,other_agent_path = [],[]
    if subject_path == 1:
        other_agent_gates = conflict[4]
        other_agent_path= conflict[3]
    elif subject_path == 2:
        other_agent_gates = conflict[2]
        other_agent_path = conflict[1]
    gates = ast.literal_eval(other_agent_gates)
    path = other_agent_path[1:-1].split(',')
    entry_gate,exit_gate = gates[0],gates[1]
    veh_intersection_exit_time = veh_state.gate_crossing_times[1]
    veh_scene_exit_time = veh_state.entry_exit_time[1]
    if conflict[-1] == 'ON_INTERSECTION':
        end_time = curr_time+constants.RELEV_VEHS_TIME_THRESH if veh_intersection_exit_time is None else veh_intersection_exit_time+constants.RELEV_VEHS_TIME_THRESH
    else:
        end_time = curr_time+constants.RELEV_VEHS_TIME_THRESH if veh_scene_exit_time is None else veh_scene_exit_time
    q_string = "SELECT DISTINCT TRAJECTORIES_0769.TRACK_ID FROM TRAJECTORIES_0769,TRACKS WHERE TRAJECTORIES_0769.TRACK_ID=TRACKS.TRACK_ID AND (TIME BETWEEN "+str(curr_time)+" AND "+str(end_time)+") AND TRACKS.TYPE <> 'Pedestrian' AND TRACKS.TRACK_ID IN (SELECT DISTINCT TRACK_ID FROM TRAJECTORY_MOVEMENTS WHERE TRAFFIC_SEGMENT_SEQ LIKE '%"+path[0][:-1]+"%"+path[1][:-2]+"%' ORDER BY TRACK_ID)"
    c.execute(q_string)
    res = c.fetchall()
    if len(res) < 1:
        print('no relevant agents for query:',q_string)
    elif len(res) > 7:
        ''' too many agents. we can reduce the number by restricting the time threshold without significant impact '''
        q_string = "SELECT DISTINCT TRAJECTORIES_0769.TRACK_ID FROM TRAJECTORIES_0769,TRACKS WHERE TRAJECTORIES_0769.TRACK_ID=TRACKS.TRACK_ID AND TIME = "+str(curr_time)+" AND TRACKS.TYPE <> 'Pedestrian' AND TRACKS.TRACK_ID IN (SELECT DISTINCT TRACK_ID FROM TRAJECTORY_MOVEMENTS WHERE TRAFFIC_SEGMENT_SEQ LIKE '%"+path[0][:-1]+"%"+path[1][:-2]+"%' ORDER BY TRACK_ID)"
        c.execute(q_string)
        res = c.fetchall()
    
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles
        
def is_out_of_view(pos):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from traffic_regions_def where name='view_area'"
    c.execute(q_string)
    res = c.fetchone()
    X = ast.literal_eval(res[4])
    Y = ast.literal_eval(res[5])
    viewport = list(zip(X,Y))
    p = path.Path(viewport)
    in_view = p.contains_points([(pos[0], pos[1])])
    return False if in_view[0] else True
        
''' True if relevant agent r_a_state can be excluded from the relevant vehicle list of vehicle veh_state '''    
def can_exclude(veh_state,r_a_state):
    if r_a_state.dist_to_sv > 150:
        return True
    ra_segment_type = constants.SEGMENT_MAP[r_a_state.current_segment]
    if veh_state.task == 'LEFT_TURN' and ra_segment_type == 'exit-lane' and r_a_state.direction != veh_state.direction:
        return True
    elif veh_state.task == 'LEFT_TURN' and (r_a_state.task=='RIGHT_TURN' and (r_a_state.leading_vehicle is not None and r_a_state.leading_vehicle.current_segment == r_a_state.current_segment) and constants.SEGMENT_MAP[r_a_state.current_segment] == 'right-turn-lane'):
        ''' exclude vehicles that are turning right from oncoming lane to the common lane but are behind a more relevant vehicle '''
        return True
    elif veh_state.task == 'RIGHT_TURN' and (r_a_state.task=='LEFT_TURN' and (r_a_state.leading_vehicle is not None and (r_a_state.leading_vehicle.current_segment == r_a_state.current_segment or constants.SEGMENT_MAP[r_a_state.leading_vehicle.current_segment] == 'prep-left-turn')) and constants.SEGMENT_MAP[r_a_state.current_segment] == 'left-turn-lane'):
        ''' exclude vehicles that are turning left from oncoming lane to the common lane but are behind a more relevant vehicle '''
        return True
    else:
        return False
    
def is_only_leading_relevant(veh_state):
    if veh_state.task == 'RIGHT_TURN' and constants.SEGMENT_MAP[veh_state.current_segment] == 'right-turn-lane' and constants.SEGMENT_MAP[veh_state.leading_vehicle.current_segment] in ['right-turn-lane','prep-right-turn']:
        return True
    else:
        return False



def get_relevant_agents(veh_state):
    relev_agents = []
    if veh_state.signal == 'R' and constants.SEGMENT_MAP[veh_state.current_segment] in ['through-lane-entry','left-turn-lane']:
        ''' only relevant agents is the leading vehicle if present, which will be determined later'''
        return relev_agents
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    path = veh_state.segment_seq
    signal = veh_state.signal
    conflicts = []
    other_agent_paths,other_agent_gates,other_agent_signal = [],[],[]
    subject_path = None
    if veh_state.gates[0] is not None and veh_state.gates[1] is not None:
        gates = veh_state.gates
        if veh_state.gate_crossing_times[1] is not None and veh_state.current_time is not None and veh_state.current_time < veh_state.gate_crossing_times[1]:
            q_string = "SELECT * FROM CONFLICT_POINTS WHERE PATH_1_GATES LIKE '"+str(gates).replace(' ','')+"' OR PATH_2 LIKE '"+str(gates).replace(' ','')+"'"
        else:
            q_string = "SELECT * FROM CONFLICT_POINTS WHERE ((PATH_1_GATES LIKE '"+str(gates).replace(' ','')+"') OR (PATH_2 LIKE '" +str(gates).replace(' ','')+"')) AND  POINT_LOCATION = 'AFTER_INTERSECTION'"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[2] == str(gates).replace(' ',''):
                subject_path = 1
                other_agent_signal = row[9][1:-1].split(',')
                other_agent_path = 'L'+'_'+row[3][4].upper()+'_'+row[3][11].upper()
                other_agents_traffic_light = get_traffic_signal(veh_state.current_time, other_agent_path)
                if other_agents_traffic_light in other_agent_signal:
                    conflicts.append((row,subject_path))
            elif row[4] == str(gates).replace(' ',''):
                subject_path = 2
                other_agent_signal = row[8][1:-1].split(',')
                other_agent_path = 'L'+'_'+row[1][4].upper()+'_'+row[3][11].upper()
                other_agents_traffic_light = get_traffic_signal(veh_state.current_time, other_agent_path)
                if other_agents_traffic_light in other_agent_signal:
                    conflicts.append((row,subject_path))
    elif path[0] != 'NA' and path[1] != 'NA':
        path_string = '['+path[0]+','+path[1]+']'
        if (veh_state.gate_crossing_times[1] is not None and veh_state.current_time is not None and veh_state.current_time < veh_state.gate_crossing_times[1]) or (veh_state.current_segment != veh_state.segment_seq[-1]):
            q_string = "SELECT * FROM CONFLICT_POINTS WHERE (PATH_1 LIKE '"+path_string+"' AND SIGNAL_STATE_PATH_1 LIKE '%"+signal \
                    +"%') OR (PATH_2 LIKE '"+path_string+"' AND SIGNAL_STATE_PATH_2 LIKE '%"+signal+"%')"
        else:
            q_string = "SELECT * FROM CONFLICT_POINTS WHERE ((PATH_1 LIKE '"+path_string+"' AND SIGNAL_STATE_PATH_1 LIKE '%"+signal \
                    +"%') OR (PATH_2 LIKE '"+path_string+"' AND SIGNAL_STATE_PATH_2 LIKE '%"+signal+"%')) AND POINT_LOCATION = 'AFTER_INTERSECTION'"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            if row[1] == path_string:
                subject_path = 1
                other_agent_signal = row[9][1:-1].split(',')
                other_agent_path = 'L'+'_'+row[3][4].upper()+'_'+row[3][11].upper()
                other_agents_traffic_light = get_traffic_signal(veh_state.current_time, other_agent_path)
                if other_agents_traffic_light in other_agent_signal:
                    conflicts.append((row,subject_path))
            elif row[3] == path_string:
                subject_path = 2
                other_agent_signal = row[8][1:-1].split(',')
                other_agent_path = 'L'+'_'+row[1][4].upper()+'_'+row[3][11].upper()
                other_agents_traffic_light = get_traffic_signal(veh_state.current_time, other_agent_path)
                if other_agents_traffic_light in other_agent_signal:
                    conflicts.append((row,subject_path))
            
    else:
        print('cannot find relevant agents for',veh_state.id,veh_state.current_time)
        
    for c in conflicts:
        curr_conflict_agents = query_agent(c[0],c[1],veh_state)
        for c_a in curr_conflict_agents: 
            if c_a not in relev_agents:
                relev_agents.append(c_a)
    return relev_agents
    
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_entry_exit_time(track_id,file_id='769'):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from v_TIMES where track_id="+str(track_id)
    c.execute(q_string)
    res = c.fetchone()
    time_tuple = (float(res[1]),float(res[2]))
    conn.close()
    return time_tuple

    
def get_traffic_signal(time,direction,file_id='769'):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    if direction == 'ALL':
        conn.row_factory = dict_factory
        c = conn.cursor()
        q_string = "SELECT * FROM TRAFFIC_LIGHTS WHERE TIME - "+str(time)+" > 0 AND FILE_ID = "+file_id+" ORDER BY TIME"
        c.execute(q_string)
        res = c.fetchall()
        return res
    else:
        c = conn.cursor()
        signal = None
        q_string = "SELECT MAX(TIME),"+direction+" FROM TRAFFIC_LIGHTS WHERE TIME - "+str(time)+" <= 0 AND FILE_ID = "+file_id+" "
        c.execute(q_string)
        res = c.fetchone()
        signal = res[1]
        conn.close()
        return signal
    
def get_time_to_next_signal(time_ts,direction,curr_signal):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT * FROM TRAFFIC_LIGHTS WHERE TIME - "+str(time_ts)+" > 0 AND "+direction+" <> '"+curr_signal+"' order by time"
    curr = c.execute(q_string)
    res = c.fetchone()
    all_directions = [description[0] for description in curr.description]
    dir_idx = all_directions.index(direction)
    if res is None:
        return (None,None)
    else:
        next_signal = res[dir_idx]
        time_to_change = float(res[-1]) - time_ts
        return (time_to_change,next_signal)
    
    
def get_actions(veh_state):
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    segment = constants.SEGMENT_MAP[veh_state.current_segment]
    if veh_state.leading_vehicle is not None:
        q_string = "SELECT * FROM ACTIONS WHERE SEGMENT = '"+segment+"' AND LEAD_VEHICLE_PRESENT IN ('Y','*') AND TASK IN ('"+veh_state.task+"','*')"
    else:
        q_string = "SELECT * FROM ACTIONS WHERE SEGMENT = '"+segment+"' AND LEAD_VEHICLE_PRESENT IN ('N','*') AND TASK IN ('"+veh_state.task+"','*')"
    if veh_state.merging_vehicle is not None:
        q_string = q_string + " AND MERGING_VEHICLE_PRESENT IN ('Y','*')"
    else:
        q_string = q_string + " AND MERGING_VEHICLE_PRESENT IN ('N','*')"
    if veh_state.relev_pedestrians is not None:
        q_string = q_string + " AND PEDESTRIAN_PRESENT IN ('Y','*')"
    else:
        q_string = q_string + " AND PEDESTRIAN_PRESENT IN ('N','*')"
    
    c.execute(q_string)
    rows = c.fetchall()
    actions = dict() 
    for res in rows:
        if res[3] is not None and (res[3] == veh_state.signal or res[3] == '*'):
            if res[1] not in  actions.items():
                actions[res[1]] = ast.literal_eval(res[2])
    
    
    
    return actions

def region_equivalence(track_region,track_segment):
    if track_region == track_segment or track_region.replace('-','_') == track_segment.replace('-','_'):
        return True
    if track_segment[:2] == 'ln':
        return track_segment[:-2] == track_region[:-2] if track_region[-2] == '-' else track_segment[:-1] == track_region[:-1]
    else:
        if 'int_entry_' in track_segment.replace('-','_'):
            if track_region[0:4] == 'l_'+track_segment[-1]+'_':
                return True
        return False 
    
def has_crossed(segment,veh_state):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    
    ''' for each segment loop and check if the vehicle has not yet crossed it'''
    
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+segment+"' AND REGION_PROPERTY = 'exit_boundary'"
    c.execute(q_string)
    res = c.fetchone()
    if res is None or len(res) < 1:
        sys.exit('exit boundary not found for '+str(segment))
    exit_pos_X = ast.literal_eval(res[4])
    exit_pos_Y = ast.literal_eval(res[5])
    m = (exit_pos_Y[1] - exit_pos_Y[0]) / (exit_pos_X[1] - exit_pos_X[0])
    c = (exit_pos_Y[0] - (m * exit_pos_X[0]))
    
    veh_pos_x, veh_pos_y = veh_state.x,veh_state.y
    if veh_state.gate_crossing_times[0] is not None:
        veh_orig_x, veh_orig_y = veh_state.track[0][1],veh_state.track[0][2]
    else:
        veh_orig_x, veh_orig_y = veh_state.path_origin[0],veh_state.path_origin[1]
    #dist_to_exit_boundary = distance_numpy([exit_pos_X[0],exit_pos_Y[0]], [exit_pos_X[1],exit_pos_Y[1]], [veh_pos_x,veh_pos_y])
    #dist_from_veh_origin_to_exit_boundary = distance_numpy([exit_pos_X[0],exit_pos_Y[0]], [exit_pos_X[1],exit_pos_Y[1]], [veh_orig_x,veh_orig_y])
    res_wrt_origin = veh_orig_y - (m*veh_orig_x) - c
    res_wrt_point = veh_pos_y - (m*veh_pos_x) - c
    
    conn.close()
    return True if np.sign(res_wrt_origin) != np.sign(res_wrt_point) else False

def assign_curent_segment(traffic_region_list,veh_state,simulation=False):
    track_segment = veh_state.segment_seq
    ''' if it is simulation, then we would have to assign a segment to the 
    point not seen in the data. '''
    assignment_from_region_failed = False
    if not simulation and traffic_region_list is not None:
        traffic_region_list = str(traffic_region_list).replace(' ','').strip(',')
        #current_segment = []
        #all_segments = ['int-entr_','execute-turn_','prepare-turn_','rt-stop_','rt-prep_turn_','rt_exec_turn_']
        traffic_region_list = traffic_region_list.split(',')
        for segment in reversed(track_segment):
            for track_region in traffic_region_list:
                if region_equivalence(track_region, segment):
                    return segment
        ''' assigning based on region failed '''
        
        assignment_from_region_failed = True  
    if simulation or assignment_from_region_failed or traffic_region_list is None:
        curr_time = veh_state.current_time
        if simulation:
                ''' get the first segment. This should be set to the correct value for time=0 since we start the simulation
                from the real scene.'''
                prev_segment = None
                try:
                    prev_segment = veh_state.current_segment
                except AttributeError:
                    sys.exit('previous segment is not set (possibly for the initial scene)')
                
                if has_crossed(prev_segment, (veh_state.x,veh_state.y)):
                    return track_segment[track_segment.index(prev_segment)+1] 
                else:
                    return prev_segment
        else:
            if not has_crossed(track_segment[0], veh_state):
                return track_segment[0]
            elif has_crossed(track_segment[-1], veh_state):
                return track_segment[-1]
            else:
                for seg,next_seg in zip(track_segment[:-1],track_segment[1:]):
                    if not has_crossed(next_seg, veh_state) and has_crossed(seg, veh_state):
                        return next_seg
        ''' if still unable to assign, try with gate crossing times'''   
        if veh_state.gate_crossing_times[0] is not None and curr_time < veh_state.gate_crossing_times[0]:
            ''' the vehicle hasn't entered the intersection, so return the first segment.'''
            return track_segment[0]
        elif veh_state.gate_crossing_times[1] is not None and curr_time > veh_state.gate_crossing_times[1]:
            ''' the vehicle has left the intersection, so return the last segment'''
            return track_segment[-1]
        ''' everything failed'''
        return None 
                        
                    

def find_index_in_list(s_sum, dist_from_origin):
    if s_sum > dist_from_origin[-1]:
        return len(dist_from_origin)-1
    idx = None
    for i,x in enumerate(list(zip(dist_from_origin[:-1],dist_from_origin[1:]))):
        if x[0] <= s_sum <= x[1]:
            idx = i
            break
    return idx
    
def insert_baseline_trajectory(l3_actions,f):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select MAX(GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_ID) FROM GENERATED_BASELINE_TRAJECTORY"
    c.execute(q_string)
    res = c.fetchone()
    max_traj_id = int(res[0]) if res[0] is not None else 0
    agent_id = int(f[3:6])
    time_ts = float(f.split('_')[-1].replace(',','.'))
    relev_agent = int(f[6:9])
    l1_action = [k for k,v in constants.L1_ACTION_CODES.items() if v == int(f[9:11])][0]
    l2_action = [k for k,v in constants.L2_ACTION_CODES.items() if v == int(f[11:13])][0]
    i_string_data = (769,agent_id,relev_agent,l1_action,l2_action,time_ts,1)
    #print('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?,?)',i_string_data)
    c.execute("SELECT * FROM GENERATED_TRAJECTORY_INFO WHERE AGENT_ID="+str(i_string_data[1])+" AND RELEV_AGENT_ID="+str(i_string_data[2])+" AND L1_ACTION='"+str(i_string_data[3])+"' AND \
                    L2_ACTION='"+str(i_string_data[4])+"' AND TIME="+str(i_string_data[5]))
    res = c.fetchone()
    if res is not None and len(res) > 0:
        traj_info_id = res[1]
        c.execute('DELETE FROM GENERATED_BASELINE_TRAJECTORY WHERE TRAJECTORY_INFO_ID='+str(traj_info_id))
        conn.commit()
    else:
        c.execute('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?)',i_string_data)
        conn.commit()
        traj_info_id = int(c.lastrowid)
    
    traj_id = max_traj_id+1
    
    traj_dets = l3_actions
    slice_len = min([len(x) for x in traj_dets[0:7]])
    tx,rx,ry,ryaw,rv,ra,rj = traj_dets[0][:slice_len],traj_dets[1][:slice_len],traj_dets[2][:slice_len],traj_dets[3][:slice_len],traj_dets[4][:slice_len],traj_dets[5][:slice_len],traj_dets[6][:slice_len]
    ins_list = list(zip([traj_id]*slice_len,[traj_info_id]*slice_len,[round(x,5) for x in tx],[round(x,5) for x in rx],[round(x,5) for x in ry],[round(x,5) for x in ryaw],[round(x,5) for x in rv],[round(x,5) for x in ra],[round(x,5) for x in rj]))
    i_string = 'INSERT INTO GENERATED_BASELINE_TRAJECTORY VALUES (?,?,?,?,?,?,?,?,?)'
    c.executemany(i_string,ins_list)
    traj_id += 1
    conn.commit()
    conn.close()

def insert_generated_trajectory(l3_actions,f):
    print("inserting trajectory: START")
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select MAX(GENERATED_TRAJECTORY.TRAJECTORY_ID) FROM GENERATED_TRAJECTORY"
    c.execute(q_string)
    res = c.fetchone()
    max_traj_id = int(res[0])
    file_id = f[0:3]
    agent_id = int(f[3:6])
    time_ts = float(f.split('_')[-1].replace(',','.'))
    relev_agent = int(f[6:9])
    l1_action = [k for k,v in constants.L1_ACTION_CODES.items() if v == int(f[9:11])][0]
    l2_action = [k for k,v in constants.L2_ACTION_CODES.items() if v == int(f[11:13])][0]
    traj_len = l3_actions.shape[0] if len(l3_actions) > 0 else 0
    i_string_data = (769,agent_id,relev_agent,l1_action,l2_action,time_ts,traj_len)
    #print('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?,?)',i_string_data)
    c.execute('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?)',i_string_data)
    conn.commit()
    if traj_len != 0:
        traj_info_id = int(c.lastrowid)
        traj_id = max_traj_id+1
        for i in np.arange(traj_len):
            traj_dets = l3_actions[i][0]
            slice_len = min([len(x) for x in traj_dets[0:7]])
            tx,rx,ry,ryaw,rv,ra,rj = traj_dets[0][:slice_len],traj_dets[1][:slice_len],traj_dets[2][:slice_len],traj_dets[3][:slice_len],traj_dets[4][:slice_len],traj_dets[5][:slice_len],traj_dets[6][:slice_len]
            ins_list = list(zip([traj_id]*slice_len,[traj_info_id]*slice_len,[round(x,5) for x in tx],[round(x,5) for x in rx],[round(x,5) for x in ry],[round(x,5) for x in ryaw],[round(x,5) for x in rv],[round(x,5) for x in ra],[round(x,5) for x in rj]))
            i_string = 'INSERT INTO GENERATED_TRAJECTORY VALUES (?,?,?,?,?,?,?,?,?)'
            c.executemany(i_string,ins_list)
            traj_id += 1
    conn.commit()
    conn.close()
    print("inserting trajectory: DONE")


def solve_quadratic(a,b,c):
    return (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a),(-b - math.sqrt(b**2 - 4*a*c)) / (2 * a)        
    
def generate_baseline_velocity(time_tx,v_s,a_s,target_vel,max_acc,max_jerk,acc):
    vels = []
    v,a = v_s,a_s
    dt = constants.LP_FREQ
    s_sum = 0
    for i,t in enumerate(time_tx):
        s = v*dt + (0.5*a*dt**2)
        s_sum += s
        v = max(v + a * dt, 0)
        if (acc and a < max_acc) or (not acc and a > max_acc):
            a = a + (max_jerk*dt)
        else:
            a = max_acc
        if (acc and v > target_vel):
            a = a - (max_jerk*dt)
        vels.append(v)
    return vels
    
def get_baseline_trajectory(agent_id,relev_agent_id,l1_action,l2_action,curr_time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select * from GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.time BETWEEN "+str(curr_time)+" AND "+str(curr_time+constants.PLAN_FREQ)+" AND GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID in (select GENERATED_TRAJECTORY_INFO.TRAJ_ID FROM GENERATED_TRAJECTORY_INFO WHERE GENERATED_TRAJECTORY_INFO.AGENT_ID="+str(agent_id)+" AND GENERATED_TRAJECTORY_INFO.RELEV_AGENT_ID="+str(relev_agent_id)+" AND GENERATED_TRAJECTORY_INFO.L1_ACTION='"+l1_action+"' AND GENERATED_TRAJECTORY_INFO.L2_ACTION='"+l2_action+"' AND GENERATED_TRAJECTORY_INFO.TIME="+str(curr_time)+") order by GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID,GENERATED_BASELINE_TRAJECTORY.time"
    c.execute(q_string)
    res = c.fetchall()
    return [list(row) for row in res]

def calc_traj_diff(traj1,traj2):
    slice_len = min(len(traj1),len(traj2))
    _t1,_t2 = [(x[1],x[2]) for x in traj1[:slice_len]], [(x[1],x[2]) for x in traj2[:slice_len]]
    residual = sum([math.hypot(x[1][0]-x[0][0], x[1][1]-x[0][1]) for x in zip(_t1,_t2)])
    return residual

def split_into_strict_order(pt_l):
    split_l,inc,temp_l = [],None,[]
    for i in np.arange(1,len(pt_l)):
        if inc is None:
            inc = True if pt_l[i][0] > pt_l[i-1][0] else False
            temp_l = [pt_l[i-1],pt_l[i]]
            continue
        elif (inc and pt_l[i][0] > pt_l[i-1][0]) or (not inc and pt_l[i][0] <= pt_l[i-1][0]):
            if len(temp_l) >= constants.MAX_FP:
                split_l.append(temp_l)
                temp_l = [pt_l[i-1]]
            temp_l.append(pt_l[i])
        else:
            split_l.append(temp_l)
            temp_l = [pt_l[i-1],pt_l[i]]
            inc = True if pt_l[i][0] > pt_l[i-1][0] else False
    split_l.append(temp_l)
    return split_l
    
def generate_baseline_trajectory(time,path,v_s,a_s,max_acc,max_jerk,v_g,dt,acc):
    dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(path[:-1],path[1:]))]
    dist_from_origin = [sum(dist_from_origin[:i]) for i in np.arange(1,len(dist_from_origin))]
    dist,vels,accs = [],[],[]
    new_path = [(path[0][0],path[0][1])]
    v,a = v_s,a_s
    #time = np.arange(dt,time[-1],dt)
    s_sum = 0
    new_time = []
    for i in time:
        s = max(v*dt + (0.5*a*dt**2), 0)
        s_sum += s
        v = max(v + a * dt, 0)
        if (acc and a < max_acc) or (not acc and a > max_acc):
            a = a + (max_jerk*dt)
        else:
            a = max_acc
        if (acc and v > v_g):
            a = a - (max_jerk*dt)
        if (not acc and v <= v_g):
            a = a + (max_jerk)*dt if abs(a) > abs(max_jerk)*dt else 0
        path_idx = find_index_in_list(s_sum, dist_from_origin)
        if path_idx is None:
            raise IndexError("path_idx is None")
        overflow = dist_from_origin[path_idx+1] - s_sum if path_idx+1 < len(dist_from_origin) else s_sum-dist_from_origin[-1]
        point = path[path_idx]
        r = overflow/math.hypot(path[path_idx+1][0]-path[path_idx][0], path[path_idx+1][1]-path[path_idx][1])
        point_x = path[path_idx][0] + r*(path[path_idx+1][0] - path[path_idx][0])
        point_y = path[path_idx][1] + r*(path[path_idx+1][1] - path[path_idx][1])
        point = (point_x,point_y)
        new_path.append(point)
        dist.append(s)
        vels.append(v)
        accs.append(a)
        new_time.append(i)
    dist = [sum(dist[:i]) for i in np.arange(1,len(dist))]
    '''
    plt.plot(new_time,vels,'g',new_time,accs,'r')
    plt.show()
    plt.plot([x[0] for x in new_path],[x[1] for x in new_path])
    plt.show()
    '''
    
    if not acc and vels[-1] == 0 and len(vels) < len(time):
        ''' pad the trajectory'''
        vels = vels + [0]*(len(time)-len(vels))
    return time,np.asarray(vels),new_path
     
        
def get_exit_boundary(segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+segment+"' and REGION_PROPERTY = 'exit_boundary'"
    c.execute(q_string)
    res = c.fetchone()
    if res is None:
        sys.exit("cannot find exit_boundary for"+segment)
    exit_pos_X = ast.literal_eval(res[4])
    exit_pos_Y = ast.literal_eval(res[5])
    return[exit_pos_X,exit_pos_Y]
    

def get_trajectories_in_db():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "SELECT * FROM GENERATED_TRAJECTORY_INFO"
    c.execute(q_string)
    res = c.fetchall()
    dir = constants.L3_ACTION_CACHE
    traj_ct,ct = 0,0
    N = len(res)
    trajs_in_db = []
    for row in res:
        file_id = str(row[0])
        traj_info_id = int(row[1])
        agent_id = str(row[2]).zfill(3)
        relev_agent_id = str(row[3]).zfill(3)
        l1_action_code = str(constants.L1_ACTION_CODES[row[4]]).zfill(2)
        l2_action_code = str(constants.L2_ACTION_CODES[row[5]]).zfill(2)
        time_ts = str(float(row[6]))
        traj_len = int(row[7])
        file_str = file_id+agent_id+relev_agent_id+l1_action_code+l2_action_code+'_'+str(time_ts).replace('.',',')
        trajs_in_db.append(file_str)
    return trajs_in_db

def get_baseline_trajectories_in_db():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select * from GENERATED_TRAJECTORY_INFO WHERE GENERATED_TRAJECTORY_INFO.TRAJ_ID IN (SELECT DISTINCT GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID FROM GENERATED_BASELINE_TRAJECTORY)"
    c.execute(q_string)
    res = c.fetchall()
    trajs_in_db = []
    for row in res:
        file_id = str(row[0])
        agent_id = str(row[2]).zfill(3)
        relev_agent_id = str(row[3]).zfill(3)
        l1_action_code = str(constants.L1_ACTION_CODES[row[4]]).zfill(2)
        l2_action_code = str(constants.L2_ACTION_CODES[row[5]]).zfill(2)
        time_ts = str(float(row[6]))
        file_str = file_id+agent_id+relev_agent_id+l1_action_code+l2_action_code+'_'+str(time_ts).replace('.',',')
        trajs_in_db.append(file_str)
    return trajs_in_db
    


    
def generate_trajectory_from_vel_profile(time,ref_path,vel_profile):
    dist_from_origin = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(ref_path[:-1],ref_path[1:]))]
    dist_from_origin = [sum(dist_from_origin[:i]) for i in np.arange(1,len(dist_from_origin))]
    new_path = [(ref_path[0][0],ref_path[0][1])]
    s_sum = 0
    for i,t in enumerate(time):
        s = vel_profile[i]*constants.LP_FREQ
        s_sum += s
        path_idx = find_index_in_list(s_sum, dist_from_origin)
        if path_idx is None:
            path_idx = len(dist_from_origin)-2
        if (path_idx+1) < len(dist_from_origin):
            overflow = dist_from_origin[path_idx+1] - s_sum
            point = ref_path[path_idx]
            r = overflow/math.hypot(ref_path[path_idx+1][0]-ref_path[path_idx][0], ref_path[path_idx+1][1]-ref_path[path_idx][1])
            point_x = ref_path[path_idx][0] + r*(ref_path[path_idx+1][0] - ref_path[path_idx][0])
            point_y = ref_path[path_idx][1] + r*(ref_path[path_idx+1][1] - ref_path[path_idx][1])
        else:
            overflow = s_sum - dist_from_origin[-1]
            point = ref_path[path_idx]
            r = overflow/math.hypot(ref_path[path_idx][0]-ref_path[path_idx-1][0], ref_path[path_idx][1]-ref_path[path_idx-1][1])
            point_x = ref_path[path_idx][0] + r*(ref_path[path_idx][0] - ref_path[path_idx-1][0])
            point_y = ref_path[path_idx][1] + r*(ref_path[path_idx][1] - ref_path[path_idx-1][1])
        point = (point_x,point_y)
        new_path.append(point)
    dist_from_origin_newpath = [0] + [math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in list(zip(new_path[:-1],new_path[1:]))]
    dist_from_origin_newpath = [sum(dist_from_origin_newpath[:i]) for i in np.arange(1,len(dist_from_origin_newpath))]
    
    return new_path
   
def get_relevant_crosswalks(veh_state):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    if isinstance(veh_state, VehicleState) or isinstance(veh_state, planning_objects.VehicleState):
        q_string = "select * from RELEVANT_CROSSWALK_MAP WHERE RELEVANT_CROSSWALK_MAP.VEH_PATH='"+veh_state.direction+"'"
    elif isinstance(veh_state, int):
        veh_id = veh_state
        q_string = "SELECT * FROM RELEVANT_CROSSWALK_MAP WHERE VEH_PATH = (select DISTINCT 'L_' || UPPER(SUBSTR(TRAFFIC_SEGMENT_SEQ,6,1)) || '_' ||UPPER(SUBSTR(TRAFFIC_SEGMENT_SEQ,LENGTH(TRAFFIC_SEGMENT_SEQ)-5,1)) FROM TRAJECTORY_MOVEMENTS  WHERE TRACK_ID="+str(veh_id)+")"
    c.execute(q_string)
    res = c.fetchone()
    if res is None:
        return None
    else:
        relev_crosswalks = (ast.literal_eval(res[1]), ast.literal_eval(res[2])) 
        return relev_crosswalks
    

def unreadable(act_str):
    tokens = act_str.split('|')
    assert(len(tokens)==4)
    l1_action = str(constants.L1_ACTION_CODES[tokens[2]]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[tokens[3]]).zfill(2)
    agent = str(tokens[0]).zfill(3)
    relev_agent = str(tokens[1]).zfill(3)
    unreadable_str = '769'+agent+relev_agent+l1_action+l2_action
    return unreadable_str    

def setup_pedestrian_info(curr_time):
    pedestrian_list = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from TRAFFIC_REGIONS_DEF WHERE TRAFFIC_REGIONS_DEF.REGION_PROPERTY='gate_line'"
    c.execute(q_string)
    res = c.fetchall()
    gate_lines = {int(row[0]):[ast.literal_eval(row[4]), ast.literal_eval(row[5])] for row in res}
    q_string = "select * from RELEVANT_CROSSWALK_MAP"
    c.execute(q_string)
    res = c.fetchall()
    all_relev_crosswalks = []
    for row in res:
        x_walk = ast.literal_eval(row[1])
        for x in x_walk:
            if x not in all_relev_crosswalks:
                all_relev_crosswalks.append(x)
    q_string = "SELECT TRAJECTORY_MOVEMENTS.TRACK_ID,GATES_PASSED,entry_time,exit_time,TRAJECTORIES_0769.* FROM v_times,TRAJECTORY_MOVEMENTS,TRAJECTORIES_0769 WHERE v_times.track_id=TRAJECTORY_MOVEMENTS.track_id and TRAJECTORY_MOVEMENTS.TYPE='Pedestrian' and gates_passed is not NULL and entry_time<="+str(curr_time)+" and exit_time>="+str(curr_time)+" and TRAJECTORIES_0769.TRACK_ID = TRAJECTORY_MOVEMENTS.track_id AND TRAJECTORIES_0769.TIME="+str(curr_time)
    c.execute(q_string)
    res = c.fetchall()
    for row in res:
        ped_state = PedestrianState(row[0],curr_time)
        gates_passed = list(ast.literal_eval(row[1]))
        ped_state.set_gates_passed(gates_passed)
        time_tup = (row[2],row[3])
        ped_state.set_scene_entry_exit_times(time_tup)
        ped_state.set_speed(row[7])
        ped_state.set_x(row[5])
        ped_state.set_y(row[6])
        ped_state.set_yaw(row[11])
        if len(gates_passed) > 1:
            q_string_2 = "select * from GATE_CROSSING_EVENTS WHERE GATE_CROSSING_EVENTS.TRACK_ID="+str(ped_state.p_id)+" AND GATE_CROSSING_EVENTS.GATE_ID IN "+str(tuple(gates_passed))
        else:
            q_string_2 = "select * from GATE_CROSSING_EVENTS WHERE GATE_CROSSING_EVENTS.TRACK_ID="+str(ped_state.p_id)+" AND GATE_CROSSING_EVENTS.GATE_ID = "+gates_passed[0]
        c.execute(q_string_2)
        gate_passing_info = c.fetchall()
        gate_passing_info = {int(g[1]):g for g in gate_passing_info}
        crosswalk_data = dict()
        for relev_xwalk in all_relev_crosswalks:
            r_gate_1,r_gate_2 = int(relev_xwalk.split('_')[1]), int(relev_xwalk.split('_')[2])
            if r_gate_1 in gates_passed or r_gate_2 in gates_passed:
                crosswalk_data[relev_xwalk] = dict()
                if r_gate_1 in gates_passed and r_gate_2 in gates_passed:
                    traversal_order = [x[0] for x in sorted([(r_gate_1,gate_passing_info[r_gate_1][6]), (r_gate_2,gate_passing_info[r_gate_2][6])], key=lambda tup: tup[1])]
                    if curr_time < min(gate_passing_info[r_gate_1][6],gate_passing_info[r_gate_2][6]):
                        loc = constants.BEFORE_CROSSWALK
                    elif curr_time > max(gate_passing_info[r_gate_1][6],gate_passing_info[r_gate_2][6]):
                        loc = constants.AFTER_CROSSWALK
                    else:
                        loc = constants.ON_CROSSWALK
                    dist_to_entry = math.hypot(ped_state.x-statistics.mean(gate_lines[traversal_order[0]][0]), ped_state.y-statistics.mean(gate_lines[traversal_order[0]][1]))
                    dist_to_exit = math.hypot(ped_state.x-statistics.mean(gate_lines[traversal_order[1]][0]), ped_state.y-statistics.mean(gate_lines[traversal_order[1]][1]))
                else:
                    ''' track likely passes by the side of the gate '''
                    if r_gate_1 in gates_passed:
                        gate_angle = math.atan2(gate_lines[r_gate_1][1][1]-gate_lines[r_gate_1][1][0], gate_lines[r_gate_1][0][1]-gate_lines[r_gate_1][0][0])
                        gate_angle = gate_angle if gate_angle >= 0 else (2*PI)-abs(gate_angle)
                        entry_angle = gate_angle + PI_BY_2
                        passing_yaw = get_pedestrian_track(ped_state,gate_passing_info[r_gate_1][6])[7]
                        if abs(entry_angle%(2*PI) - passing_yaw%(2*PI)) < PI_BY_2:
                            traversal_order = [r_gate_1,r_gate_2]
                        else:
                            traversal_order = [r_gate_2,r_gate_1]
                    else:
                        gate_angle = math.atan2(gate_lines[r_gate_2][0][1]-gate_lines[r_gate_2][0][0], gate_lines[r_gate_2][1][1]-gate_lines[r_gate_2][1][0])
                        gate_angle = gate_angle if gate_angle >= 0 else (2*PI)-abs(gate_angle)
                        entry_angle = gate_angle + PI_BY_2
                        passing_yaw = get_pedestrian_track(ped_state,gate_passing_info[r_gate_1][6])[7]
                        if abs(entry_angle%(2*PI) - passing_yaw%(2*PI)) < PI_BY_2:
                            traversal_order = [r_gate_2,r_gate_1]
                        else:
                            traversal_order = [r_gate_1,r_gate_2]
                    dist_to_entry = math.hypot(ped_state.x-statistics.mean(gate_lines[traversal_order[0]][0]), ped_state.y-statistics.mean(gate_lines[traversal_order[0]][1]))
                    dist_to_exit = math.hypot(ped_state.x-statistics.mean(gate_lines[traversal_order[1]][0]), ped_state.y-statistics.mean(gate_lines[traversal_order[1]][1]))
                    gl_1 = gate_lines[r_gate_1]
                    gl_2 = gate_lines[r_gate_2]
                    l_1, l_2 = np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_1[0][1], gl_1[1][1]]), np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_2[0][0], gl_2[1][0]]) 
                    angle_wrt_boundary_1 = angle_between_lines(l_1,l_2)
                    l_1, l_2 = np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_1[0][0], gl_1[1][0]]), np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_2[0][1], gl_2[1][1]])
                    angle_wrt_boundary_2 = angle_between_lines(l_1,l_2)
                    if angle_wrt_boundary_1 > PI_BY_2 and angle_wrt_boundary_2 > PI_BY_2:
                        loc = constants.ON_CROSSWALK
                    else:
                        if dist_to_entry <= dist_to_exit:
                            loc = constants.BEFORE_CROSSWALK
                        else:
                            loc = constants.AFTER_CROSSWALK
                crosswalk_data[relev_xwalk]['location'] = loc
                crosswalk_data[relev_xwalk]['dist_to_entry'] = dist_to_entry
                crosswalk_data[relev_xwalk]['dist_to_exit'] = dist_to_exit
                if relev_xwalk in constants.CROSSWALK_SIGNAL_MAP:
                    signal = get_traffic_signal(curr_time, constants.CROSSWALK_SIGNAL_MAP[relev_xwalk])
                    next_change = get_time_to_next_signal(curr_time, constants.CROSSWALK_SIGNAL_MAP[relev_xwalk], signal)
                    crosswalk_data[relev_xwalk]['signal'] = signal
                    crosswalk_data[relev_xwalk]['next_change'] = next_change
                crosswalk_data[relev_xwalk]['traversal_order'] = traversal_order
                
            else:
                continue
        ped_state.set_crosswalks(crosswalk_data)
        pedestrian_list.append(ped_state)
        '''
        dist_to_gates,gate_passing_times,signal_states = dict(), dict(), dict()
        for gates in gates_passed:
            if gates in gate_lines:
                dist_to_gate = math.hypot(ped_state.x-statistics.mean(gate_lines[gates][0]), ped_state.y-statistics.mean(gate_lines[gates][1]))
                dist_to_gates[gates] = dist_to_gate
            if gates in gate_passing_info:
                gate_passing_times[gates] = gate_passing_info[gates][6]
        for k,v in constants.CROSSWALK_SIGNAL_MAP.items():
            gate_ids = [int(k.split('_')[1]), int(k.split('_')[2])]
            if gate_ids[0] in gates_passed or gate_ids[1] in gates_passed:
                signal_states[k] = get_traffic_signal(curr_time, v)
                time_to_next_change = get_time_to_next_signal(curr_time, v, signal_states[k])
                signal_states[k] = (signal_states[k], time_to_next_change[0], time_to_next_change[1])
        ped_state.set_signal(signal_states)           
        ped_state.set_dist_to_gates(dist_to_gates)
        ped_state.set_gate_passing_times(gate_passing_times)
        location_info = dict()
        for gate_pairs in constants.CROSSWALK_GATES:
            if gate_pairs[0] in gates_passed or gate_pairs[1] in gates_passed:
                sorted_pairs = sorted(gate_pairs)
                path_key = "P"+"_"+str(sorted_pairs[0])+"_"+str(sorted_pairs[1])
                loc = None
                if (gate_pairs[0] in gate_passing_times and gate_pairs[1] not in gate_passing_times) or (gate_pairs[1] in gate_passing_times and gate_pairs[0] not in gate_passing_times):
                    
                    gl_1 = gate_lines[gate_pairs[0]]
                    gl_2 = gate_lines[gate_pairs[1]]
                    l_1, l_2 = np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_1[0][1], gl_1[1][1]]), np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_2[0][0], gl_2[1][0]]) 
                    angle_wrt_boundary_1 = angle_between_lines(l_1,l_2)
                    l_1, l_2 = np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_1[0][0], gl_1[1][0]]), np.asarray([ped_state.x, ped_state.y])-np.asarray([gl_2[0][1], gl_2[1][1]])
                    angle_wrt_boundary_2 = angle_between_lines(l_1,l_2)
                    if angle_wrt_boundary_1 > PI_BY_2 and angle_wrt_boundary_2 > PI_BY_2:
                        f=1
                    else:
                        f=1
                else:
                    if gate_pairs[0] in gate_passing_times:
                        if gate_pairs[1] in gate_passing_times:
                            if curr_time < min(gate_passing_times[gate_pairs[0]],gate_passing_times[gate_pairs[1]]):
                                loc = constants.BEFORE_CROSSWALK
                            elif curr_time > max(gate_passing_times[gate_pairs[0]],gate_passing_times[gate_pairs[1]]):
                                loc = constants.AFTER_CROSSWALK
                            else:
                                loc = constants.ON_CROSSWALK
                
                location_info[path_key] = loc
        ped_state.set_location_info(location_info)
        '''
    return pedestrian_list
                
                        
def get_relevant_pedestrians(veh_state, pedestrian_info):
    relev_pedestrians = []
    if veh_state.relev_crosswalks is None or constants.SEGMENT_MAP[veh_state.current_segment] == 'exit-lane':
        return None
    else:
        xwalks,near_gates = veh_state.relev_crosswalks[0], veh_state.relev_crosswalks[1] 
        for xwalk in xwalks:
            for ped_state in pedestrian_info:
                if xwalk in ped_state.crosswalks:
                    if ped_state.crosswalks[xwalk]['location'] == constants.ON_CROSSWALK:
                        relev_pedestrians.append(ped_state)
                    elif ped_state.crosswalks[xwalk]['location'] == constants.BEFORE_CROSSWALK and ped_state.crosswalks[xwalk]['dist_to_entry'] < constants.PEDESTRIAN_CROSSWALK_DIST_THRESH:
                        relev_pedestrians.append(ped_state)
    return relev_pedestrians if len(relev_pedestrians) > 0 else None
                        
            
                               
            
        
        
                
        
          

def get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,curr_time):
    r_a_current_segment = assign_curent_segment(r_a_track_region,r_a_state,False)
    if hasattr(r_a_state, 'entry_exit_time') and r_a_current_segment is None:
        entry_exit_time = r_a_state.entry_exit_time
        if curr_time < entry_exit_time[0]:
            r_a_current_segment = r_a_track_segment_seq[0]
        elif curr_time > entry_exit_time[1]:
            r_a_current_segment = r_a_track_segment_seq[-1]
    if r_a_current_segment is None:
        print(r_a_track_region,r_a_track_segment_seq,r_a_state.id,curr_time)
        sys.exit('no current segment found for relev agent')
    return r_a_current_segment


def get_viewport():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from traffic_regions_def where name='view_area'"
    c.execute(q_string)
    res = c.fetchone()
    X = ast.literal_eval(res[4])
    Y = ast.literal_eval(res[5])
    return [X,Y]

def clip_trajectory_to_viewport(res):
    if len(res) == 9:
        time, x, y, yaw, v, a, j, T, plan_type = res
    else:
        time, x, y, yaw, v, a, j = res
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from traffic_regions_def where name='view_area'"
    c.execute(q_string)
    res = c.fetchone()
    X = ast.literal_eval(res[4])
    Y = ast.literal_eval(res[5])
    viewport = list(zip(X,Y))
    p = path.Path(viewport)
    clip_idx = None
    for i in np.arange(len(time)-1,-1,-1):
        in_view = p.contains_points([(x[i], y[i])])
        if in_view[0]:
            clip_idx = i
            if clip_idx < len(time)-1:
                brk=1
            break
    if clip_idx is None:
        return None
    else:
        res = np.array([time[:clip_idx+1], x[:clip_idx+1], y[:clip_idx+1], yaw[:clip_idx+1], v[:clip_idx+1], a[:clip_idx+1], j[:clip_idx+1]])
        return res
            
            
def get_agents_for_task(task_str):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORY_MOVEMENTS,SEGMENT_SEQ_MAP WHERE TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ=SEGMENT_SEQ_MAP.SEGMENT_SEQ AND SEGMENT_SEQ_MAP.DIRECTION='"+task_str+"' ORDER BY TRACK_ID"
    c.execute(q_string)
    res = c.fetchall()       
    agents = [int(x[0]) for x in res]
    return agents
    #return [149]

''' this function interpolates track information only for real trajectories '''
def interpolate_track_info(veh_state,forward,backward,partial_track=None):
    if partial_track is not None and len(partial_track)>0:
        track_info = partial_track
    else:
        track_info = [None]*9
    veh_id,curr_time = veh_state.id,veh_state.current_time
    track_info[0],track_info[6] = veh_id,curr_time
    veh_entry_segment, veh_exit_segment = veh_state.segment_seq[0],veh_state.segment_seq[-1]
    if not hasattr(veh_state, 'track'):
        q_string = "select TIME,TRAFFIC_REGIONS,X,Y,SPEED from trajectories_0769 where track_id="+str(veh_id)+" order by time"
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        c.execute(q_string)
        res = c.fetchall()
        traffic_regions = []
        for r in res:
            traffic_regions.append((float(r[0]),r[1]))
        veh_entry_speed, veh_exit_speed = res[0][4],res[-1][4]
        conn.close()
    else:
        for r in veh_state.track:
            traffic_regions = [(float(x[6]),x[8]) for x in veh_state.track]
        veh_entry_speed, veh_exit_speed = veh_state.track[0][3],veh_state.track[-1][3]
    for_idx,back_idx = None,None
    if not forward and not backward:
        ''' interpolate in the middle'''
        idx = [x[0] for x in traffic_regions].index(curr_time)
        for i in np.arange(idx,len(traffic_regions)):
            if traffic_regions[i][1] is not None and len(traffic_regions[i][1]) > 1:
                for_idx = i
                break
        for j in np.arange(idx,-1,-1):
            if traffic_regions[j][1] is not None and len(traffic_regions[j][1]) > 1:
                back_idx = j
                break
        if for_idx is None:
            ''' the missing entry is the last one '''
            track_info[8] = veh_exit_segment
        elif back_idx is None:
            ''' the missing entry is the first one'''
            track_info[8] = veh_entry_segment
        else:
            if abs(idx - back_idx) < abs(for_idx - idx):
                ''' missing idx is closer to a previously assigned value in the past'''
                track_info[8] = traffic_regions[back_idx][1]
            else:
                track_info[8] = traffic_regions[for_idx][1]
    elif forward:
        ''' extrapolate forward in time '''
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+veh_exit_segment+"' AND REGION_PROPERTY = 'center_line'"
        c.execute(q_string)
        res = c.fetchone()
        if res is None:
            sys.exit("cannot find centerline for"+veh_exit_segment)
        exit_pos_X = ast.literal_eval(res[4])
        exit_pos_Y = ast.literal_eval(res[5])
        angle_of_centerline = math.atan2(exit_pos_Y[1]-exit_pos_Y[0],exit_pos_X[1]-exit_pos_X[0])
        proj_pos_X = exit_pos_X[1] + veh_entry_speed * math.cos(angle_of_centerline) * abs(veh_state.entry_exit_time[1] - veh_state.current_time)
        proj_pos_Y = exit_pos_Y[1] + veh_entry_speed * math.sin(angle_of_centerline) * abs(veh_state.entry_exit_time[1] - veh_state.current_time)
        track_info[8] = veh_state.segment_seq[-1]
        track_info[1] = proj_pos_X
        track_info[2] = proj_pos_Y
        track_info[3] = veh_entry_speed
        track_info[4] = 0
        track_info[5] = 0
        track_info[7] = angle_of_centerline if angle_of_centerline > 0 else 2 * math.pi + angle_of_centerline 
        conn.close()
    elif backward:
        ''' extrapolate backward in time '''
        idx = 0
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+veh_entry_segment+"' AND REGION_PROPERTY = 'center_line'"
        c.execute(q_string)
        res = c.fetchone()
        entry_pos_X = ast.literal_eval(res[4])
        entry_pos_Y = ast.literal_eval(res[5])
        angle_of_centerline = math.atan2(entry_pos_Y[0]-entry_pos_Y[1],entry_pos_X[0]-entry_pos_X[1])
        proj_pos_X = entry_pos_X[0] + veh_entry_speed * math.cos(angle_of_centerline) * abs(veh_state.entry_exit_time[0] - veh_state.current_time)
        proj_pos_Y = entry_pos_Y[0] + veh_entry_speed * math.sin(angle_of_centerline) * abs(veh_state.entry_exit_time[0] - veh_state.current_time)
        track_info[8] = veh_state.segment_seq[0]
        track_info[1] = proj_pos_X
        track_info[2] = proj_pos_Y
        track_info[3] = veh_entry_speed
        track_info[4] = 0
        track_info[5] = 0
        track_info[7] = math.pi + (angle_of_centerline if angle_of_centerline > 0 else 2 * math.pi + angle_of_centerline)
        conn.close()
    return track_info    
    
def guess_track_info(veh_state,partial_track=None):
    veh_id,curr_time = veh_state.id,veh_state.current_time
    curr_time = float(curr_time)
    entry_time,exit_time = veh_state.entry_exit_time[0],veh_state.entry_exit_time[1]
    veh_state.set_entry_exit_time((entry_time,exit_time))
    if entry_time <= curr_time <= exit_time:
        ''' need to interpolate '''
        track_info = interpolate_track_info(veh_state,False,False,partial_track)
    elif curr_time > exit_time:
        ''' need to extrapolate forward in time'''
        track_info = interpolate_track_info(veh_state,True,False,partial_track)
    elif curr_time < entry_time:
        ''' need to extrapolate backward in time'''
        track_info = interpolate_track_info(veh_state,False,True,partial_track)
    return np.asarray(track_info)
    

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

def get_n_w_vehicles(time):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TRACK_ID FROM TRAJECTORIES_0769 WHERE TIME = "+str(time)+" AND (TRAFFIC_REGIONS LIKE '%l_n_w%')"
    c.execute(q_string)
    res = c.fetchall()
    vehicles = []
    for row in res:
        vehicles.append(row[0])
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
                        

def dist_to_line(A,B,pt):
    ''' line is from A to B '''
    d=((pt[0]-A[0])*(B[1]-A[1]))-((pt[1]-A[1])*(B[0]-A[0]))
    return d  
    

