'''
Created on Jan 16, 2020

@author: Atrisha
'''
import sqlite3
import math
import numpy as np
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
import matplotlib.pyplot as plt
import ast
import sys
import constants
from jsonschema.exceptions import relevance
import motion_planner

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

def get_centerline(lane_segment):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+lane_segment+"' and REGION_PROPERTY='center_line' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    center_coordinates = None
    for row in res:
        x_coords = ast.literal_eval(row[0])
        y_coords = ast.literal_eval(row[1])
        center_coordinates = list(zip(x_coords,y_coords))
    conn.close()
    return center_coordinates


def dist_along_yaw(pt1,pt2,yaw,pos):
    ''' r*cos(yaw-slope + 90) = d '''
    d = abs(distance_numpy([pt1[0],pt1[1]], [pt2[0],pt2[1]], [pos[0],pos[1]]))
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    r = d / (math.cos(yaw - slope + (.5*math.pi)))
    return abs(r)

def kph_to_mps(kph):
    return kph/3.6

def fresnet_to_map(o_x,o_y,X,Y,centerline_angle):
    theta = abs(centerline_angle)
    M_X,M_Y = [],[]
    for x,y in zip(X,Y):
        m_x = o_x + x*math.cos(theta)
        point_angle_with_fresnet = math.atan2(y, x)
        _a = math.hypot(x, y)
        m_y = o_y - (math.hypot(x, y) * math.sin(theta-point_angle_with_fresnet))
        M_X.append(m_x)
        M_Y.append(m_y)
    return M_X,M_Y
        
        

def split_in_n(pt1,pt2,N):
    step = (pt2[0] - pt1[0])/N
    x_coords = [pt1[0] + (step*i) for i in np.arange(N)]
    x_coords = x_coords + [pt2[0]]
    step = (pt2[1] - pt1[1])/N
    y_coords = [pt1[1] + (step*i) for i in np.arange(N)]
    y_coords = y_coords + [pt2[1]]
    return list(zip(x_coords,y_coords))

def construct_state_grid(pt1,pt2,N,tol):
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    slope_comp = (math.pi/2) - slope
    central_coords = split_in_n(pt1,pt2,N)
    grid = [central_coords]
    for r in tol:
        grid.append([(x[0]-(r*math.cos(slope_comp)),x[1]+(r*math.sin(slope_comp))) for x in central_coords])
    return grid
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


def get_track(veh_state):
    agent_id = veh_state.id
    curr_time = veh_state.current_time
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    if curr_time is not None:
        q_string = "select * from trajectories_0769 where track_id="+str(agent_id)+" and time="+str(curr_time)+" order by time"
    else:
        q_string = "select * from trajectories_0769 where track_id="+str(agent_id)+" order by time"
    c.execute(q_string)
    res = c.fetchall()
    l = []
    for row in res:
        l.append(row)
    conn.close()
    return np.asarray(l)

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
    segment_seq = None
    q_string = "SELECT SEGMENT_SEQ FROM SEGMENT_SEQ_MAP WHERE ENTRY_GATE = "+str(gates[0])+" AND EXIT_GATE = "+str(gates[1])
    c.execute(q_string)
    res = c.fetchone()
    if res is None:
        print(q_string)
        sys.exit('unknown gate sequence. update segment_seq table')
    segment_seq = ast.literal_eval(res[0])
    conn.close()
    return segment_seq
    
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
    q_string = "SELECT DISTINCT TRAJECTORIES_0769.TRACK_ID FROM TRAJECTORIES_0769,TRACKS WHERE TRAJECTORIES_0769.TRACK_ID=TRACKS.TRACK_ID AND (TIME BETWEEN "+str(curr_time-constants.RELEV_VEHS_TIME_THRESH)+" AND "+str(curr_time+constants.RELEV_VEHS_TIME_THRESH)+") AND TRACKS.TYPE <> 'Pedestrian' AND TRACKS.TRACK_ID IN (SELECT DISTINCT TRACK_ID FROM TRAJECTORY_MOVEMENTS WHERE TRAFFIC_SEGMENT_SEQ LIKE '%"+path[0][:-1]+"%"+path[1][:-2]+"%' ORDER BY TRACK_ID)"
    c.execute(q_string)
    res = c.fetchall()
    if len(res) < 1:
        print('no relevant agents for query:',q_string)
    for row in res:
        vehicles.append(row[0])
    conn.close()
    return vehicles
        

def get_leading_vehicles(veh_state):
    
    path = veh_state.segment_seq
    curr_time = veh_state.current_time
    current_segment = veh_state.current_segment
    if current_segment[0:-1] == 'ln_s_-':
        return None
    
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
    print(list(ex_b_positions.keys()))
    veh_dist_to_segment_exit = (math.hypot(ex_b_positions[current_segment][0][0] - veh_pos_x, ex_b_positions[current_segment][1][0] - veh_pos_y) + \
                                math.hypot(ex_b_positions[current_segment][0][1] - veh_pos_x, ex_b_positions[current_segment][1][1] - veh_pos_y))/2
    
    ''' find the vehicles that are in the current segment or the next and appears within the window of the subject vehicle '''
    q_string = "SELECT T.TRACK_ID FROM TRAJECTORY_MOVEMENTS T, v_TIMES V WHERE (T.TRAFFIC_SEGMENT_SEQ LIKE '%"+current_segment+"%' OR T.TRAFFIC_SEGMENT_SEQ LIKE '%"+next_segment+"%') AND T.TRACK_ID = V.TRACK_ID AND (V.ENTRY_TIME <= "+str(curr_time)+" AND V.EXIT_TIME >= "+str(curr_time)+") AND T.TRACK_ID <> "+str(veh_id)
    c.execute(q_string)
    res = c.fetchall()
    potential_lead_vehicles = []
    if len(res) > 0:
        for row in res:
            leading_vehicle_id = row[0]
            ''' find the position of the potential lead vehicle in the current time '''
            q_string = "select * from trajectories_0769 where track_id="+str(leading_vehicle_id)+" and time = "+str(curr_time)
            c.execute(q_string)
            pt_res = c.fetchone()
            l_v_state = motion_planner.VehicleState()
            l_v_state.set_id(pt_res[0])
            l_v_state.set_current_time(curr_time)
            l_v_track = get_track(l_v_state)
            l_v_state.set_track_info(l_v_track[0,])
            if len(l_v_track) == 0 or (len(l_v_track) > 0 and len(l_v_track[0,8]) == 0) :
                l_v_track_region = guess_track_info(l_v_state)[8,]
                if l_v_track_region is None:
                    sys.exit('need to guess traffic region for relev agent')
            else:
                l_v_track_region = l_v_track[0,8]
            l_v_track_segment_seq = get_track_segment_seq(l_v_state.id)
            l_v_current_segment = get_current_segment(l_v_state,l_v_track_region,l_v_track_segment_seq,curr_time)
            if l_v_current_segment not in ex_b_positions.keys():
                ''' potential lead vehicle is not in the path, so ignore '''
                continue
            else:
                lead_vehicle_pos = (float(pt_res[1]),float(pt_res[2]))
                l_v_segment_ex_b = ex_b_positions[l_v_current_segment]
                l_v_dist_to_segment_exit = (math.hypot(ex_b_positions[l_v_current_segment][0][0] - lead_vehicle_pos[0], ex_b_positions[l_v_current_segment][1][0] - lead_vehicle_pos[1]) + \
                                    math.hypot(ex_b_positions[l_v_current_segment][0][1] - lead_vehicle_pos[0], ex_b_positions[l_v_current_segment][1][1] - lead_vehicle_pos[1]))/2
                l_v_state.set_dist_to_segment_exit(l_v_dist_to_segment_exit)
                if l_v_current_segment == current_segment and l_v_dist_to_segment_exit > veh_dist_to_segment_exit:
                    ''' this vehicle is behind the subject vehicle '''
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


def get_relevant_agents(veh_state):
    relev_agents = []
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
        if veh_state.gate_crossing_times[1] is not None and veh_state.current_time is not None and veh_state.current_time < veh_state.gate_crossing_times[1]:
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
    

def get_traffic_signal(time,direction):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    signal = None
    q_string = "SELECT MAX(TIME),"+direction+" FROM TRAFFIC_LIGHTS WHERE TIME - "+str(time)+" <= 0"
    c.execute(q_string)
    res = c.fetchone()
    signal = res[1]
    conn.close()
    return signal
    
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

def assign_curent_segment(traffic_region_list,track_segment):
    traffic_region_list = str(traffic_region_list).replace(' ','').strip(',')
    #current_segment = []
    #all_segments = ['int-entr_','execute-turn_','prepare-turn_','rt-stop_','rt-prep_turn_','rt_exec_turn_']
    traffic_region_list = traffic_region_list.split(',')
    for segment in reversed(track_segment):
        for track_region in traffic_region_list:
            if region_equivalence(track_region, segment):
                return segment
    return None  


def get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,curr_time):
    r_a_current_segment = assign_curent_segment(r_a_track_region,r_a_track_segment_seq)
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


def interpolate_track_info(veh_state,forward,backward,partial_track=None):
    if partial_track is not None:
        track_info = partial_track
    else:
        track_info = [None]*9
    veh_id,curr_time = veh_state.id,veh_state.current_time
    track_info[0],track_info[6] = veh_id,curr_time
    veh_entry_segment, veh_exit_segment = veh_state.segment_seq[0],veh_state.segment_seq[-1]
    q_string = "select TIME,TRAFFIC_REGIONS,X,Y,SPEED from trajectories_0769 where track_id="+str(veh_id)+" and time between "+str(curr_time-2)+" and "+str(curr_time+2)+" order by time"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    c.execute(q_string)
    res = c.fetchall()
    traffic_regions = []
    for r in res:
        traffic_regions.append((float(r[0]),r[1]))
    veh_entry_speed, veh_exit_speed = res[0][4],res[-1][4]
    for_idx,back_idx = 0,0
    if not forward and not backward:
        ''' interpolate in the middle'''
        track_info
        idx = [x[0] for x in traffic_regions].index(curr_time)
        for i in np.arange(idx,len(traffic_regions)):
            if traffic_regions[i][1] is not None and len(traffic_regions[i][1]) > 1:
                for_idx = i
                break
        for j in np.arange(idx,-1,-1):
            if traffic_regions[j][1] is not None and len(traffic_regions[j][1]) > 1:
                back_idx = j
                break
        if idx - back_idx < for_idx - idx:
            track_info[8] = traffic_regions[back_idx][1]
        else:
            track_info[8] = traffic_regions[for_idx][1]
    elif forward:
        ''' extrapolate forward in time '''
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+veh_exit_segment+"' AND REGION_PROPERTY = 'center_line'"
        c.execute(q_string)
        res = c.fetchone()
        if res is None:
            sys.exit("cannot find centerline for"+veh_exit_segment)
        exit_pos_X = ast.literal_eval(res[4])
        exit_pos_Y = ast.literal_eval(res[5])
        angle_of_centerline = math.atan2(exit_pos_Y[1]-exit_pos_Y[0],exit_pos_X[1]-exit_pos_X[0])
        proj_pos_X = exit_pos_X[1] + veh_entry_speed * math.cos(angle_of_centerline)
        proj_pos_Y = exit_pos_Y[1] + veh_entry_speed * math.cos(angle_of_centerline)
        idx = len(traffic_regions)-1
        for i in np.arange(idx,0,-1):
            if traffic_regions[i][1] is not None and len(traffic_regions[i][1]) > 1:
                for_idx = i
                break
        track_info[8] = traffic_regions[for_idx][1]
        track_info[1] = proj_pos_X
        track_info[2] = proj_pos_Y
        track_info[3] = veh_entry_speed
        track_info[4] = 0
        track_info[5] = 0
        track_info[7] = angle_of_centerline if angle_of_centerline > 0 else 2 * math.pi + angle_of_centerline 
        
    elif backward:
        ''' extrapolate backward in time '''
        idx = 0
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME = '"+veh_entry_segment+"' AND REGION_PROPERTY = 'center_line'"
        c.execute(q_string)
        res = c.fetchone()
        entry_pos_X = ast.literal_eval(res[4])
        entry_pos_Y = ast.literal_eval(res[5])
        angle_of_centerline = math.atan2(entry_pos_Y[0]-entry_pos_Y[1],entry_pos_X[0]-entry_pos_X[1])
        proj_pos_X = entry_pos_X[0] + veh_entry_speed * math.cos(angle_of_centerline)
        proj_pos_Y = entry_pos_Y[0] + veh_entry_speed * math.sin(angle_of_centerline)
        for i in np.arange(idx,len(traffic_regions)):
            if traffic_regions[i][1] is not None and len(traffic_regions[i][1]) > 1:
                back_idx = i
                break
        track_info[8] = traffic_regions[back_idx][1]
        track_info[1] = proj_pos_X
        track_info[2] = proj_pos_Y
        track_info[3] = veh_entry_speed
        track_info[4] = 0
        track_info[5] = 0
        track_info[7] = angle_of_centerline if angle_of_centerline > 0 else 2 * math.pi + angle_of_centerline
    conn.close()
    return track_info    
    
def guess_track_info(veh_state,partial_track=None):
    veh_id,curr_time = veh_state.id,veh_state.current_time
    curr_time = float(curr_time)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT ENTRY_TIME,EXIT_TIME FROM v_TIMES WHERE TRACK_ID = "+str(veh_id)
    c.execute(q_string)
    res = c.fetchone()
    entry_time,exit_time = res[0],res[1]
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
    conn.close()
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
                        
