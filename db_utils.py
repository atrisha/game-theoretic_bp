'''
Created on Jan 15, 2020

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3
import utils
import sys
import constants
from planning_objects import VehicleState
import ast
import matplotlib.pyplot as plt
from game_tree_builder import get_leading_vehicles
import os
from os import listdir
from collections import OrderedDict
import pandas as pd
from cost_evaluation import eval_complexity



def fix_exec_turn():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    ''' if a vehicle crosses prep-turn_x and before the relevant exit gates:
            add exec-turn_s to the area '''
    q_string = "SELECT TRACK_ID FROM TRACKS;"
    res = c.execute(q_string)
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    exec_turn_map = [('prep_turn_s',131),('prep-turn_n',34),('prep-turn_n',132),('prep-turn_e',18),('prep-turn_w',73),('prep-turn_e',130),('prep_turn_s',63)]
    veh_count = 0 
    for veh in vehicles:
        veh_count += 1
        print('fixing vehicle',veh_count,'/283')
        for e_map in exec_turn_map:
            q_string = "SELECT TRACK_ID,TIME,TRAFFIC_REGIONS FROM TRAJECTORIES_0769 WHERE TRACK_ID = "+str(veh)+" AND TIME BETWEEN (SELECT MAX(TIME) FROM TRAJECTORIES_0769 WHERE "+\
             "TRACK_ID = "+str(veh)+" AND TRAFFIC_REGIONS LIKE '%"+e_map[0]+"%') AND (SELECT TIME FROM GATE_CROSSING_EVENTS WHERE GATE_ID = "+str(e_map[1])+" AND TRACK_ID = "+str(veh)+")"
            c.execute(q_string)
            res = c.fetchall()
            for _traj_pts in res:
                if 'exec-turn_'+e_map[0][-1] not in str(_traj_pts[2]):
                    u_string = "UPDATE TRAJECTORIES_0769 SET TRAFFIC_REGIONS='"+_traj_pts[2]+",exec-turn_"+e_map[0][-1]+"' WHERE TRACK_ID="+str(_traj_pts[0])+" AND TIME="+str(_traj_pts[1])
                    if _traj_pts[0] == 11 and _traj_pts[1] == 5.372033:
                        brk = 9
                    c.execute(u_string)
                    conn.commit()
    conn.commit()
    conn.close()


def fix_lane_assignments():
    ''' '''    
    gate_map = {(65,18) : 'ln_s_-2', }
    
def assign_traffic_segment_seq():
    traffic_segments = None
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRACK_ID FROM TRACKS WHERE TYPE <> 'Pedestrian' AND TYPE <> 'Bicycle'"
    res = c.execute(q_string)
    vehicles = []
    for row in res:
        vehicles.append(row[0])
    veh_count = 0 
    direction_less = []
    u_strings = []
    for veh in vehicles:
        veh_count += 1
        #print('assigning traffic region',veh,'/283')
        q_string = "SELECT TRAFFIC_REGIONS FROM TRAJECTORIES_0769 WHERE TRACK_ID = "+str(veh)
        c.execute(q_string)
        res = c.fetchall()
        track_regions = []
        if veh == 99:
            brk = 8
        for row in res:
            if row[0] is not None and len(row[0]) > 1:
                track_regions.append(row[0])
        
        path,gates,direction = utils.get_path_gates_direction(track_regions,veh)
        if veh == 2:
            brek = 4
        if gates[0] is None and gates[1] is None:
            if not path[0] is 'NA' and not path[1] is 'NA':
                o,d = path[0][3],path[1][3]
                if (o,d) == ('n','s') or (o,d) == ('s','n') or (o,d) == ('e','w') or (o,d) == ('w','e'):
                    traffic_segments = [path[0],'int_entry_'+o,path[1]]
                elif (o,d) == ('n','e') or (o,d) == ('s','w') or (o,d) == ('e','s'):
                    traffic_segments = [path[0],'prep-turn_'+o,'exec-turn_'+o,path[0]]
                elif (o,d) == ('n','w') or (o,d) == ('e','n'):
                    traffic_segments = [path[0],'l_'+o+'_'+d,path[1]]
                elif (o,d) == ('s','e') or (o,d) == ('w','s'):
                    traffic_segments = [path[0],'rt-stop_'+o,'rt_prep_turn_'+o,'rt-exec-turn_'+o,path[1]]
                u_string = "UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ='"+str(traffic_segments)+"' WHERE TRACK_ID="+str(veh)
                u_strings.append(["UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ=? WHERE TRACK_ID=?",str(traffic_segments).replace(' ',''),str(veh)])
                print(u_string)
        else:
            if gates[0] is not None and gates[1] is not None:
                possible_traffic_segments = utils.get_traffic_segment_from_gates(gates)
                if path[0] == 'NA' and path[-1] == 'NA':
                    traffic_segments = possible_traffic_segments[0]
                else:
                    for t_s in possible_traffic_segments:
                        ''' there are multiple possible segments based on the gates, so reconcile with the path and decide which is the right one'''
                        if path[0] == t_s[0] and path[-1] == t_s[-1]:
                            traffic_segments = t_s
                            break
                        elif path[-1] != 'NA' and path[-1] == t_s[-1]:
                            traffic_segments = t_s
                        elif path[0] != 'NA' and path[0] == t_s[0]:
                            traffic_segments = t_s
                        else:
                            traffic_segments = t_s
                u_string = "UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ='"+str(traffic_segments)+"' WHERE TRACK_ID="+str(veh)
                u_strings.append(["UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ=? WHERE TRACK_ID=?",str(traffic_segments).replace(' ',''),str(veh)])
                print(u_string)
            else:
                if len(gates) > 0:
                    direction_less.append(veh)
    if len(direction_less) > 0:
        print(direction_less)
        sys.exit(str(len(direction_less))+' direction less vehicles found. please fix them first')
    else:
        for u in u_strings:
            c.execute(u[0],(u[1],u[2]))
            
        '''
        if o is not None and d is not None:
            if (o,d) == ('n','s') or (o,d) == ('s','n') or (o,d) == ('e','w') or (o,d) == ('w','e'):
                traffic_segments = ['ln_'+o+'_','int_entry_'+o,'ln_'+d+'_']
            else:
                traffic_segments = ['ln_'+o+'_','prep-turn_'+o,'exec-turn_'+o,'ln_'+d+'_']
        '''
        #u_string = "UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ=? WHERE TRACK_ID=?"
        #print(u_string)
        #c.execute(u_string,(str(traffic_segments),str(veh)))
    conn.commit()
    conn.close()




def replace_single_quotes(filepath):
    # Read in the file
    with open(filepath, 'r') as file :
        filedata = file.read()
    
    # Replace the target string
    filedata = filedata.replace(',",', ',"",')

    # Write the file out again
    with open(filepath, 'w') as file:
        file.write(filedata)
    
def insert_trajectories():
    traffic_segments = dict()
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\2401\\DJI_0769_traj(segments).csv"
    replace_single_quotes(file_path)
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        ct = 0
        for row in csvreader:
            size = len(row)
            if ct > 0:
                ins_list = [None if x.strip()=='-' else x for x in row[:8]]
                ins_list = [x.strip("'") if x is not None else None for x in ins_list]
                track_id = ins_list[0]
                i = 0
                if str(track_id) == '26':
                    brk = 5
                rest = row[8:]
                while i+8 < size: 
                    ins_traj_list = rest[i:i+8]
                    if len(ins_traj_list) == 8:
                        ins_traj_list = [None if x.strip()=='-' else x for x in ins_traj_list]
                        ins_traj_list = [x.strip('"') if x is not None else None for x in ins_traj_list]
                        traj_ts = ins_traj_list[5]
                        
                        if ins_traj_list[7] is not None and len(ins_traj_list[7]) > 1:
                            key_str = str(track_id)+'_'+str(float(traj_ts))
                            traffic_segments[key_str] = ins_traj_list[7]
                    i += 8
            ct += 1
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\2401\\DJI_0769_traj.csv"
    replace_single_quotes(file_path)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    '''
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            line.replace(',",', ',,')
    '''
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        ct = 0
        for row in csvreader:
            size = len(row)
            if ct > 0:
                ins_list = [None if x.strip()=='-' else x for x in row[:8]]
                #ins_list = [x.strip("'") if x is not None else None for x in ins_list]
                print(ins_list)
                #q_string = "INSERT INTO TRACKS VALUES (0769,"+ins_list[0]+",'"+ins_list[1]+"',"+ins_list[2]+","+ins_list[4]+",'"+ \
                #ins_list[6]+"',"+ins_list[7]+",'"+ins_list[8]+"',"+ins_list[9]+","+ins_list[10]+","+ins_list[11]+')'
                track_id = ins_list[0]  
                q_string = "INSERT INTO TRACKS VALUES (0769,?,?,?,?,?,?,?,?,?,?)"
                c.execute(q_string, (ins_list[0],ins_list[1],None,None,ins_list[2],ins_list[3],ins_list[4],ins_list[5],ins_list[6],ins_list[7]))
                i = 0
                rest = row[8:]
                while i+8 < size: 
                    ins_traj_list = rest[i:i+8]
                    if len(ins_traj_list) == 8:
                        ins_traj_list = [None if x.strip()=='-' else x for x in ins_traj_list]
                        #ins_traj_list = [x.strip("'") if x is not None else None for x in ins_traj_list]
                        traj_ts = ins_traj_list[5]
                        key_str = str(track_id)+'_'+str(float(traj_ts))
                        if key_str in traffic_segments and traffic_segments[key_str] is not None:
                            if ins_traj_list[7] is not None:
                                ins_traj_list[7] = ins_traj_list[7] + ',' + traffic_segments[key_str]
                            else:
                                ins_traj_list[7] = traffic_segments[key_str]
                        q_string = "INSERT INTO TRAJECTORIES_0769 VALUES (?,?,?,?,?,?,?,?,?)"
                        #print(' ',ins_traj_list)
                        c.execute(q_string,(ins_list[0],ins_traj_list[0],ins_traj_list[1],ins_traj_list[2],ins_traj_list[3],ins_traj_list[4],ins_traj_list[5],ins_traj_list[6],ins_traj_list[7]))
                    i += 8
            ct += 1
        print('total trajectories:',ct)
    conn.commit()
    conn.close()
    print('# fixing execute turn regions')
    fix_exec_turn()
    

def insert_gate_crossings():
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\DJI_0769_gate_crossings_csv.csv"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    '''
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            line.replace(',",', ',,')
    '''
    with open(file_path, newline='') as csvfile:
        d=7
        
        
    conn.commit()
    conn.close()
    
    
def insert_trajectory_movements():
    
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\2401\\DJI_0769_traj_movement.csv"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    '''
    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            line.replace(',",', ',,')
    '''
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', skipinitialspace=True)
        ct = 0
        for row in csvreader:
            ins_list = row[:-1]
            ins_list = [None if x=='---' else x for x in ins_list]
            if ct > 0:
                q_string = "INSERT INTO TRAJECTORY_MOVEMENTS VALUES (0769,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                
                c.execute(q_string, (ins_list[0],ins_list[1],ins_list[2],ins_list[3],ins_list[4],ins_list[5],ins_list[6],ins_list[7],ins_list[8],ins_list[9],ins_list[10],ins_list[11], 
                          ins_list[12],ins_list[13],ins_list[14],ins_list[15],ins_list[16],ins_list[17],ins_list[18],ins_list[19],ins_list[20],ins_list[21],
                          ins_list[22],ins_list[23],ins_list[24],ins_list[25],ins_list[26],ins_list[27],ins_list[28],ins_list[29],ins_list[30],ins_list[31],
                          ins_list[32],ins_list[33],ins_list[34],None))
                
                print(ins_list)
                
            ct += 1
            
    conn.commit()
    conn.close()
    
    
def insert_gate_crossing_events():
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\2401\\DJI_0769_gate_crossings.csv"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        ct = 0
        for row in csvreader:
            ins_list = row
            ins_list = [None if x==' ' else x for x in ins_list]
            if ct > 0:
                q_string = "INSERT INTO `GATE_CROSSING_EVENTS` VALUES (0769,?,?,?,?,?,?,?,?,?,?)"
                
                c.execute(q_string, (ins_list[0],ins_list[1],ins_list[2],ins_list[3],ins_list[4],ins_list[5],ins_list[6],ins_list[7],ins_list[8],ins_list[9]))
                print(ins_list)
                
            ct += 1
            
    conn.commit()
    conn.close()
    
def insert_traffic_lights():
    file_path = "D:\\intersections_dataset\\all_tracks\\exported\\DJI_0769_traffic_lights.csv"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT DISTINCT TIME FROM TRAJECTORIES_0769"
    c.execute(q_string)
    res = c.fetchall()
    all_times = []
    for row in res:
        all_times.append(float(row[0]))
    
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        ct = 0
        for row in csvreader:
            ins_list = row
            if ct > 0:
                time_ms = float(ins_list[-1])
                closest_time = min(all_times, key=lambda x:abs(x-time_ms))
                q_string = "INSERT INTO `TRAFFIC_LIGHTS` VALUES (0769,?,?,?,?,?,?,?,?,?)"
                c.execute(q_string, (ins_list[0],ins_list[1],ins_list[2],ins_list[3],ins_list[4],ins_list[5],ins_list[6],ins_list[7],closest_time))
                
                
            ct += 1
            
    conn.commit()
    conn.close()


def assign_curent_segment():
    '''
    traffic_region_list = str(traffic_region_list).replace(' ','')
    current_segment = []
    all_segments = ['int-entr_','exec-turn_','prep-turn_','rt-stop_','rt-prep_turn_','rt_exec_turn_']
    traffic_region_list = traffic_region_list.split(',')
    for region in traffic_region_list:
        if region.startswith('l') or region.startswith('ln'):
            _tags = region.split('_')
            origin = _tags[1]
            for segment in all_segments:
                if segment+origin in traffic_region_list:
                    current_segment.append(segment+origin)
    if len(current_segment) > 1:
        print(current_segment)
        sys.exit('Error: current segment ambiguous')
        
    traffic_segments = None
    '''
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    vehs = utils.get_agents_for_task('S_W')
    #vehs = [140]
    q_string = "select TRAJECTORIES_0769.*,TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ,v_TIMES.ENTRY_TIME,v_TIMES.EXIT_TIME from TRAJECTORIES_0769,TRAJECTORY_MOVEMENTS,v_TIMES where TRAJECTORIES_0769.TRACK_ID=TRAJECTORY_MOVEMENTS.TRACK_ID and v_TIMES.TRACK_ID = TRAJECTORIES_0769.TRACK_ID ORDER BY TRAJECTORIES_0769.TRACK_ID,TRAJECTORIES_0769.TIME"
    res = c.execute(q_string)
    traj_info = dict()
    for row in res:
        if row[0] in vehs:
            if row[0] not in traj_info:
                traj_info[row[0]] = [row]
            else:
                traj_info[row[0]].append(row)
    N = sum([len(x) for x in traj_info.values()])
    ct = 1
    color_map = {'ln_s_1':'g','prep-turn_s':'orange','exec-turn_s':'coral','ln_w_-1':'lime','ln_w_-2':'lime',
                 'ln_n_2':'g','ln_n_3':'g','l_n_s_l':'orange','l_n_s_r':'orange','ln_s_-1':'lime','ln_s_-2':'lime'}
    
    for agent_id, track in traj_info.items():
        u_strings = []
        colors = []
        assigned_segments = []
        veh_state = VehicleState()
        veh_state.set_id(agent_id)
        veh_state.set_segment_seq(ast.literal_eval(track[0][9]) if track[0][9] is not None else None)
        path,gates,direction = utils.get_path_gates_direction([x[8] for x in track],agent_id)
        veh_state.set_gates(gates)
        gate_crossing_times = utils.gate_crossing_times(veh_state)
        veh_state.set_gate_crossing_times(gate_crossing_times)
        if gate_crossing_times[0] is None:
            ''' this vehicle track starts in the middle '''
            q_string = "select * from TRAFFIC_REGIONS_DEF where NAME = '"+veh_state.segment_seq[0]+"' AND REGION_PROPERTY = 'center_line'"
            print(q_string)
            c.execute(q_string)
            res = c.fetchone()
            if res is None or len(res) < 1:
                sys.exit('center line not found for '+veh_state.segment_seq[0])
            path_origin = (ast.literal_eval(res[4])[0],ast.literal_eval(res[5])[0])
            veh_state.set_path_origin(path_origin)
        veh_state.set_full_track(track)
        veh_state.set_entry_exit_time((float(track[0][10]), float(track[0][11])))
        v_plots = []
        for t_idx,track_pt in enumerate(track):
            ct += 1
            veh_state.set_current_time(track_pt[6])
            veh_state.x = float(track_pt[1])
            veh_state.y = float(track_pt[2])
            veh_track_region = None
            '''
            if track_pt[8] is None or len(track_pt[8]) == 0:
                veh_track_region = utils.guess_track_info(veh_state)[8,]
                if veh_track_region is None:
                    print(agent_id,track_pt[6])
                    sys.exit('need to guess traffic region for agent')
            else:
                veh_track_region = track_pt[8]
            '''
            if veh_state.current_time == 88.421667:
                brk = 1
    
            current_segment = utils.assign_curent_segment(veh_track_region,veh_state)
            '''
            if len(assigned_segments) > 0 and current_segment != assigned_segments[-1]:
                v_plots.append((track[t_idx][6],color_map[assigned_segments[-1]]))
            ''' 
            #print(track_pt[0],track_pt[6],veh_track_region,track_pt[9],current_segment,str(ct)+'/'+str(N))
            
            assigned_segments.append(current_segment)
            '''
            if current_segment is None:
                colors.append('black')
            else:
                colors.append(color_map[current_segment])
            '''
        boundaries = []
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        
        ''' for each segment loop and check if the vehicle has not yet crossed it'''
        '''
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE NAME IN ('ln_s_1','prep-turn_s','exec-turn_s','ln_w_-1') AND REGION_PROPERTY = 'exit_boundary'"
        c.execute(q_string)
        rows = c.fetchall()
        for res in rows:
            exit_pos_X = ast.literal_eval(res[4])
            exit_pos_Y = ast.literal_eval(res[5])
            boundaries.append((exit_pos_X,exit_pos_Y))
        plt.axis('equal')
        for b in boundaries:
            plt.plot(b[0],b[1],'-')
        '''
        #plt.scatter([x[1] for x in track],[x[2] for x in track],c=colors)
        #plt.show()
        '''
        l1_actions = []
        vel_color_map = {'decelerate':'orange','wait':'r','proceed':'g','track_speed':'b','NA':'black'}
        vel_colors = []
        l1_actions = ['wait' if 0 <= v <= 1 else 'NA' for v in [x[3] for x in track]]
        for i,v in enumerate([x[3] for x in track]):
            if l1_actions[i] != 'wait':
                action = 'NA'
                if assigned_segments[i] == 'ln_s_1':
                    for j in np.arange(i,len(track)):
                        if l1_actions[j] == 'wait' and assigned_segments[j] in ['ln_s_1','prep-turn_s']:
                            action = 'decelerate'
                            break
                    if action == 'NA':
                        action = 'proceed'
                elif assigned_segments[i] == 'prep-turn_s':
                    for j in np.arange(i,len(track)):
                        if l1_actions[j] == 'wait' and assigned_segments[j] in ['prep-turn_s']:
                            action = 'wait'
                            break
                    if action == 'NA':
                        action = 'proceed'
                elif assigned_segments[i] == 'exec-turn_s':
                    for j in np.arange(i,len(track)):
                        if l1_actions[j] == 'wait' and assigned_segments[j] in ['exec-turn_s']:
                            action = 'wait'
                            break
                    if action == 'NA':
                        action = 'proceed'
                elif assigned_segments[i] == 'ln_w_-1':
                    action = 'track_speed'
                l1_actions[i] = action
            vel_colors.append(vel_color_map[l1_actions[i]])
         
        plt.scatter([x[6] for x in track],[x[3] for x in track],c=vel_colors)
        plt.axvspan(track[0][6], v_plots[0][0], alpha=0.15, color=v_plots[0][1])
        for idx in np.arange(len(v_plots)-1):
            plt.axvspan(v_plots[idx][0], v_plots[idx+1][0], alpha=0.15, color=v_plots[idx+1][1])
        plt.axvspan(v_plots[-1][0], track[-1][6], alpha=0.15, color=color_map[assigned_segments[-1]])
        plt.show()
        '''
        i_string = 'REPLACE INTO TRAJECTORIES_0769_EXT VALUES (?,?,?,?,?)'
        dat_l = []
        for i in np.arange(len(track)):
            #u_string = "UPDATE TRAJECTORIES_0769_EXT VALUES (?,?,?,?)"
            #u_string = "UPDATE TRAJECTORIES_0769_EXT SET L1_ACTION='"+" WHERE TRACK_ID="+" AND TIME="+"str(_traj_pts[1])"
            #u_strings.append(u_string)
            #u_string = "UPDATE TRAJECTORIES_0769_EXT SET L2_ACTION='"+" WHERE TRACK_ID="+" AND TIME="+"str(_traj_pts[1])"
            #u_strings.append(u_string)
            dat_l.append((agent_id, track[i][6], assigned_segments[i], None, None))
            print((agent_id, track[i][6], assigned_segments[i], None,None))
        c.executemany(i_string,dat_l)
        conn.commit()            
    conn.close()
    return None    
            
def assign_actions():
    vehs_sw = utils.get_agents_for_task('S_W')
    vehs_ns = utils.get_agents_for_task('N_S')
    vehs = vehs_sw + vehs_ns
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select TRAJECTORIES_0769.*,TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ,v_TIMES.ENTRY_TIME,v_TIMES.EXIT_TIME,TRAJECTORIES_0769_EXT.* from TRAJECTORIES_0769,TRAJECTORY_MOVEMENTS,v_TIMES,TRAJECTORIES_0769_EXT where TRAJECTORIES_0769.TRACK_ID=TRAJECTORY_MOVEMENTS.TRACK_ID and v_TIMES.TRACK_ID = TRAJECTORIES_0769.TRACK_ID and TRAJECTORIES_0769.TRACK_ID=TRAJECTORIES_0769_EXT.TRACK_ID and TRAJECTORIES_0769.TIME=TRAJECTORIES_0769_EXT.TIME ORDER BY TRAJECTORIES_0769.TRACK_ID,TRAJECTORIES_0769.TIME"
    res = c.execute(q_string)
    traj_info = dict()
    for row in res:
        if row[0] in vehs:
            lane_type = constants.SEGMENT_MAP[row[-3]]
            if lane_type not in traj_info:
                traj_info[lane_type] = dict()
                traj_info[lane_type][row[0]] = [row]
            else:
                if row[0] not in traj_info[lane_type]:
                    traj_info[lane_type][row[0]] = [np.array(row)]
                else:
                    traj_info[lane_type][row[0]].append(np.array(row))
    ct = 0
    for lane_type, track_dict in traj_info.items():
        ct += 1
        for agent_id in track_dict.keys():
            for idx,track_pt in enumerate(track_dict[agent_id]):
                track_pt = np.asarray(track_pt)
                veh_state = VehicleState()
                veh_state.set_track_info(track_pt)
                veh_state.set_current_segment(track_pt[-3,])
                segment_seq = ast.literal_eval(track_pt[9])
                veh_state.set_segment_seq(segment_seq)
                leading_vehicle = get_leading_vehicles(veh_state)
                veh_state.set_leading_vehicle(leading_vehicle)
                direction = 'L_'+segment_seq[0][3].upper()+'_'+segment_seq[-1][3].upper()
                traffic_signal = utils.get_traffic_signal(veh_state.time, direction)
                veh_state.set_traffic_light(traffic_signal)
                possible_actions = utils.get_actions(veh_state)
                X = [float(x[6]) for x in track_dict[agent_id]]
                Y = [float(x[3]) for x in track_dict[agent_id]]
                plt.plot(np.arange(len(Y)),Y,'g')
        plt.title(lane_type)
        plt.show()


def insert_generated_traj_info():
    if not os.path.isfile(constants.L3_ACTION_CACHE+'traj_metadata.dict'):
        dir = constants.L3_ACTION_CACHE
        N,ct = len(listdir(dir)),0
        state_dict = OrderedDict()
        for f in listdir(dir):
            ct += 1
            traj = utils.pickle_load(os.path.join(dir, f))
            if f == '7690110000101_0':
                brk=1
            if len(traj) != 0:
                agent_id = int(f[3:6])
                time_ts = round(float(f.split('_')[-1])/1000,3)
                relev_agent = None if f[6:9] == '000' else int(f[6:9])
                l1_action = [k for k,v in constants.L1_ACTION_CODES.items() if v == int(f[9:11])][0]
                l2_action = [(k,len(traj)) for k,v in constants.L2_ACTION_CODES.items() if v == int(f[11:13])][0]
                lead_vehicle = None
                v = VehicleState()
                if relev_agent is None:
                    v.set_id(agent_id)
                else:
                    v.set_id(relev_agent)
                time_tuple = utils.get_entry_exit_time(v.id)
                v.set_entry_exit_time(time_tuple)
                if agent_id==11:
                    brk =1 
                agent_track = utils.get_track(v, None)
                all_times = [float(agent_track[i][6,]) for i in np.arange(len(agent_track))]
                real_ts = -1
                for ts in all_times:
                    if round(float(ts)*1000) == round(time_ts*1000):
                        real_ts = ts
                        break
                if real_ts != -1:
                    time_ts = real_ts
                else:
                    print('ts',time_ts,v.id,real_ts,all_times[0],all_times[-1])
                f_new = f[0:14]+str(time_ts).replace('.',',')
                os.rename(os.path.join(dir,f),os.path.join(dir,f_new))
                v.set_current_time(time_ts)
                print(ct,'/',N,agent_id,time_ts,relev_agent,l1_action,l2_action)
                if time_ts not in state_dict:
                    state_dict[time_ts] = dict()
                if agent_id not in state_dict[time_ts]:
                    state_dict[time_ts][agent_id] = dict()
                    state_dict[time_ts][agent_id]['actions'] = dict()
                if relev_agent is None:
                    if l1_action not in state_dict[time_ts][agent_id]['actions']:
                        state_dict[time_ts][agent_id]['actions'][l1_action] = [l2_action]
                    else:
                        state_dict[time_ts][agent_id]['actions'][l1_action].append(l2_action)
                else:
                    if 'relev_agents' not in state_dict[time_ts][agent_id]: 
                        state_dict[time_ts][agent_id]['relev_agents'] = dict()
                    if relev_agent not in state_dict[time_ts][agent_id]['relev_agents']:
                        state_dict[time_ts][agent_id]['relev_agents'][relev_agent] = dict()
                    if l1_action not in state_dict[time_ts][agent_id]['relev_agents'][relev_agent]:
                        state_dict[time_ts][agent_id]['relev_agents'][relev_agent][l1_action] = [l2_action]
                    else:
                        state_dict[time_ts][agent_id]['relev_agents'][relev_agent][l1_action].append(l2_action)
        state_dict = OrderedDict(sorted((float(key), value) for key, value in state_dict.items()))       
        utils.pickle_dump(constants.L3_ACTION_CACHE+'traj_metadata.dict', state_dict)
    else:
        state_dict = utils.pickle_load(constants.L3_ACTION_CACHE+'traj_metadata.dict')
    i_strings = []
    for ts,traj_det in state_dict.items():
        for sub_agent,state_info in traj_det.items():
            for k,v in state_info.items():
                if k == 'relev_agents':
                    for ra,rv in v.items():
                        ac_l = []
                        for l1,l2 in rv.items():
                            if l1 in ['track_speed', 'follow_lead', 'decelerate-to-stop', 'wait_for_lead_to_cross']:
                                print('---- skipped')
                                continue
                            for l2_a in l2:
                                #act_str = unreadable(str(sub_agent)+'|'+str(ra)+'|'+l1+'|'+l2_a)
                                i_string_data = (769,sub_agent,ra,l1,l2_a[0],ts,l2_a[1])
                                i_strings.append(i_string_data)
                else:
                    for l1,l2 in v.items():
                        if l1 in ['track_speed', 'follow_lead', 'decelerate-to-stop', 'wait_for_lead_to_cross']:
                            print('---- skipped')
                            continue
                        for l2_a in l2:
                            #act_str = unreadable(str(k)+'|000|'+l1+'|'+l2_a)
                            i_string_data = (769,sub_agent,0,l1,l2_a[0],ts,l2_a[1])
                            i_strings.append(i_string_data)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    
    for i_s in i_strings:
        print('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?)',i_s)
        c.execute('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,NULL,?,?,?,?,?,?)',i_s)
    conn.commit()
    conn.close()
                


def insert_generated_trajectories():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    c.execute('delete from GENERATED_TRAJECTORY')
    conn.commit()
    q_string = "SELECT * FROM GENERATED_TRAJECTORY_INFO ORDER BY AGENT_ID"
    c.execute(q_string)
    res = c.fetchall()
    dir = constants.L3_ACTION_CACHE
    traj_ct,ct = 0,0
    N = len(res)
    for row in res:
        if row[4] in ['track_speed', 'follow_lead', 'decelerate-to-stop', 'wait_for_lead_to_cross']:
            print('---- skipped')
            continue
        file_id = str(row[0])
        traj_id = int(row[1])
        agent_id = str(row[2]).zfill(3)
        relev_agent_id = str(row[3]).zfill(3)
        l1_action_code = str(constants.L1_ACTION_CODES[row[4]]).zfill(2)
        l2_action_code = str(constants.L2_ACTION_CODES[row[5]]).zfill(2)
        time_ts = str(float(row[6]))
        file_str = file_id+agent_id+relev_agent_id+l1_action_code+l2_action_code+'_'+str(time_ts).replace('.',',')
        has_file = os.path.isfile(constants.L3_ACTION_CACHE+file_str)
        if not has_file:
            u_string = "UPDATE GENERATED_TRAJECTORY_INFO SET TRAJ_LEN=0 WHERE GENERATED_TRAJECTORY_INFO.TRAJ_ID="+str(traj_id)
            c.execute(u_string)
            conn.commit()
            print('---- updated')
            continue
        if file_str == '7690110000101_2,002':
            brk = 1
        trajectories = utils.load_traj_from_str(constants.L3_ACTION_CACHE+file_str)
        N_traj = len(trajectories)
        lt_ct = 0
        assert(abs(len(trajectories)-int(row[-1]))<=2)
        print(str(traj_id)+' '+'('+str(ct)+'/'+str(N)+')',':', row[4])
        for traj in trajectories:
            traj_ct += 1
            lt_ct += 1
            traj['TRAJECTORY_ID'] = [traj_ct]*traj['time'].size
            traj['TRAJECTORY_INFO_ID'] = [traj_id]*traj['time'].size
            #traj.rename(columns={'v':'speed','a':'long_accel','j':'long_jerk'})
            cols = traj.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            _traj = traj[cols]
            #traj.to_sql('GENERATED_TRAJECTORY', engine, if_exists='append', index=False, method='multi', chunksize = 1000)
            dat_l = _traj.values.tolist()
            i_string = 'INSERT INTO GENERATED_TRAJECTORY VALUES (?,?,?,?,?,?,?,?,?)'
            c.executemany(i_string,dat_l)
        print('inserted',lt_ct)
        print()
        conn.commit()
        ct += 1
        
    conn.close()
    
    
        
    
    
def insert_trajectory_complexity():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "SELECT TRAJECTORY_ID FROM GENERATED_TRAJECTORY_COMPLEXITY"
    c.execute(q_string)
    already_in = [x[0] for x in c.fetchall()]
    q_string = "SELECT * FROM GENERATED_TRAJECTORY_INFO ORDER BY AGENT_ID"
    c.execute(q_string)
    res = c.fetchall()
    dir = constants.L3_ACTION_CACHE
    traj_ct,ct = 0,0
    N = len(res)
    for row in res:
        file_id = str(row[0])
        traj_info_id = int(row[1])
        agent_id = str(row[2]).zfill(3)
        relev_agent_id = str(row[3]).zfill(3)
        l1_action_code = str(constants.L1_ACTION_CODES[row[4]]).zfill(2)
        l2_action_code = str(constants.L2_ACTION_CODES[row[5]]).zfill(2)
        file_str = file_id+agent_id+relev_agent_id+l1_action_code+l2_action_code
        gen_trajs = dict()
        q_string = "SELECT * FROM GENERATED_TRAJECTORY where TRAJECTORY_INFO_ID="+str(traj_info_id)
        cursor = c.execute(q_string)
        data_index = [description[0] for description in cursor.description]
        gen_traj_res = c.fetchall()
        for traj_pt in gen_traj_res:
            if traj_pt[0] not in gen_trajs:
                gen_trajs[traj_pt[0]] = [traj_pt]
            else:
                gen_trajs[traj_pt[0]].append(traj_pt)
        ct += 1
        gen_trajs = {k:np.asarray(v) for k,v in gen_trajs.items()}
        i_string_l = []
        for id,traj in gen_trajs.items():
            print('('+str(ct)+'/'+str(N)+')'+str(id))
            if id in already_in:
                continue
            try:
                compl = eval_complexity(traj,file_str)
            except:
                compl = None
            
            i_string_l.append((id,compl))
        i_string = "INSERT INTO GENERATED_TRAJECTORY_COMPLEXITY VALUES (?,?)"
        if len(i_string_l) > 0:
            c.executemany(i_string,i_string_l)
            #plt.plot(np.arange(traj.shape[0]),traj[:,6])    
        #plt.title(row[4])
        #plt.show()
        conn.commit()
    conn.close()



#assign_curent_segment()
