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
                u_strings.append(["UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ=? WHERE TRACK_ID=?",str(traffic_segments),str(veh)])
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
                u_strings.append(["UPDATE TRAJECTORY_MOVEMENTS SET TRAFFIC_SEGMENT_SEQ=? WHERE TRACK_ID=?",str(traffic_segments),str(veh)])
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
    color_map = {'ln_s_1':'g','prep-turn_s':'orange','exec-turn_s':'coral','ln_w_-1':'lime','ln_w_-2':'lime'}
    
    for agent_id, track in traj_info.items():
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
            q_string = "select * from TRAFFIC_REGIONS_DEF where NAME = '"+str(veh_state.segment_seq).replace("'","''")+"' AND REGION_PROPERTY = 'path_origin'"
            print(q_string)
            c.execute(q_string)
            res = c.fetchone()
            if res is None or len(res) < 1:
                sys.exit('path origin not found for '+str(veh_state.segment_seq))
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
            if len(assigned_segments) > 0 and current_segment != assigned_segments[-1]:
                v_plots.append((track[t_idx][6],color_map[assigned_segments[-1]]))
                
            #print(track_pt[0],track_pt[6],veh_track_region,track_pt[9],current_segment,str(ct)+'/'+str(N))
            
            assigned_segments.append(current_segment)
            if current_segment is None:
                colors.append('black')
            else:
                colors.append(color_map[current_segment])
        boundaries = []
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        
        ''' for each segment loop and check if the vehicle has not yet crossed it'''
        
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
        
        plt.scatter([x[1] for x in track],[x[2] for x in track],c=colors)
        plt.show()
        
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
        
        for i in np.arange(len(track)):
            u_string = "INSERT INTO TRAJECTORIES_0769_EXT VALUES (?,?,?,?)"
            c.execute(u_string, (agent_id, track[i][6], assigned_segments[i], l1_actions[i]))
            print(u_string,(agent_id, track[i][6], assigned_segments[i], l1_actions[i]))
        conn.commit()            
    conn.close()
    return None    
            
def assign_l1_actions():
    vehs = utils.get_agents_for_task('S_W')
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select TRAJECTORIES_0769.*,TRAJECTORY_MOVEMENTS.TRAFFIC_SEGMENT_SEQ,v_TIMES.ENTRY_TIME,v_TIMES.EXIT_TIME from TRAJECTORIES_0769,TRAJECTORY_MOVEMENTS,v_TIMES where TRAJECTORIES_0769.TRACK_ID=TRAJECTORY_MOVEMENTS.TRACK_ID and v_TIMES.TRACK_ID = TRAJECTORIES_0769.TRACK_ID ORDER BY TRAJECTORIES_0769.TRACK_ID,TRAJECTORIES_0769.TIME"
    res = c.execute(q_string)
    traj_info = dict()
    for row in res:
        if row[0] in vehs:
            if row[0] not in traj_info:
                traj_info[row[0]] = [row]
            else:
                traj_info[row[0]].append(row)
    for agent_id, track in traj_info.items():
        X = [float(x[6]) for x in track]
        Y = [float(x[3]) for x in track]
        plt.plot(X,Y)
        plt.show()


#assign_curent_segment()


