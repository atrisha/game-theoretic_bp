'''
Created on Jan 29, 2020

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3
import utils
import sys
import constants
import motion_planner
import ast
import math
import cost_evaluation
import pickle
import os.path
from planning_objects import VehicleState
import matplotlib.pyplot as plt
import visualizer
from os import listdir
from collections import OrderedDict


def setup_lead_vehicle(v_state,from_ra):
    if not from_ra:
        l_v_state = VehicleState()
        l_v_state.x = v_state.x
        l_v_state.y = v_state.y
        l_v_state.id = v_state.id
        l_v_state.set_current_time(v_state.current_time+constants.PLAN_FREQ)
        acc_proj = v_state.long_acc
        l_v_direction = veh_direction = 'L_'+v_state.segment_seq[0][3].upper()+'_'+v_state.segment_seq[-1][3].upper()
        l_v_state.set_segment_seq(v_state.segment_seq)
        path,gates,direction = utils.get_path_gates_direction(None,l_v_state.id)
        task = constants.TASK_MAP[direction]
        l_v_state.set_gates(gates)
        gate_crossing_times = utils.gate_crossing_times(l_v_state)
        l_v_state.set_gate_crossing_times(gate_crossing_times)
        ''' this is the segment from previous time stamp at this point'''
        l_v_state.current_segment = v_state.current_segment
        ''' segment updated '''
        current_segment = utils.assign_curent_segment(None, l_v_state, True)
        l_v_state.set_current_segment(current_segment)
        if current_segment[0:2] == 'ln':
            veh_current_lane = current_segment
        elif 'int-entry' in current_segment:
            dir_key = str(v_state.segment_seq[0][3:])+'-'+v_state.segment_seq[-1][3:]
            veh_current_lane = constants.LANE_MAP[dir_key]
        else:
            veh_current_lane = veh_direction
        l_v_state.set_current_lane(veh_current_lane)
        
        trajectory_plan = motion_planner.TrajectoryPlan(None,'NORMAL',constants.TASK_MAP[l_v_direction])
        trajectory_plan.set_lead_vehicle(None)
        l3_actions = trajectory_plan.generate_trajectory(l_v_state)
        return l3_actions
        





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
    #print(list(ex_b_positions.keys()))
    veh_dist_to_segment_exit = (math.hypot(ex_b_positions[current_segment][0][0] - veh_pos_x, ex_b_positions[current_segment][1][0] - veh_pos_y) + \
                                math.hypot(ex_b_positions[current_segment][0][1] - veh_pos_x, ex_b_positions[current_segment][1][1] - veh_pos_y))/2
    
    ''' find the vehicles that are in the current segment or the next and appears within the window of the subject vehicle '''
    if next_segment[-2] == '-':
        q_string = "SELECT T.TRACK_ID FROM TRAJECTORY_MOVEMENTS T, v_TIMES V WHERE (T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+current_segment+"''%' OR T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+next_segment[:-1]+"%') AND T.TRACK_ID = V.TRACK_ID AND (V.ENTRY_TIME <= "+str(curr_time)+" AND V.EXIT_TIME >= "+str(curr_time)+") AND T.TRACK_ID <> "+str(veh_id)
    else:
        q_string = "SELECT T.TRACK_ID FROM TRAJECTORY_MOVEMENTS T, v_TIMES V WHERE (T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+current_segment+"''%' OR T.TRAFFIC_SEGMENT_SEQ LIKE '%''"+next_segment+"''%') AND T.TRACK_ID = V.TRACK_ID AND (V.ENTRY_TIME <= "+str(curr_time)+" AND V.EXIT_TIME >= "+str(curr_time)+") AND T.TRACK_ID <> "+str(veh_id)
    c.execute(q_string)
    res = c.fetchall()
    potential_lead_vehicles = []
    if len(res) > 0:
        for row in res:
            leading_vehicle_id = row[0]
            ''' find the position of the potential lead vehicle in the current time '''
            q_string = "select * from trajectories_0769,trajectories_0769_ext where trajectories_0769.track_id=trajectories_0769_ext.track_id and trajectories_0769.time=trajectories_0769_ext.time and trajectories_0769.track_id="+str(leading_vehicle_id)+" and trajectories_0769.time = "+str(curr_time)
            c.execute(q_string)
            pt_res = c.fetchone()
            l_v_state = VehicleState()
            if pt_res is None:
                ''' this means that there is no entry for this vehicle in trajectories_0769_ext yet'''
                continue
            l_v_state.set_id(pt_res[0])
            l_v_state.set_current_time(curr_time)
            l_v_track = utils.get_track(l_v_state,curr_time)
            l_v_state.set_track_info(l_v_track[0,])
            l_v_track_segment_seq = utils.get_track_segment_seq(l_v_state.id)
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

def get_l3_action_file(file_id,agent_id,relev_agent_id, curr_time, l1_action,l2_action):
    _t = float(curr_time)
    _t1 = _t * 1000
    _t2 = round(_t1)
    curr_time = round(float(curr_time)*1000)
    file_id = '769'
    agent_id = str(agent_id).zfill(3)
    relev_agent_id = str(relev_agent_id).zfill(3)
    l1_action = str(constants.L1_ACTION_CODES[l1_action]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[l2_action]).zfill(2)
    file_key = file_id+agent_id+relev_agent_id+l1_action+l2_action+'_'+str(curr_time)
    return file_key

def save_get_l3_action_file(file_id,agent_id,relev_agent_id, curr_time, l1_action,l2_action):
    _t = float(curr_time)
    _t1 = _t * 1000
    _t2 = round(_t1)
    str(curr_time).replace('.', ',')
    curr_time = str(curr_time).replace('.', ',')
    file_id = '769'
    agent_id = str(agent_id).zfill(3)
    relev_agent_id = str(relev_agent_id).zfill(3)
    l1_action = str(constants.L1_ACTION_CODES[l1_action]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[l2_action]).zfill(2)
    file_key = file_id+agent_id+relev_agent_id+l1_action+l2_action+'_'+str(curr_time)
    return file_key

    
def generate_action_plans(veh_state,track_info,selected_action=None):
    agent_track = veh_state.track
    agent_id = veh_state.id
    curr_time = float(track_info[6,])
    veh_state.set_current_time(curr_time)
    track_region_seq = veh_state.segment_seq
    veh_state.set_track_info(track_info)
    current_segment = track_info[11,]
    veh_state.set_current_segment(current_segment)
    path,gates,direction = utils.get_path_gates_direction(agent_track[:,8],agent_id)
    veh_state.set_direction(direction)
    traffic_light = utils.get_traffic_signal(curr_time, direction)
    task = constants.TASK_MAP[direction]
    veh_state.set_traffic_light(traffic_light)
    veh_state.set_gates(gates)
    gate_crossing_times = utils.gate_crossing_times(veh_state)
    veh_state.set_gate_crossing_times(gate_crossing_times)
    if current_segment is None:
        print(curr_time,track_info[8,],track_region_seq)
        sys.exit('no current segment found')
    agent_loc = (track_info[1,],track_info[2,])
    veh_state.set_current_time(curr_time)
    veh_state.set_current_segment(current_segment)
    veh_direction = 'L_'+track_region_seq[0][3].upper()+'_'+track_region_seq[-1][3].upper()
    if current_segment[0:2] == 'ln':
        veh_current_lane = current_segment
    elif 'int-entry' in current_segment:
        dir_key = str(track_region_seq[0][3:])+'-'+track_region_seq[-1][3:]
        veh_current_lane = constants.LANE_MAP[dir_key]
    else:
        veh_current_lane = veh_direction
    veh_state.set_current_lane(veh_current_lane)
        
    if curr_time not in veh_state.action_plans:
        veh_state.action_plans[curr_time] = dict()
    sub_v_lead_vehicle = get_leading_vehicles(veh_state)
    veh_state.set_leading_vehicle(sub_v_lead_vehicle)
    if selected_action is None:
        actions = utils.get_actions(veh_state)
    else:
        actions = selected_action[0]
    actions_l1 = list(actions.keys())
    print('curr_time',curr_time,'subject veh:',agent_id,'leading vehicle:',sub_v_lead_vehicle.id if sub_v_lead_vehicle is not None else 'None')
    l3_acts_for_plot = []
    for l1 in actions_l1:
        actions_l2 = actions[l1]
        for l2 in actions_l2:
            if l1 not in veh_state.action_plans[curr_time]:
                veh_state.action_plans[curr_time][l1] = dict()
            veh_state.action_plans[curr_time][l1][l2] = None
            trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,task)
            trajectory_plan.set_lead_vehicle(sub_v_lead_vehicle)
            ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
            file_key = constants.L3_ACTION_CACHE+get_l3_action_file(None, agent_id, 0, curr_time, l1, l2)
            if not os.path.isfile(file_key):
                l3_actions = trajectory_plan.generate_trajectory(veh_state)
                if len(l3_actions) > 0:
                    utils.pickle_dump(file_key, l3_actions)
                else:
                    utils.pickle_dump(file_key, dict())
                print('loaded from cache: False')
            else:
                print('loaded from cache: True')
                l3_actions = utils.pickle_load(file_key)
                if len(l3_actions) == 0:
                    l3_actions = trajectory_plan.generate_trajectory(veh_state)
                    if len(l3_actions) > 0:
                        utils.pickle_dump(file_key, l3_actions)
                    else:
                        utils.pickle_dump(file_key, dict())
                else:
                    brk = 1   
            #veh_state.action_plans[curr_time][l1][l2] = np.copy(l3_actions)
            print('time',curr_time,'agent',agent_id,l1,l2)
    #visualizer.plot_all_paths(veh_state)
    l3_acts_for_plot = []
    ''' find the relevant agents for the current subject agent. '''
    if selected_action is not None and selected_action[1] is not None:
        relev_agents = list(selected_action[1].keys())
    else:
        relev_agents = utils.get_relevant_agents(veh_state)
        if sub_v_lead_vehicle is not None:
            relev_agents.append(veh_state.leading_vehicle.id)
    print('relev agents',relev_agents)
    for r_a in relev_agents:
        if r_a == agent_id:
            continue
        r_a_state = motion_planner.VehicleState()
        r_a_state.set_id(r_a)
        r_a_state.set_current_time(curr_time)
        r_a_track = utils.get_track(r_a_state,curr_time)
        r_a_track_segment_seq = utils.get_track_segment_seq(r_a)
        r_a_state.set_segment_seq(r_a_track_segment_seq)
        r_a_state.action_plans = dict()
        r_a_state.set_current_time(curr_time)
        entry_exit_time = utils.get_entry_exit_time(r_a_state.id)
        r_a_state.set_entry_exit_time(entry_exit_time)
        if curr_time not in r_a_state.action_plans:
            r_a_state.action_plans[curr_time] = dict()
    
        if len(r_a_track) == 0:
            ''' this agent is out of the view currently'''
            r_a_state.set_out_of_view(True)
            r_a_track = None
        else:
            r_a_state.set_out_of_view(False)
            r_a_state.set_track_info(r_a_track[0,])
            r_a_track = r_a_track[0,]
        
        if r_a_state.out_of_view or r_a_track[11] is None:
            r_a_track_info = utils.guess_track_info(r_a_state,r_a_track)
            r_a_state.set_track_info(r_a_track_info)
            r_a_track_region = r_a_track_info[8,]
            if r_a_track_region is None:
                sys.exit('need to guess traffic region for relev agent')
            r_a_current_segment = utils.get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,curr_time)
        else:
            r_a_current_segment = r_a_track[11]
        r_a_state.set_current_segment(r_a_current_segment)
        ''' 
        r_a_current_segment = r_a_track[0,11]
        r_a_state.set_current_segment(r_a_current_segment)
        #for now we will only take into account the leading vehicles of the subject agent's relevant vehicles when constructing the possible actions.'''
        lead_vehicle = get_leading_vehicles(r_a_state)
        r_a_state.set_leading_vehicle(lead_vehicle)
        r_a_direction = 'L_'+r_a_track_segment_seq[0][3].upper()+'_'+r_a_track_segment_seq[-1][3].upper()
        r_a_traffic_light = utils.get_traffic_signal(curr_time, r_a_direction)
        r_a_state.set_traffic_light(r_a_traffic_light)
        r_a_state.set_direction(r_a_direction)
        r_a_current_lane = None
        if r_a_current_segment[0:2] == 'ln':
            r_a_current_lane = r_a_current_segment
        elif 'int-entry' in r_a_current_segment:
            dir_key = str(r_a_track_segment_seq[0][3:])+'-'+r_a_track_segment_seq[-1][3:]
            r_a_current_lane = constants.LANE_MAP[dir_key]
        else:
            r_a_current_lane = r_a_direction
        r_a_state.set_current_lane(r_a_current_lane)
        r_a_task = constants.TASK_MAP[r_a_direction]
        if selected_action is None:
            r_a_actions = utils.get_actions(r_a_state)
        else:
            r_a_actions = selected_action[1][r_a]
        r_a_actions_l1 = list(r_a_actions.keys())
        for l1 in r_a_actions_l1:
            actions_l2 = r_a_actions[l1]
            for l2 in actions_l2:
                ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
                file_key = constants.L3_ACTION_CACHE+get_l3_action_file(None, agent_id, r_a_state.id, curr_time, l1, l2)
                if l1 not in r_a_state.action_plans[curr_time]:
                    r_a_state.action_plans[curr_time][l1] = dict()
                    r_a_state.action_plans[curr_time][l1][l2] = None
                trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,r_a_task)
                trajectory_plan.set_lead_vehicle(lead_vehicle)
                print('time',curr_time,'agent',agent_id,'relev agent',r_a_state.id,l1,l2)
                if not os.path.isfile(file_key):
                    l3_actions = trajectory_plan.generate_trajectory(r_a_state)
                    if len(l3_actions) > 0:
                        utils.pickle_dump(file_key, l3_actions)
                    else:
                        utils.pickle_dump(file_key, dict())
                    print('loaded from cache: False')
                else:
                    #l3_actions = utils.pickle_load(file_key)
                    print('loaded from cache: True')
                    l3_actions = utils.pickle_load(file_key)
                    if len(l3_actions) == 0:
                        l3_actions = trajectory_plan.generate_trajectory(r_a_state)
                        if len(l3_actions) > 0:
                            utils.pickle_dump(file_key, l3_actions)
                        else:
                            utils.pickle_dump(file_key, dict())
                    else:
                        brk=1
                    
                #l3_action_size = l3_actions.shape[0] if l3_actions is not None else 0
                #r_a_state.action_plans[curr_time][l1][l2] = np.copy(l3_actions)
        #visualizer.plot_all_paths(r_a_state)
        if 'relev_agents' not in veh_state.action_plans[curr_time]:
            veh_state.action_plans[curr_time]['relev_agents'] = [r_a_state]
        else:
            veh_state.action_plans[curr_time]['relev_agents'].append(r_a_state)
    #print(relev_agents)
    return veh_state
    


''' this method generate plans for a vehicle from the start of its real trajectory time to the end of its trajectory
and stores trajectory plans in L3_ACTION_CACHE. It's called hopping because at every time interval the state hops back
to the real state in the real trajectory, and does not go on a counterfactual path. '''
def generate_hopping_plans(state_dicts=None):
    if state_dicts is not None:
        agent_ids = state_dicts.keys()
    else:
        agent_ids = utils.get_agents_for_task('S_W')
    for agent_id in agent_ids: 
        ''' veh_state object maintains details about an agent'''
        veh_state = motion_planner.VehicleState()
        veh_state.set_id(agent_id)
        veh_state.set_current_time(None)
        
        ''' find the sequence of segments of the agent. This defines its path. '''
        track_region_seq = utils.get_track_segment_seq(veh_state.id)
        veh_state.set_segment_seq(track_region_seq)
        
        ''' get the agent's trajectory'''
        agent_track = utils.get_track(veh_state,None)
        veh_state.set_full_track(agent_track)
        veh_state.set_entry_exit_time((float(agent_track[0][6]), float(agent_track[-1][6])))
        veh_state.action_plans = dict()
        veh_state.relev_agents = []
        
        ''' we will build the plans with actions @ 1Hz'''
        timestamp_l = []
        if state_dicts is not None:
            selected_time_ts = list(state_dicts[agent_id].keys())
            for ts in selected_time_ts:
                track_info = utils.get_track(veh_state,ts)[0,]
                curr_time = ts
                timestamp_l.append(curr_time)
                selected_action = (state_dicts[agent_id][ts]['action'],state_dicts[agent_id][ts]['relev_agents'])
                generate_action_plans(veh_state,track_info,selected_action)
        else:
            selected_time_ts = np.arange(0,len(agent_track),constants.DATASET_FPS*constants.PLAN_FREQ)
            for i in selected_time_ts:
                track_info = agent_track[i]
                curr_time = float(track_info[6,])
                timestamp_l.append(curr_time)
                generate_action_plans(veh_state,track_info)


''' this cleans up the hopping plan generation process'''
def finalize_hopping_plans():
    if not os.path.isfile(constants.L3_ACTION_CACHE+'zero_len_trajectories.dict'):
        dir = constants.L3_ACTION_CACHE
        N,ct = 0,0
        state_dict = dict()
        for f in listdir(dir):
            traj = utils.pickle_load(os.path.join(dir, f))
            N += 1
            if len(traj) == 0:
                agent_id = int(f[3:6])
                time_ts = round(float(f.split('_')[-1])/1000,3)
                relev_agent = None if f[6:9] == '000' else int(f[6:9])
                v = VehicleState()
                if relev_agent is None:
                    v.set_id(agent_id)
                    time_tuple = utils.get_entry_exit_time(agent_id)
                else:
                    v.set_id(relev_agent)
                    time_tuple = utils.get_entry_exit_time(relev_agent)
                agent_track = utils.get_track(v, None)
                if time_ts < time_tuple[0] or time_ts > time_tuple[1]:
                    #print('action outside of view',agent_id,relev_agent,f.split('_')[-1],time_ts)
                    continue 
                if agent_id == 58 and relev_agent is None:
                    brk=1
                all_times = [float(agent_track[i][6,]) for i in np.arange(len(agent_track))]
                real_ts = -1
                for ts in all_times:
                    if ts==104.437667:
                        brk = 1
                    if round(float(ts)*1000) == time_ts*1000:
                        real_ts = ts
                        break
                if real_ts == -1:
                    track = utils.get_track(v, time_ts)
                else:
                    track = utils.get_track(v, real_ts)
                if len(track) < 1:
                    ct += 1
                    print('cant find track for',agent_id,relev_agent,f.split('_')[-1],time_ts,real_ts)
                    continue
                time_ts = real_ts
                l1_action = [k for k,v in constants.L1_ACTION_CODES.items() if v == int(f[9:11])][0]
                l2_action = [k for k,v in constants.L2_ACTION_CODES.items() if v == int(f[11:13])][0]
                print(agent_id,time_ts,relev_agent,l1_action,l2_action)
                if agent_id not in state_dict:
                    state_dict[agent_id] = dict()
                if time_ts not in state_dict[agent_id]:
                    state_dict[agent_id][time_ts] = dict()
                if 'action' not in state_dict[agent_id][time_ts]:
                    state_dict[agent_id][time_ts]['action'] = dict()
                if 'relev_agents' not in state_dict[agent_id][time_ts]:
                    state_dict[agent_id][time_ts]['relev_agents'] = dict()    
                if relev_agent is None:
                    if l1_action not in state_dict[agent_id][time_ts]['action']:
                        state_dict[agent_id][time_ts]['action'][l1_action] = [l2_action]
                    else:
                        state_dict[agent_id][time_ts]['action'][l1_action].append(l2_action)
                else:
                    if relev_agent not in state_dict[agent_id][time_ts]['relev_agents']:
                        state_dict[agent_id][time_ts]['relev_agents'][relev_agent] = dict()
                        state_dict[agent_id][time_ts]['relev_agents'][relev_agent][l1_action] = [l2_action]
                    else:
                        if l1_action not in state_dict[agent_id][time_ts]['relev_agents'][relev_agent]:
                            state_dict[agent_id][time_ts]['relev_agents'][relev_agent][l1_action] = [l2_action]
                        else:
                            state_dict[agent_id][time_ts]['relev_agents'][relev_agent][l1_action].append(l2_action)
                
                ct += 1
        print(ct,'/',N)
        utils.pickle_dump(constants.L3_ACTION_CACHE+'zero_len_trajectories.dict', state_dict)
    else:
        state_dict = utils.pickle_load(constants.L3_ACTION_CACHE+'zero_len_trajectories.dict')
        generate_hopping_plans(state_dict)
    


def get_traj_metadata():
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
                l2_action = [k for k,v in constants.L2_ACTION_CODES.items() if v == int(f[11:13])][0]
                lead_vehicle = None
                v = VehicleState()
                if relev_agent is None:
                    v.set_id(agent_id)
                else:
                    v.set_id(relev_agent)
                time_tuple = utils.get_entry_exit_time(v.id)
                v.set_entry_exit_time(time_tuple)
                agent_track = utils.get_track(v, None)
                all_times = [float(agent_track[i][6,]) for i in np.arange(len(agent_track))]
                real_ts = -1
                for ts in all_times:
                    if round(float(ts)*1000) == time_ts*1000:
                        real_ts = ts
                        break
                if real_ts != -1:
                    time_ts = real_ts
                v.set_current_time(time_ts)
                '''
                if l1_action == 'track_speed':
                    track_region_seq = utils.get_track_segment_seq(v.id)
                    v.set_segment_seq(track_region_seq)
                    if real_ts == -1:
                        track = utils.get_track(v, time_ts)
                    else:
                        track = utils.get_track(v, real_ts)
                    if len(track) > 0:
                        v.set_track_info(track[0,])
                        current_segment = track[0,][11,]
                    else:
                        r_a_track_info = utils.guess_track_info(v,None)
                        v.set_track_info(r_a_track_info)
                        r_a_track_region = r_a_track_info[8,]
                        if r_a_track_region is None:
                            sys.exit('need to guess traffic region for relev agent')
                        current_segment = utils.get_current_segment(v,r_a_track_region,track_region_seq,time_ts)
                    
                    v.set_current_segment(current_segment)
                    v.set_current_time(time_ts)
                    try:
                        lead_vehicle = get_leading_vehicles(v)
                    except ValueError:
                        brk=1
                        raise
                    if lead_vehicle is not None:
                        l1_action = 'follow_lead'
                        f_new = str(f[0:9]+str(constants.L1_ACTION_CODES[l1_action]).zfill(2)+f[11:])
                        os.rename(os.path.join(dir,f),os.path.join(dir,f_new))
                '''
                print(ct,'/',N,agent_id,time_ts,relev_agent,l1_action,l2_action)
                if time_ts not in state_dict:
                    state_dict[time_ts] = dict()
                if agent_id not in state_dict[time_ts]:
                    state_dict[time_ts][agent_id] = dict()
                if relev_agent is None:
                    if l1_action not in state_dict[time_ts][agent_id]:
                        state_dict[time_ts][agent_id][l1_action] = [l2_action]
                    else:
                        state_dict[time_ts][agent_id][l1_action].append(l2_action)
                else:
                    if 'relev_agents' not in state_dict[time_ts]: 
                        state_dict[time_ts]['relev_agents'] = dict()
                    if relev_agent not in state_dict[time_ts]['relev_agents']:
                        state_dict[time_ts]['relev_agents'][relev_agent] = dict()
                    if l1_action not in state_dict[time_ts]['relev_agents'][relev_agent]:
                        state_dict[time_ts]['relev_agents'][relev_agent][l1_action] = [l2_action]
                    else:
                        state_dict[time_ts]['relev_agents'][relev_agent][l1_action].append(l2_action)
        state_dict = OrderedDict(sorted((float(key), value) for key, value in state_dict.items()))       
        utils.pickle_dump(constants.L3_ACTION_CACHE+'traj_metadata.dict', state_dict)
    else:
        state_dict = utils.pickle_load(constants.L3_ACTION_CACHE+'traj_metadata.dict')
    return state_dict

    
    
''' this method calculates the equilibria based on the trajectories that were generated in the hopping plans.
Assumes that the trajectories are already present in the cache.'''
def calc_eqs_for_hopping_trajectories():
    traj_metadata = get_traj_metadata()
    show_plots = True
    step_size_secs = constants.PLAN_FREQ
    for k,v in traj_metadata.items():
        print('------time:',k)
        eq_list = cost_evaluation.calc_equilibria(constants.L3_ACTION_CACHE,k,v,'mean')
        if show_plots:
            for eq_idx, eq_info in enumerate(eq_list):
                eqs, traj_idx, eq_payoffs = eq_info[0], eq_info[1], eq_info[2]
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                for _ag_idx,act in enumerate(eqs):
                    agent_id = int(act[6:9]) if int(act[6:9])!=0 else int(act[3:6]) 
                    file_key = constants.L3_ACTION_CACHE+act
                    traj = utils.pickle_load(file_key)
                    #visualizer.plot_all_trajectories(traj,ax1,ax2,ax3)
                    plan_horizon_slice = step_size_secs * int(constants.PLAN_FREQ / constants.LP_FREQ)
                    ''' we can cut the slice from 0 since the trajectory begins at the given timestamp (i) and not from the beginning of the scene '''
                    traj_slice = traj[int(traj_idx[_ag_idx])][0][:,:plan_horizon_slice]
                    traj_vels = traj_slice[4,:] 
                    '''get the timestamp in ms and convert it to seconds'''
                    horizon_start = int(act.split('_')[1])/1000
                    horizon_end = step_size_secs + (int(act.split('_')[1])+step_size_secs)/1000
                    traj_vels_times = [horizon_start + x for x in traj_slice[0,:]]
                    #visualizer.plot_velocity(list(zip(traj_vels_times,traj_vels)),agent_id,(horizon_start,horizon_end),_ag_idx,ax4)
                plt.show()
                plt.clf()


#calc_eqs_for_hopping_trajectories()