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
        





def get_l3_action_file(file_id,agent_id,relev_agent_id, time_ts, l1_action,l2_action):
    _t = float(time_ts)
    _t1 = _t * 1000
    _t2 = round(_t1)
    #time_ts = round(float(time_ts)*1000)
    file_id = '769'
    agent_id = str(agent_id).zfill(3)
    relev_agent_id = str(relev_agent_id).zfill(3)
    l1_action = str(constants.L1_ACTION_CODES[l1_action]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[l2_action]).zfill(2)
    file_key = file_id+agent_id+relev_agent_id+l1_action+l2_action+'_'+str(time_ts).replace('.',',')
    return file_key

def save_get_l3_action_file(file_id,agent_id,relev_agent_id, time_ts, l1_action,l2_action):
    _t = float(time_ts)
    _t1 = _t * 1000
    _t2 = round(_t1)
    time_ts = str(time_ts).replace('.', ',')
    file_id = '769'
    agent_id = str(agent_id).zfill(3)
    relev_agent_id = str(relev_agent_id).zfill(3)
    l1_action = str(constants.L1_ACTION_CODES[l1_action]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[l2_action]).zfill(2)
    file_key = file_id+agent_id+relev_agent_id+l1_action+l2_action+'_'+str(time_ts)
    return file_key

    
def generate_action_plans(veh_state,track_info,selected_action=None,trajs_in_db=None):
    agent_track = veh_state.track
    agent_id = veh_state.id
    time_ts = float(track_info[6,])
    pedestrian_info = utils.setup_pedestrian_info(time_ts)
    veh_state.set_current_time(time_ts)
    track_region_seq = veh_state.segment_seq
    veh_state.set_track_info(track_info)
    current_segment = track_info[11,]
    veh_state.set_current_segment(current_segment)
    path,gates,direction = utils.get_path_gates_direction(agent_track[:,8],agent_id)
    veh_direction = 'L_'+track_region_seq[0][3].upper()+'_'+track_region_seq[-1][3].upper()
    veh_state.set_direction(veh_direction)
    relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
    veh_state.set_relev_crosswalks(relev_crosswalks)
    traffic_light = utils.get_traffic_signal(time_ts, veh_direction)
    task = constants.TASK_MAP[veh_direction]
    veh_state.set_task(task)
    veh_state.set_traffic_light(traffic_light)
    veh_state.set_gates(gates)
    gate_crossing_times = utils.gate_crossing_times(veh_state)
    veh_state.set_gate_crossing_times(gate_crossing_times)
    if current_segment is None:
        print(time_ts,track_info[8,],track_region_seq)
        sys.exit('no current segment found')
    agent_loc = (track_info[1,],track_info[2,])
    veh_state.set_current_time(time_ts)
    veh_state.set_current_segment(current_segment)
    
    if current_segment[0:2] == 'ln':
        veh_current_lane = current_segment
    elif 'int-entry' in current_segment:
        dir_key = str(track_region_seq[0][3:])+'-'+track_region_seq[-1][3:]
        veh_current_lane = constants.LANE_MAP[dir_key]
    else:
        veh_current_lane = veh_direction
    veh_state.set_current_lane(veh_current_lane)
        
    if time_ts not in veh_state.action_plans:
        veh_state.action_plans[time_ts] = dict()
    sub_v_lead_vehicle = utils.get_leading_vehicles(veh_state)
    veh_state.set_leading_vehicle(sub_v_lead_vehicle)
    merging_vehicle = utils.get_merging_vehicle(veh_state)
    veh_state.set_merging_vehicle(merging_vehicle)
    relev_pedestrians = utils.get_relevant_pedestrians(veh_state, pedestrian_info)
    veh_state.set_relev_pedestrians(relev_pedestrians)
    if selected_action is None:
        actions = utils.get_actions(veh_state)
    else:
        actions = selected_action[0]
    actions_l1 = list(actions.keys())
    print('time_ts',time_ts,'subject veh:',agent_id,'leading vehicle:',sub_v_lead_vehicle.id if sub_v_lead_vehicle is not None else 'None')
    l3_acts_for_plot = []
    for l1 in actions_l1:
        actions_l2 = actions[l1]
        for l2 in actions_l2:
            print('time',time_ts,'agent',agent_id,l1,l2)
            if l1 not in veh_state.action_plans[time_ts]:
                veh_state.action_plans[time_ts][l1] = dict()
            veh_state.action_plans[time_ts][l1][l2] = None
            trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,task,constants.BASELINE_TRAJECTORIES_ONLY)
            trajectory_plan.set_lead_vehicle(sub_v_lead_vehicle)
            ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
            file_key = get_l3_action_file(None, agent_id, 0, time_ts, l1, l2)
            if file_key not in trajs_in_db:# and (l1=='decelerate-to-stop' or l1=='wait_for_lead_to_cross'):
                print('loaded from cache: False')
                
                if not constants.BASELINE_TRAJECTORIES_ONLY:
                    l3_actions = trajectory_plan.generate_trajectory(veh_state)
                    utils.insert_generated_trajectory(l3_actions, file_key)
                else:
                    l3_actions = trajectory_plan.generate_baseline(veh_state)
                    if l3_actions is not None and len(l3_actions) > 0:
                        utils.insert_baseline_trajectory(l3_actions, file_key)
                '''
                if len(l3_actions) > 0:
                    utils.insert_generated_trajectory(l3_actions, file_key)
                else:
                    utils.pickle_dump(constants.L3_ACTION_CACHE+file_key, dict())
                '''
            else:
                print('loaded from cache: True')   
            #veh_state.action_plans[time_ts][l1][l2] = np.copy(l3_actions)
            
    #visualizer.plot_all_paths(veh_state)
    l3_acts_for_plot = []
    ''' find the relevant agents for the current subject agent. '''
    if selected_action is not None and selected_action[1] is not None:
        relev_agents = list(selected_action[1].keys())
    else:
        relev_agents = utils.get_relevant_agents(veh_state)
        if sub_v_lead_vehicle is not None:
            if utils.is_only_leading_relevant(veh_state):
                relev_agents = [veh_state.leading_vehicle.id]
            else:
                relev_agents.append(veh_state.leading_vehicle.id)
    print('relev agents',relev_agents)
    for r_a in relev_agents:
        if r_a == agent_id:
            continue
        if r_a == 4 and time_ts==50.5505:
            brk = 1
        r_a_state = motion_planner.VehicleState()
        r_a_state.set_id(r_a)
        r_a_state.set_current_time(time_ts)
        r_a_track = utils.get_track(r_a_state,time_ts)
        r_a_track_segment_seq = utils.get_track_segment_seq(r_a)
        r_a_state.set_segment_seq(r_a_track_segment_seq)
        r_a_state.action_plans = dict()
        r_a_state.set_current_time(time_ts)
        entry_exit_time = utils.get_entry_exit_time(r_a_state.id)
        r_a_state.set_entry_exit_time(entry_exit_time)
        if time_ts not in r_a_state.action_plans:
            r_a_state.action_plans[time_ts] = dict()
    
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
            if r_a_track_info[1,] is None:
                brk = 1
            r_a_state.set_track_info(r_a_track_info)
            r_a_track_region = r_a_track_info[8,]
            if r_a_track_region is None:
                sys.exit('need to guess traffic region for relev agent')
            r_a_current_segment = utils.get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,time_ts)
        else:
            r_a_current_segment = r_a_track[11]
        
            
        r_a_state.set_current_segment(r_a_current_segment)
        ''' 
        r_a_current_segment = r_a_track[0,11]
        r_a_state.set_current_segment(r_a_current_segment)
        #for now we will only take into account the leading vehicles of the subject agent's relevant vehicles when constructing the possible actions.'''
        lead_vehicle = utils.get_leading_vehicles(r_a_state)
        r_a_state.set_leading_vehicle(lead_vehicle)
        merging_vehicle = utils.get_merging_vehicle(r_a_state)
        r_a_state.set_merging_vehicle(merging_vehicle)
        r_a_direction = 'L_'+r_a_track_segment_seq[0][3].upper()+'_'+r_a_track_segment_seq[-1][3].upper()
        r_a_traffic_light = utils.get_traffic_signal(time_ts, r_a_direction)
        r_a_state.set_traffic_light(r_a_traffic_light)
        r_a_state.set_direction(r_a_direction)
        relev_crosswalks = utils.get_relevant_crosswalks(r_a_state)
        r_a_state.set_relev_crosswalks(relev_crosswalks)
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
        r_a_state.set_task(r_a_task)
        relev_pedestrians = utils.get_relevant_pedestrians(r_a_state, pedestrian_info)
        r_a_state.set_relev_pedestrians(relev_pedestrians)
        ''' check if this relevant vehicle can be excluded '''
        if utils.can_exclude(veh_state,r_a_state):
            continue
        
        if selected_action is None:
            r_a_actions = utils.get_actions(r_a_state)
        else:
            r_a_actions = selected_action[1][r_a]
        r_a_actions_l1 = list(r_a_actions.keys())
        for l1 in r_a_actions_l1:
            actions_l2 = r_a_actions[l1]
            for l2 in actions_l2:
                ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
                file_key = get_l3_action_file(None, agent_id, r_a_state.id, time_ts, l1, l2)
                if l1 not in r_a_state.action_plans[time_ts]:
                    r_a_state.action_plans[time_ts][l1] = dict()
                    r_a_state.action_plans[time_ts][l1][l2] = None
                trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,r_a_task,constants.BASELINE_TRAJECTORIES_ONLY)
                trajectory_plan.set_lead_vehicle(lead_vehicle)
                print('time',time_ts,'agent',agent_id,'relev agent',r_a_state.id,l1,l2)
                if file_key not in trajs_in_db:# and (l1=='decelerate-to-stop' or l1=='wait_for_lead_to_cross'):
                    print('loaded from cache: False')
                    
                    if not constants.BASELINE_TRAJECTORIES_ONLY:
                        l3_actions = trajectory_plan.generate_trajectory(r_a_state)
                        utils.insert_generated_trajectory(l3_actions, file_key)
                    else:
                        l3_actions = trajectory_plan.generate_baseline(r_a_state)
                        if l3_actions is not None and len(l3_actions) > 0:
                            utils.insert_baseline_trajectory(l3_actions, file_key)
                    '''
                    if len(l3_actions) > 0:
                        utils.insert_generated_trajectory(l3_actions, file_key)
                    else:
                        utils.pickle_dump(constants.L3_ACTION_CACHE+file_key, dict())
                    '''
                else:
                    #l3_actions = utils.pickle_load(file_key)
                    print('loaded from cache: True')
                    
                    
                #l3_action_size = l3_actions.shape[0] if l3_actions is not None else 0
                #r_a_state.action_plans[time_ts][l1][l2] = np.copy(l3_actions)
        #visualizer.plot_all_paths(r_a_state)
        if 'relev_agents' not in veh_state.action_plans[time_ts]:
            veh_state.action_plans[time_ts]['relev_agents'] = [r_a_state]
        else:
            veh_state.action_plans[time_ts]['relev_agents'].append(r_a_state)
    
    #print(relev_agents)
    return veh_state
    


''' this method generate plans for a vehicle from the start of its real trajectory time to the end of its trajectory
and stores trajectory plans in L3_ACTION_CACHE. It's called hopping because at every time interval the state hops back
to the real state in the real trajectory, and does not go on a counterfactual path. '''
def generate_hopping_plans(state_dicts=None):
    skip_existing = False
    if skip_existing:
        trajs_in_db = utils.get_trajectories_in_db() if not constants.BASELINE_TRAJECTORIES_ONLY else utils.get_baseline_trajectories_in_db()
    else:
        trajs_in_db = dict()
    if state_dicts is not None:
        agent_ids = state_dicts.keys()
    else:
        agent_ids = utils.get_agents_for_task('N_E') 
    for agent_id in agent_ids: 
        ''' veh_state object maintains details about an agent'''
        veh_state = VehicleState()
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
                time_ts = ts
                timestamp_l.append(time_ts)
                selected_action = (state_dicts[agent_id][ts]['action'],state_dicts[agent_id][ts]['relev_agents'])
                generate_action_plans(veh_state,track_info,selected_action,trajs_in_db)
        else:
            selected_time_ts = np.arange(0,len(agent_track),constants.DATASET_FPS*constants.PLAN_FREQ)
            for i in selected_time_ts:
                track_info = agent_track[i]
                time_ts = float(track_info[6,])
                timestamp_l.append(time_ts)
                generate_action_plans(veh_state,track_info,None,trajs_in_db)



    


    


#generate_hopping_plans()