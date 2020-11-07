'''
Created on Jan 29, 2020

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3

from all_utils import utils
import sys
import constants
import motion_planners
from motion_planners import motion_planner
import ast
import math
import equilibria
from equilibria import cost_evaluation
import pickle
import os.path
from motion_planners.planning_objects import VehicleState,SceneState
import matplotlib.pyplot as plt
import visualizer
from os import listdir
from collections import OrderedDict
from all_utils.thread_utils import CustomMPS,CustomThreaExs,ThreadSafeObject
log = constants.common_logger





def setup_lead_vehicle(v_state,from_ra):
    from all_utils import utils
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
    file_id = constants.CURRENT_FILE_ID
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
    file_id = constants.CURRENT_FILE_ID
    agent_id = str(agent_id).zfill(3)
    relev_agent_id = str(relev_agent_id).zfill(3)
    l1_action = str(constants.L1_ACTION_CODES[l1_action]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[l2_action]).zfill(2)
    file_key = file_id+agent_id+relev_agent_id+l1_action+l2_action+'_'+str(time_ts)
    return file_key


def push_trajectories_to_db():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_generated_trajectories_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    if not constants.TRAJECTORY_TYPE == 'GAUSSIAN':
        ''' Baseline and boundary regenerates the generated_info, so we can clear this.'''
        traj_info_dict = None
        c.execute("DELETE FROM GENERATED_TRAJECTORY_INFO")
        conn.commit()
        ''' Boundary inserts Baseline data too.'''
        if constants.TRAJECTORY_TYPE == 'BOUNDARY':
            c.execute("DELETE FROM GENERATED_BASELINE_TRAJECTORY")
            conn.commit()
    else:
        c.execute("SELECT * FROM GENERATED_TRAJECTORY_INFO")
        res = c.fetchall()
        traj_info_dict = {(row[2],row[3],row[4],row[5],row[6]):row[1] for row in res}
    all_files = os.listdir(os.path.join(constants.CACHE_DIR, constants.TEMP_TRAJ_CACHE))
    all_files.sort()
    N = len(all_files)
    boundary_ins_list,baseline_ins_list,gaussian_ins_list = [],[],[]
    bounary_max_tid,baseline_max_tid,gaussian_tid = None,None,None
    traj_info_id = 1
    traj_info_ins_list = []
    for idx,file in enumerate(all_files):
        log.info(str(idx)+'/'+str(N))
        filename = os.path.join(constants.CACHE_DIR,constants.TEMP_TRAJ_CACHE,file)
        traj_plan = utils.pickle_load(filename)
        if not constants.TRAJECTORY_TYPE == 'GAUSSIAN':
            i_string_boundary = 'INSERT INTO '+'GENERATED_BOUNDARY_TRAJECTORY'+' VALUES (?,?,?,?,?,?,?,?,?)'
            i_string_baseline = 'INSERT INTO '+'GENERATED_BASELINE_TRAJECTORY'+' VALUES (?,?,?,?,?,?,?,?,?)'
            print(traj_plan.veh_state.l1_action,len(traj_plan.trajectories['baseline']), len(traj_plan.trajectories['boundary']))
            ins_list,new_max_trajid,traj_info_ins_tup = traj_plan.insert_baseline_trajectory('GENERATED_BOUNDARY_TRAJECTORY',traj_info_id,None,bounary_max_tid)
            boundary_ins_list.extend(ins_list)
            bounary_max_tid = new_max_trajid
            ins_list,new_max_trajid,traj_info_ins_tup = traj_plan.insert_baseline_trajectory('GENERATED_BASELINE_TRAJECTORY',traj_info_id,None,baseline_max_tid)
            traj_info_id += 1
            baseline_ins_list.extend(ins_list)
            baseline_max_tid = new_max_trajid
            traj_info_ins_list.append(traj_info_ins_tup)
            if len(boundary_ins_list) > 500000:
                c.executemany(i_string_boundary,boundary_ins_list)
                conn.commit()
                boundary_ins_list = []
            if len(baseline_ins_list) > 500000:
                c.executemany(i_string_baseline,baseline_ins_list)
                conn.commit()
                baseline_ins_list = []
                
        else:
            i_string = 'INSERT INTO '+'GENERATED_GAUSSIAN_TRAJECTORY'+' VALUES (?,?,?,?,?,?,?,?,?)'
            if 'gaussian' in traj_plan.trajectories:
                print(traj_plan.veh_state.l1_action,len(traj_plan.trajectories['gaussian']))
                ins_list,new_max_trajid,traj_info_ins_tup = traj_plan.insert_baseline_trajectory('GENERATED_GAUSSIAN_TRAJECTORY',None,traj_info_dict,gaussian_tid)
                traj_info_ins_list.append(traj_info_ins_tup)
                gaussian_ins_list.extend(ins_list)
                gaussian_tid = new_max_trajid
            if len(gaussian_ins_list) > 500000:
                c.executemany(i_string,gaussian_ins_list)
                conn.commit()
                gaussian_ins_list = []
    if not constants.TRAJECTORY_TYPE == 'GAUSSIAN':
        c.executemany(i_string_boundary,boundary_ins_list)
        conn.commit()
        c.executemany(i_string_baseline,baseline_ins_list)
        conn.commit()
        f=1
    else:
        c.executemany(i_string,gaussian_ins_list)
        conn.commit()
        f=1
    if not constants.TRAJECTORY_TYPE == 'GAUSSIAN':
        for t in traj_info_ins_list:
            print(t)
        c.executemany('INSERT INTO GENERATED_TRAJECTORY_INFO VALUES (?,?,?,?,?,?,?,?)',traj_info_ins_list)
        conn.commit()
    conn.close()
        
                
        
        



    
def generate_action_plans(param_list):
    from all_utils import utils
    import copy
    veh_state,track_info,selected_action,trajs_in_db = param_list[0], param_list[1], param_list[2], param_list[3] 
    veh_state = copy.deepcopy(veh_state)
    agent_track = veh_state.track
    agent_id = veh_state.id
    time_ts = float(track_info[6,])
    if time_ts > 75:
        brk=1
    pedestrian_info = utils.setup_pedestrian_info(time_ts)
    vehicles_info = utils.get_vehicles_info(time_ts)
    scene_state = SceneState(pedestrian_info,vehicles_info)
    ag_file_key = os.path.join(constants.CACHE_DIR,constants.L3_ACTION_CACHE,str(veh_state.id)+'-0_'+str(time_ts).replace('.', ','))
    if os.path.exists(ag_file_key):
        veh_state = utils.pickle_load(ag_file_key)
        veh_state = copy.deepcopy(veh_state)
        task = veh_state.task
        sub_v_lead_vehicle = veh_state.leading_vehicle
        
    else:
        veh_state.set_current_time(time_ts)
        try:
            #file_key.split('_')[-1].replace(',','.')
            assert str(time_ts)==str(veh_state.current_time), str(time_ts)+ '!=' + str(veh_state.current_time)
        except AssertionError:
            brk=1
            raise
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
        try:
            #file_key.split('_')[-1].replace(',','.')
            assert str(time_ts)==str(veh_state.current_time), str(time_ts)+ '!=' + str(veh_state.current_time)
        except AssertionError:
            brk=1
            raise
        
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
        utils.pickle_dump(constants.L3_ACTION_CACHE,str(veh_state.id)+'-0_'+str(time_ts).replace('.', ','), veh_state)
        try:
            #file_key.split('_')[-1].replace(',','.')
            assert str(time_ts)==str(veh_state.current_time), str(time_ts)+ '!=' + str(veh_state.current_time)
        except AssertionError:
            brk=1
            raise
        
    if selected_action is None:
        actions = utils.get_actions(veh_state)
    else:
        actions = selected_action[0]
    actions_l1 = list(actions.keys())
    veh_state.set_scene_state(scene_state)
    print('time_ts',time_ts,'subject veh:',agent_id,'leading vehicle:',sub_v_lead_vehicle.id if sub_v_lead_vehicle is not None else 'None')
    l3_acts_for_plot = []
    for l1 in actions_l1:
        actions_l2 = actions[l1]
        for l2 in actions_l2:
            print('time',time_ts,'agent',agent_id,l1,l2)
            try:
                #file_key.split('_')[-1].replace(',','.')
                assert str(time_ts)==str(veh_state.current_time), str(time_ts)+ '!=' + str(veh_state.current_time)
            except AssertionError:
                brk=1
                raise
            
            if l1 not in veh_state.action_plans[time_ts]:
                veh_state.action_plans[time_ts][l1] = dict()
            veh_state.action_plans[time_ts][l1][l2] = None
            generate_boundary = False if constants.BASELINE_TRAJECTORIES_ONLY else True
            veh_state.set_current_l1_action(l1)
            veh_state.set_current_l2_action(l2)
            file_key = get_l3_action_file(None, agent_id, 0, time_ts, l1, l2)
            try:
                #file_key.split('_')[-1].replace(',','.')
                assert str(time_ts)==str(veh_state.current_time), str(time_ts)+ '!=' + str(veh_state.current_time)
            except AssertionError:
                brk=1
                raise
             
            try:
                #file_key.split('_')[-1].replace(',','.')
                assert file_key.split('_')[-1].replace(',','.')==str(veh_state.current_time), file_key.split('_')[-1].replace(',','.')+ '!=' + str(veh_state.current_time)
            except AssertionError:
                brk=1
                raise
                
            trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,task,constants.TRAJECTORY_TYPE,veh_state,agent_id,0)
            trajectory_plan.set_lead_vehicle(sub_v_lead_vehicle)
            ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
            file_key = get_l3_action_file(None, agent_id, 0, time_ts, l1, l2)
            if file_key not in trajs_in_db:# and (l1=='decelerate-to-stop' or l1=='wait_for_lead_to_cross'):
                print('loaded from cache: False')
                trajectory_plan.generate()
                try:
                    #file_key.split('_')[-1].replace(',','.')
                    assert file_key.split('_')[-1].replace(',','.')==str(veh_state.current_time), file_key.split('_')[-1].replace(',','.')+ '!=' + str(veh_state.current_time)
                except AssertionError:
                    brk=1
                    raise
                try:
                    #file_key.split('_')[-1].replace(',','.')
                    assert file_key.split('_')[-1].replace(',','.')==str(trajectory_plan.veh_state.current_time), file_key.split('_')[-1].replace(',','.')+ '!=' + str(trajectory_plan.veh_state.current_time)
                except AssertionError:
                    brk=1
                    raise
                
                utils.pickle_dump(constants.TEMP_TRAJ_CACHE,file_key,trajectory_plan)
                #trajectory_plan.insert_baseline_trajectory()
                f=1
                #utils.insert_generated_trajectory(l3_actions, file_key)
                
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
        ra_file_key = constants.L3_ACTION_CACHE + str(veh_state.id)+'-'+str(r_a)+'_'+str(time_ts).replace('.', ',')
        if os.path.exists(ra_file_key):
            r_a_state = utils.pickle_load(ra_file_key)
            lead_vehicle = r_a_state.leading_vehicle
            r_a_task = r_a_state.task
        else:
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
            utils.pickle_dump(constants.L3_ACTION_CACHE,str(veh_state.id)+'-'+str(r_a_state.id)+'_'+str(time_ts).replace('.', ','),r_a_state)
        ''' check if this relevant vehicle can be excluded '''
        dist_to_sv = math.hypot(veh_state.x-r_a_state.x,veh_state.y-r_a_state.y)
        veh_state.set_dist_to_sub_agent(0.0)
        r_a_state.set_dist_to_sub_agent(dist_to_sv)
        
        if utils.can_exclude(veh_state,r_a_state):
            continue
        
        if selected_action is None:
            r_a_actions = utils.get_actions(r_a_state)
        else:
            r_a_actions = selected_action[1][r_a]
        r_a_actions_l1 = list(r_a_actions.keys())
        r_a_state.set_scene_state(scene_state)
        for l1 in r_a_actions_l1:
            actions_l2 = r_a_actions[l1]
            for l2 in actions_l2:
                ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
                file_key = get_l3_action_file(None, agent_id, r_a_state.id, time_ts, l1, l2)
                if l1 not in r_a_state.action_plans[time_ts]:
                    r_a_state.action_plans[time_ts][l1] = dict()
                    r_a_state.action_plans[time_ts][l1][l2] = None
                generate_boundary = False if constants.BASELINE_TRAJECTORIES_ONLY else True
                r_a_state.set_current_l1_action(l1)
                r_a_state.set_current_l2_action(l2)
            
                trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,r_a_task,constants.TRAJECTORY_TYPE,r_a_state,agent_id,r_a_state.id)
                trajectory_plan.set_lead_vehicle(lead_vehicle)
                print('time',time_ts,'agent',agent_id,'relev agent',r_a_state.id,l1,l2)
                if file_key not in trajs_in_db:# and (l1=='decelerate-to-stop' or l1=='wait_for_lead_to_cross'):
                    print('loaded from cache: False')
                    trajectory_plan.generate()
                    #trajectory_plan.insert_baseline_trajectory()
                    utils.pickle_dump(constants.TEMP_TRAJ_CACHE,file_key,trajectory_plan)
                    #utils.insert_generated_trajectory(l3_actions, file_key)
                    
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
def generate_hopping_plans():
    from all_utils import utils
    skip_existing = True
    task_list = ['S_E','S_W','N_E','W_S','W_N','E_S']
    agent_ids = []
    if skip_existing:
        #trajs_in_db = utils.get_trajectories_in_db() if not constants.BASELINE_TRAJECTORIES_ONLY else utils.get_baseline_trajectories_in_db()
        trajs_in_db = utils.get_processed_files(constants.TEMP_TRAJ_CACHE)
    else:
        trajs_in_db = dict()
    for t in task_list:
        agent_ids = agent_ids + [int(x) for x in utils.get_agents_for_task(t)] 
    agent_ids.sort()
    all_params = []
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
        try:
            veh_state.set_entry_exit_time((float(agent_track[0][6]), float(agent_track[-1][6])))
        except IndexError:
            brk = 1
        veh_state.action_plans = dict()
        veh_state.relev_agents = []
        
        ''' we will build the plans with actions @ 1Hz'''
        timestamp_l = []
        selected_time_ts = np.arange(0,len(agent_track),constants.DATASET_FPS*constants.PLAN_FREQ)
        for i in selected_time_ts:
            track_info = agent_track[i]
            time_ts = float(track_info[6,])
            timestamp_l.append(time_ts)
            all_params.append([veh_state,track_info,None,trajs_in_db])
    
    n=12
    all_params.sort(key=lambda x: x[1][6])
    chunks = [all_params[i:i+n] for i in range(0, len(all_params), n)]
    
    cmp = CustomThreaExs()
    import time
    N = len(chunks)
    for d_idx,chunk in enumerate(chunks):
        done = False
        done = cmp.execute_no_return(generate_action_plans, chunk)
        print('DONE',d_idx,N,done)
    '''
    
    for p in all_params:
        generate_action_plans(p)
    '''

def generate_lattice_boundary_trajectories():
    from all_utils import utils
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select count(*) from GENERATED_BASELINE_TRAJECTORY"
    c.execute(q_string)
    res = c.fetchone()
    if res[0]==0:
        sys.exit('baseline trajectory not generated')
    q_string = "select * from GENERATED_TRAJECTORY_INFO ORDER BY AGENT_ID,TIME,RELEV_AGENT_ID"
    c.execute(q_string)
    res = c.fetchall()
    N_traj_info = len(res)
    for inf_row_idx,row in enumerate(res):
        traj_info_det = {'file_id':row[0],'traj_info_id':row[1],'ag_id':row[2],'ra_id':row[3],'l1_action':row[4],'l2_action':row[5],'time_ts':row[6]}
        log.info(str(inf_row_idx)+'/'+str(N_traj_info)+' running with ' + str(list(traj_info_det.values())))
        curr_ag_id = traj_info_det['ag_id'] if traj_info_det['ra_id']==0 else traj_info_det['ra_id']
        time_ts = traj_info_det['time_ts'] 
        if traj_info_det['time_ts']==0:
            ag_file_key = os.path.join(constants.CACHE_DIR,constants.L3_ACTION_CACHE,str(curr_ag_id)+'-0_'+str(traj_info_det['time_ts'])+',0')
        else:
            ag_file_key = os.path.join(constants.CACHE_DIR,constants.L3_ACTION_CACHE,str(curr_ag_id)+'-0_'+str(traj_info_det['time_ts']).replace('.', ','))
        if curr_ag_id==2 and time_ts==12.012:
            brk=1
        if os.path.exists(ag_file_key):
            log.info('loaded')
            veh_state = utils.pickle_load(ag_file_key)
        else:
            log.info('constructed')
            veh_state = utils.setup_vehicle_state(curr_ag_id, time_ts)
            utils.pickle_dump(str(veh_state.id)+'-0_'+str(time_ts).replace('.', ','), veh_state)
        q_string = "select * from GENERATED_BASELINE_TRAJECTORY WHERE TRAJECTORY_INFO_ID="+str(traj_info_det['traj_info_id'])+' ORDER BY TIME'
        c.execute(q_string)
        baseline_t = c.fetchall()
        trajectory_plan = motion_planner.TrajectoryPlan(traj_info_det['li_action'],traj_info_det['l2_action'],veh_state.task,False,veh_state)
        
        
          
            
        
    

    


    

def main():
    constants.CURRENT_FILE_ID = sys.argv[1]
    constants.TRAJECTORY_TYPE = sys.argv[2]
    constants.TEMP_TRAJ_CACHE = 'temp_traj_cache_'+constants.CURRENT_FILE_ID+'_'+constants.TRAJECTORY_TYPE
    constants.L3_ACTION_CACHE = 'l3_action_trajectories_'+constants.CURRENT_FILE_ID
    constants.setup_logger()
    generate_hopping_plans()
    push_trajectories_to_db()
    #utils.remove_files(constants.TEMP_TRAJ_CACHE)
if __name__ == '__main__':    
    main()