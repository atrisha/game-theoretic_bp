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
            l_v_track = utils.get_track(l_v_state,curr_time)
            l_v_state.set_track_info(l_v_track[0,])
            if len(l_v_track) == 0 or (len(l_v_track) > 0 and len(l_v_track[0,8]) == 0) :
                l_v_track_region = utils.guess_track_info(l_v_state)[8,]
                if l_v_track_region is None:
                    sys.exit('need to guess traffic region for relev agent')
            else:
                l_v_track_region = l_v_track[0,8]
            l_v_track_segment_seq = utils.get_track_segment_seq(l_v_state.id)
            l_v_current_segment = utils.get_current_segment(l_v_state,l_v_track_region,l_v_track_segment_seq,curr_time)
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


def build_game_tree():
    agent_id = 11
    veh_state = motion_planner.VehicleState()
    veh_state.set_id(agent_id)
    veh_state.set_current_time(None)
    track_region_seq = utils.get_track_segment_seq(veh_state.id)
    veh_state.set_segment_seq(track_region_seq)
    agent_track = utils.get_track(veh_state,None)
    veh_state.action_history = dict()
    veh_state.relev_agents = []
    for i in np.arange(0,30,30):
        track_info = agent_track[i]
        curr_time = float(track_info[6,])
        veh_state.set_track_info(track_info)
        if track_info[8,] is None or len(track_info[8,]) == 0:
            veh_track_region = utils.guess_track_info(veh_state,track_info)[8,]
            if veh_track_region is None:
                print(track_info[8,],agent_id,curr_time)
                sys.exit('need to guess traffic region for agent')
        else:
            veh_track_region = track_info[8,]
        current_segment = utils.assign_curent_segment(veh_track_region,track_region_seq)
        veh_state.set_current_segment(current_segment)
        path,gates,direction = utils.get_path_gates_direction(agent_track[:,8],agent_id)
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
        if curr_time not in veh_state.action_history:
            veh_state.action_history[curr_time] = dict()
        actions_l1 = constants.L1_ACTION_MAP[current_segment[:-1].replace('-','_')]
        actions_l2 = constants.L2_ACTIONS
        for l1 in actions_l1:
            for l2 in actions_l2:
                if l1 not in veh_state.action_history[curr_time]:
                    veh_state.action_history[curr_time][l1] = dict()
                veh_state.action_history[curr_time][l1][l2] = None
                trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,task)
                trajectory_plan.set_lead_vehicle(None)
                l3_actions = trajectory_plan.generate_trajectory(veh_state)
                veh_state.action_history[curr_time][l1][l2] = np.copy(l3_actions)
                print('time',curr_time,'agent',agent_id,l1,l2)
        relev_agents = utils.get_relevant_agents(veh_state)
        for r_a in relev_agents:
            if r_a == agent_id:
                continue
            r_a_state = motion_planner.VehicleState()
            r_a_state.set_id(r_a)
            r_a_state.set_current_time(curr_time)
            r_a_track = utils.get_track(r_a_state,curr_time)
            r_a_track_segment_seq = utils.get_track_segment_seq(r_a)
            r_a_state.set_segment_seq(r_a_track_segment_seq)
            r_a_state.action_history = dict()
            if curr_time not in r_a_state.action_history:
                r_a_state.action_history[curr_time] = dict()
        
            if len(r_a_track) == 0:
                ''' this agent is out of the view currently'''
                r_a_state.set_out_of_view(True)
                r_a_track = None
            else:
                r_a_state.set_out_of_view(False)
                r_a_state.set_track_info(r_a_track[0,])
                r_a_track = r_a_track[0,]
            if r_a_state.out_of_view or len(r_a_track[8]) == 0:
                r_a_track_info = utils.guess_track_info(r_a_state,r_a_track)
                r_a_state.set_track_info(r_a_track_info)
                r_a_track_region = r_a_track_info[8,]
                if r_a_track_region is None:
                    sys.exit('need to guess traffic region for relev agent')
            else:
                r_a_track_region = r_a_track[8]
            r_a_current_segment = utils.get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,curr_time)
            r_a_state.set_current_segment(r_a_current_segment)
            r_a_actions_l1 = constants.L1_ACTION_MAP[r_a_current_segment[:-1].replace('-','_')]
            #print('relev agent l1 actions for agent',agent_id,':',r_a,r_a_actions_l1)
            lead_vehicle = get_leading_vehicles(r_a_state)
            #print('lead vehicle for',r_a,':',lead_vehicle.id if lead_vehicle is not None else None)
            r_a_direction = 'L_'+r_a_track_segment_seq[0][3].upper()+'_'+r_a_track_segment_seq[-1][3].upper()
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
            actions_l1 = constants.L1_ACTION_MAP[r_a_current_segment[:-1].replace('-','_')]
            actions_l2 = constants.L2_ACTIONS
            for l1 in actions_l1:
                for l2 in actions_l2:
                    if l1 not in r_a_state.action_history[curr_time]:
                        r_a_state.action_history[curr_time][l1] = dict()
                        r_a_state.action_history[curr_time][l1][l2] = None
                    trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,r_a_task)
                    trajectory_plan.set_lead_vehicle(lead_vehicle)
                    print('time',curr_time,'agent',agent_id,'relev agent',r_a_state.id,l1,l2)
                    l3_actions = trajectory_plan.generate_trajectory(r_a_state)
                    r_a_state.action_history[curr_time][l1][l2] = np.copy(l3_actions)
            veh_state.relev_agents.append(r_a_state)
        print(relev_agents)

build_game_tree()
    