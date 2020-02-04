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

    
   
def get_l2_actions(veh_state,actions_l1):
    motion_planner.generate_trajectory(veh_state,actions_l1)

def build_game_tree():
    agent_id = 11
    veh_state = motion_planner.VehicleState()
    veh_state.set_id(agent_id)
    veh_state.set_current_time(None)
    track_region_seq = utils.get_track_segment_seq(veh_state.id)
    veh_state.set_segment_seq(track_region_seq)
    agent_track = utils.get_track(veh_state)
    for i in np.arange(0,len(agent_track),30):
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
        actions_l1 = constants.L1_ACTION_MAP[current_segment[:-1].replace('-','_')]
        print('time',curr_time,'agent segment',current_segment,'l1 actions',actions_l1)
        #get_l2_actions(veh_state,actions_l1)
        relev_agents = utils.get_relevant_agents(veh_state)
        for r_a in relev_agents:
            if r_a == agent_id:
                continue
            r_a_state = motion_planner.VehicleState()
            r_a_state.set_id(r_a)
            r_a_state.set_current_time(curr_time)
            r_a_track = utils.get_track(r_a_state)
            r_a_track_segment_seq = utils.get_track_segment_seq(r_a)
            r_a_state.set_segment_seq(r_a_track_segment_seq)
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
            print('relev agents for agent',agent_id,':',r_a, r_a_track_region,r_a_track_segment_seq)
            r_a_current_segment = utils.get_current_segment(r_a_state,r_a_track_region,r_a_track_segment_seq,curr_time)
            r_a_state.set_current_segment(r_a_current_segment)
            r_a_actions_l1 = constants.L1_ACTION_MAP[r_a_current_segment[:-1].replace('-','_')]
            print('relev agent l1 actions for agent',agent_id,':',r_a,r_a_actions_l1)
            lead_vehice = utils.get_leading_vehicles(r_a_state)
            print('lead vehicle for',r_a,':',lead_vehice.id if lead_vehice is not None else None)
            
        print(relev_agents)
        trajectory_plan = motion_planner.TrajectoryPlan()
        trajectory_plan.set_l1_action(actions_l1)

build_game_tree()
    