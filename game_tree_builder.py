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

def get_relevant_agents(current_segment,pt_time,agent_loc):
    relev_agents = []
    if current_segment.startswith('prep-turn_s') or current_segment.startswith('exec-turn_s'):
        ''' add oncoming vehicles on intersection '''
        relev_agents += utils.get_n_s_vehicles_on_intersection(pt_time)
        ''' add oncoming vehicles about to enter the intersection '''
        relev_agents += utils.get_n_s_vehicles_before_intersection(pt_time)
        ''' add oncoming north vehicles about to turn east'''
        relev_agents += utils.get_n_e_vehicles_before_intersection(pt_time)
        ''' add oncoming north south vehicles turning east'''
        relev_agents += utils.get_n_e_vehicles_on_intersection(pt_time)
        ''' add oncoming north south vehicles turning west'''
        relev_agents += utils.get_n_e_vehicles_on_intersection(pt_time)
        ''' add closest east vehicles '''
        relev_agents += utils.get_closest_east_vehicles_before_intersection(pt_time,agent_loc)
        ''' add closest west vehicles'''
        relev_agents += utils.get_closest_west_vehicles_before_intersection(pt_time,agent_loc)
        ''' add any other vehicle on the intersection '''
        
        if current_segment.startswith('exec-turn_s'):
            ''' add vehicles turning north to west '''
            relev_agents += utils.get_n_w_vehicles(pt_time)
        print(pt_time,current_segment,relev_agents)
    else:
        print(current_segment)
        sys.exit('segment not supported')
            
            
        
    
    
    '''
    q_string = "select distinct track_id from trajectories_0769 where time between 0 and 12.846167"
    res = c.execute(q_string)
    l = []
    for row in res:
        l.append(row[0])
    q_string = "SELECT * FROM TRAJECTORY_MOVEMENTS WHERE ((ENTRY_GATE = 60 AND EXIT_GATE = 18 ) OR (EXIT_GATE = 18 AND ENTRY_GATE IS NULL)) AND (TRACK_ID IN "+str(tuple(l))+")"
    res = c.execute(q_string)
    for row in res:
        print(row)
    '''
   
def get_l2_actions(veh_state,actions_l1):
    motion_planner.generate_trajectory(veh_state,actions_l1)

def build_game_tree():
    agent_id = 11
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "select * from trajectories_0769 where track_id="+str(agent_id)+" order by time"
    res = c.execute(q_string)
    l = []
    for row in res:
        l.append(row)
    track_region_seq = utils.get_track_segment_seq(agent_id)
    agent_track = np.asarray(l)
    for i in np.arange(0,len(agent_track),30):
        point = agent_track[i]
        veh_state = motion_planner.VehicleState(point)
        current_segment = utils.assign_curent_segment(point[8,],track_region_seq)
        veh_state.set_current_segment(current_segment)
        veh_state.set_path(utils.get_path(agent_track[:,8]))
        if current_segment is None:
            print('no current segment found')
            break
        pt_time = point[6,]
        agent_loc = (point[1,],point[2,])
        actions_l1 = constants.ACTION_MAP[current_segment[:-1]]
        print(actions_l1)
        get_l2_actions(veh_state,actions_l1)
        get_relevant_agents(current_segment,pt_time,agent_loc)
    
    
    conn.close()

build_game_tree()
    