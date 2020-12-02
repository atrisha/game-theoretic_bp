'''
Created on Jun 4, 2020

@author: Atrisha
'''
import sqlite3
import constants
import os
from all_utils import utils
from astropy.units import act
import ast
from matplotlib import path
import matplotlib.pyplot as plt
log = constants.common_logger
from visualizer import visualizer
import numpy as np
from equilibria.cost_evaluation import *

class TrajectoryUtils:
    
    def __init__(self,veh=None):
        if isinstance(veh, int):
            self.veh_id = veh
        else:
            self.veh_state = veh
            
    def remove_out_of_lane_trajectories(self):
        file_ids_to_process = [775]
        show_plot = True
        working_viewport = [(538805.45,4813966.04),(538896.21,4814013.41),(538863.54,4814058.15),(538785.24,4814009.88)]
        viewport_p = path.Path(working_viewport)
        lane_bounds = dict()
        for file_id in file_ids_to_process:
            conn_trajdb = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+str(file_id)+'\\uni_weber_generated_trajectories_'+str(file_id)+'.db')
            c_trajdb = conn_trajdb.cursor()
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+str(file_id)+'\\uni_weber_'+str(file_id)+'.db')
            c = conn.cursor()
            q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF WHERE REGION_PROPERTY='lane_boundary'"
            c.execute(q_string)
            res = c.fetchall()
            #plt.axis("equal")
            for row in res:
                #plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
                X,Y = ast.literal_eval(row[4]),ast.literal_eval(row[5])
                lane_bounds[row[0]] = list(zip(X,Y))
                #plt.plot(X+[X[0]],Y+[Y[0]])
            #plt.plot([x[0] for x in working_viewport]+[working_viewport[0][0]],[x[1] for x in working_viewport]+[working_viewport[0][1]])
            #plt.show()
            q_string = "SELECT TRACK_ID,TRAFFIC_SEGMENT_SEQ FROM TRAJECTORY_MOVEMENTS"
            c.execute(q_string)
            res = c.fetchall()
            traj_direction = dict()
            for row in res:
                if row[1] is not None and len(row[1]) > 0:
                    print(row[1])
                    _d = ast.literal_eval(row[1])
                    traj_direction[row[0]] = _d[0][3].upper()+'_'+_d[-1][3].upper()
            q_string = "SELECT * FROM GENERATED_TRAJECTORY_INFO"
            c_trajdb.execute(q_string)
            res = c_trajdb.fetchall()
            all_traj_dict = dict()
            for row in res:
                veh_id = row[2] if row[3] == 0 else row[3]
                direction = traj_direction[veh_id]
                if direction not in all_traj_dict:
                    all_traj_dict[direction] = dict()
                all_traj_dict[direction][row[1]] = {'BASELINE':[],'BOUNDARY':[],'GAUSSIAN':[]}
            for dir,dir_det in all_traj_dict.items():
                for table in ['BASELINE','BOUNDARY','GAUSSIAN']:
                    print(dir,table)
                    remove_list,retain_list =[],[]
                    if show_plot:
                        plt.figure()
                        plt.plot([x[0] for x in lane_bounds[dir]]+[lane_bounds[dir][0][0]],[x[1] for x in lane_bounds[dir]]+[lane_bounds[dir][0][1]])
                        plt.plot([x[0] for x in working_viewport]+[working_viewport[0][0]],[x[1] for x in working_viewport]+[working_viewport[0][1]],color='red')
                    removed_traj_info_dets = []
                    for traj_info_id,_ in dir_det.items():
                        trajs_dict = dict()
                        q_string = "select * FROM GENERATED_"+table+"_TRAJECTORY WHERE TRAJECTORY_INFO_ID ="+str(traj_info_id)+" order by TRAJECTORY_ID,TIME"
                        c_trajdb.execute(q_string)
                        res = c_trajdb.fetchall()
                        for row in res:
                            if row[0] not in trajs_dict:
                                trajs_dict[row[0]] = []
                            trajs_dict[row[0]].append((row[3],row[4]))
                        for t_info,traj in trajs_dict.items():
                            in_viewport_view = viewport_p.contains_points(traj)
                            if any(in_viewport_view):
                                st_clip_idx,en_clip_idx = 0,len(traj)-1
                                for idx,pt in enumerate(traj):
                                    in_view = viewport_p.contains_points([pt])
                                    if in_view[0]:
                                        st_clip_idx = idx
                                        break
                                if st_clip_idx < en_clip_idx:
                                    for idx,pt in enumerate(reversed(traj)):
                                        in_view = viewport_p.contains_points([pt])
                                        if in_view[0]:
                                            en_clip_idx = en_clip_idx - idx
                                            break
                                cliped_traj = traj[st_clip_idx:en_clip_idx+1]
                                lb_path = path.Path(lane_bounds[dir])
                                in_view = lb_path.contains_points(cliped_traj)
                                if sum(in_view)/len(in_view) <= 0.9:
                                    remove_list.append(t_info)
                                    removed_traj_info_dets.append(traj_info_id)
                                    if show_plot:
                                        plt.plot([x[0] for x in cliped_traj],[x[1] for x in cliped_traj])
                                    
                                else:
                                    retain_list.append(t_info)
                            else:
                                retain_list.append(t_info)
                    print(removed_traj_info_dets[0:10])
                    if show_plot:
                        visualizer.plot_traffic_regions()
                        plt.show()
                    print('remove '+str(len(remove_list))+'/'+str(len(remove_list)+len(retain_list)))
                    q_string = "DELETE FROM GENERATED_"+table+"_TRAJECTORY WHERE TRAJECTORY_ID=?"
                    if len(remove_list) > 0:
                        c_trajdb.executemany(q_string,[(x,) for x in remove_list])
                        conn_trajdb.commit()
                        f=1
        conn_trajdb.close()
        conn.close()
                                    
            
        
            
    def eval_wait_or_proceed(self,wait_actions,proceed_actions,vel_profile,veh_state):
        plan_vel_profile,horizon_vel_profile = [],[]
        for ts,v in vel_profile:
            if ts < vel_profile[0][0] + constants.PLAN_FREQ + constants.LP_FREQ:
                plan_vel_profile.append(v)
                horizon_vel_profile.append(v)
            elif ts <= vel_profile[0][0] + 5:
                horizon_vel_profile.append(v)
            else:
                break
        if (plan_vel_profile[-1] - plan_vel_profile[0]) > 0.5:
            proceed,wait = True,False
        else:
            proceed,wait = False,True
        if plan_vel_profile[-1] < 0.5:
            proceed,wait = False,True
        return proceed,wait
                
    def eval_stationary(self,wait_actions,proceed_actions,vel_profile,veh_state):    
        plan_vel_profile,horizon_vel_profile = [],[]
        for ts,v in vel_profile:
            if ts < vel_profile[0][0] + constants.PLAN_FREQ + constants.LP_FREQ:
                plan_vel_profile.append(v)
                horizon_vel_profile.append(v)
            elif ts <= vel_profile[0][0] + 5:
                horizon_vel_profile.append(v)
            else:
                break
        if plan_vel_profile[-1] < 0.1 and plan_vel_profile[0] < 0.1:
            return True
        else:
            return False
        
    def update_l1_action_in_eq_data(self,param_str):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        if param_str == 'all':
            u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=NULL"
        else:
            u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=NULL where EQ_CONFIG_PARMS='"+param_str+"'"
        c.execute(u_string)
        conn.commit()
        q_string = "SELECT * FROM L1_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()
        l1_act_map = {(row[0],row[1]):row[2] for row in res}
        if param_str == 'all':
            u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=? where TIME=? AND TRACK_ID=?"
        else:
            u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=? where TIME=? AND TRACK_ID=? AND EQ_CONFIG_PARMS='"+param_str+"'"
        u_list = []
        for k,v in l1_act_map.items():
            u_list.append((v,k[0],k[1]))
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close() 
        
    def update_next_signal_change_in_eq_data(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM EQUILIBRIUM_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()
        l1_act_map = {row[0]:(row[4],'L_'+row[2],row[5]) for row in res}
        u_list = []
        for k,v in l1_act_map.items():
            if v[2] is not None and v[1] is not None:
                next_change = utils.get_time_to_next_signal(v[0],v[1],v[2])
                if next_change[0] is not None:
                    u_list.append((str(next_change),k))
        u_string = "update EQUILIBRIUM_ACTIONS SET NEXT_SIGNAL_CHANGE=? where L1L2_EQ_ID=?"
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close() 
        
    def update_pedestrian_info_in_eq_data(self,param_str):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM EQUILIBRIUM_ACTIONS WHERE EQ_CONFIG_PARMS='"+param_str+"'"
        c.execute(q_string)
        res = c.fetchall()
        l1_act_map = {row[0]:(row[3],row[4]) for row in res}
        all_files = os.listdir(os.path.join(constants.ROOT_DIR, constants.L3_ACTION_CACHE))
        all_files.sort()
        N = len(all_files)
        file_key_map = dict()
        for idx,file in enumerate(all_files):
            org_str = str(file)
            st = file.replace(',','.')
            ag_id = int(st.split('-')[0])
            relev_id,time = st.split('-')[1].split('_')
            if int(relev_id) == 0:
                file_key_map[(ag_id,time)] = os.path.join(constants.ROOT_DIR,constants.L3_ACTION_CACHE,org_str)
            else:
                if (int(relev_id),time) not in file_key_map:
                    file_key_map[(int(relev_id),time)] = os.path.join(constants.ROOT_DIR,constants.L3_ACTION_CACHE,org_str)
        ''' entries might already be there in the table for other eq params. Try to get it from there first. '''
        q_string = "select TRACK_ID,TIME,PEDESTRIAN FROM EQUILIBRIUM_ACTIONS WHERE PEDESTRIAN IS NOT NULL"
        c.execute(q_string)
        res = c.fetchall()
        peds_info_map_indb = {(row[0],row[1]):row[2] for row in res}
        u_list = []
        ct = 0
        N = len(l1_act_map)
        for k,v in l1_act_map.items():
            ct += 1 
            log.info('processing '+str(ct)+'/'+str(N))
            time_ts,ag_id = v[1],v[0]
            if (ag_id,time_ts) in peds_info_map_indb:
                has_ped = peds_info_map_indb[(ag_id,time_ts)]
                u_list.append((has_ped,k))
            else:
                pedestrian_info = utils.setup_pedestrian_info(time_ts)
                ag_file_key = (ag_id,str(time_ts)) if time_ts !=0 else (ag_id,str(time_ts)+'.0') 
                if ag_file_key in file_key_map:
                    veh_state = utils.pickle_load(file_key_map[ag_file_key])
                    task = constants.TASK_MAP[veh_state.direction]
                    veh_state.set_task(task)
                else:
                    log.info(str(ag_file_key) + " not found in cache. Setting up")
                    veh_state = utils.setup_vehicle_state(ag_id, time_ts)
                relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
                veh_state.set_relev_crosswalks(relev_crosswalks)
                relev_pedestrians = utils.get_relevant_pedestrians(veh_state, pedestrian_info)
                if relev_pedestrians is not None:
                    u_list.append(('Y',k))
                else:
                    u_list.append(('N',k))
        u_string = "update EQUILIBRIUM_ACTIONS SET PEDESTRIAN=? where L1L2_EQ_ID=?"
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close() 
            
    def assign_l1_actions(self,selected_agent_id=None):
        
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        conn_trajdb = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_generated_trajectories_'+constants.CURRENT_FILE_ID+'.db')
        c_trajdb = conn_trajdb.cursor()
        if selected_agent_id is None:
            q_string = "SELECT DISTINCT TIME,AGENT_ID FROM GENERATED_TRAJECTORY_INFO UNION SELECT DISTINCT TIME,RELEV_AGENT_ID FROM GENERATED_TRAJECTORY_INFO ORDER BY TIME"
        else:
            q_string = "SELECT DISTINCT TIME,AGENT_ID FROM GENERATED_TRAJECTORY_INFO where AGENT_ID="+str(selected_agent_id)
        c_trajdb.execute(q_string)
        res = c_trajdb.fetchall()
        time_track_list = [(row[0],row[1]) for row in res if row[1]!=0]
        action_list = []
        ct,N = 0,len(time_track_list)
        all_files = os.listdir(os.path.join(constants.CACHE_DIR, constants.L3_ACTION_CACHE))
        all_files.sort()
        N = len(all_files)
        file_key_map = dict()
        for idx,file in enumerate(all_files):
            org_str = str(file)
            st = file.replace(',','.')
            ag_id = int(st.split('-')[0])
            relev_id,time = st.split('-')[1].split('_')
            if int(relev_id) == 0:
                file_key_map[(ag_id,time)] = os.path.join(constants.CACHE_DIR,constants.L3_ACTION_CACHE,org_str)
            else:
                if (int(relev_id),time) not in file_key_map:
                    file_key_map[(int(relev_id),time)] = os.path.join(constants.CACHE_DIR,constants.L3_ACTION_CACHE,org_str)
                
        u_string = "REPLACE INTO L1_ACTIONS VALUES (?,?,?)"
        u_list = []
        for time_ts,agent_id in time_track_list:
            if agent_id == 100 and time_ts == 100.1:
                brk = 1
            ct += 1
            pedestrian_info = utils.setup_pedestrian_info(time_ts)
            ag_file_key = (agent_id,str(time_ts)) if time_ts !=0 else (agent_id,str(time_ts)+'.0') 
            if ag_file_key in file_key_map:
                #log.info("loading from cache")
                veh_state = utils.pickle_load(file_key_map[ag_file_key])
                task = constants.TASK_MAP[veh_state.direction]
                veh_state.set_task(task)
    
            else:
                log.info(str(ag_file_key) + " not found in cache. Setting up")
                veh_state = utils.setup_vehicle_state(agent_id, time_ts)
            relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
            veh_state.set_relev_crosswalks(relev_crosswalks)
            relev_pedestrians = utils.get_relevant_pedestrians(veh_state, pedestrian_info)
            veh_state.set_relev_pedestrians(relev_pedestrians)
            actions = list(utils.get_actions(veh_state).keys())
            actions.sort()
            if actions not in action_list:
                action_list.append(actions)
            wait_actions,proceed_actions = [],[]
            for action in actions:
                if action in constants.WAIT_ACTIONS:
                    wait_actions.append(action)
                else:
                    proceed_actions.append(action)
            horizon_end_ts = float(time_ts) + constants.PLAN_FREQ + constants.LP_FREQ
            q_string = "select TIME,speed from TRAJECTORIES_0"+constants.CURRENT_FILE_ID+" where TRAJECTORIES_0"+constants.CURRENT_FILE_ID+".TRACK_ID="+str(agent_id)+" AND TIME >= "+str(time_ts)+" ORDER BY TIME"
            c.execute(q_string)
            res = c.fetchall()
            vel_profile = [(row[0],row[1]) for row in res]
            emp_act = []
            proceed,wait = False,False
            if len(wait_actions) > 0 and len(proceed_actions) > 0:
                proceed,wait = self.eval_wait_or_proceed(wait_actions,proceed_actions,vel_profile,veh_state)
            if (len(proceed_actions) > 0 and (not proceed and not wait)) or (proceed and not wait):
                if veh_state.leading_vehicle is not None:
                    for act in proceed_actions:
                        if 'lead' in act:
                            emp_act.append(act)
                            break
                else:
                    emp_act.append(proceed_actions[0])
                if len(proceed_actions) > 0 and len(emp_act) == 0:
                    if veh_state.task != 'STRAIGHT' and 'proceed-turn' in proceed_actions:
                        emp_act.append('proceed-turn')
                    else:
                        if veh_state.leading_vehicle is not None:
                            emp_act.append('follow_lead')
                        else:
                            emp_act.append('track_speed')
            if (len(wait_actions) > 0 and (not proceed and not wait)) or (wait and not proceed)> 0:
                if 'wait-for-pedestrian' in wait_actions:
                    emp_act.append('wait-for-pedestrian')
                stationary = self.eval_stationary(wait_actions,proceed_actions,vel_profile,veh_state)
                if veh_state.leading_vehicle is None:
                    if stationary:
                        if 'wait-on-red' in wait_actions:
                            emp_act.append('wait-on-red')
                        else:
                            if 'wait-for-oncoming' in wait_actions:
                                emp_act.append('wait-for-oncoming')
                    else:
                        
                        if constants.SEGMENT_MAP[veh_state.current_segment] in constants.ENTRY_LANES:
                            if 'decelerate-to-stop' in wait_actions:
                                emp_act.append('decelerate-to-stop')
                        else:
                            if 'wait-for-oncoming' in wait_actions:
                                emp_act.append('wait-for-oncoming')
                else:
                    if stationary:
                        if 'wait-on-red' in wait_actions:
                            emp_act.append('wait-on-red')
                        else:
                            if 'wait_for_lead_to_cross' in wait_actions:
                                emp_act.append('wait_for_lead_to_cross')
                    else:
                        if constants.SEGMENT_MAP[veh_state.current_segment] in constants.ENTRY_LANES:
                            if 'decelerate-to-stop' in wait_actions:
                                emp_act.append('decelerate-to-stop')
                            
                        else:
                            if 'wait-for-oncoming' in wait_actions:
                                emp_act.append('wait-for-oncoming')
                if len(emp_act) == 0 and len(wait_actions) == 1:
                    emp_act.append(wait_actions[0]) 
            l2_act = None
            print(constants.CURRENT_FILE_ID+':'+str(ct)+'/'+str(N)+' '+str((time_ts,agent_id,emp_act)))
            if len(emp_act) > 0:
                plan_vel_profile = [x[1] for x in vel_profile[:min(len(vel_profile),constants.PLAN_FREQ*constants.DATASET_FPS)]]
                if abs(plan_vel_profile[-1] - plan_vel_profile[0]) >= constants.MAX_LONG_ACC_NORMAL:
                    l2_act = 'AGGRESSIVE'
                else:
                    l2_act = 'NORMAL'
            emp_act = [utils.unreadable(str(agent_id)+'|'+str(0)+'|'+x+'|'+l2_act) for x in emp_act]
            u_list.append((time_ts,agent_id,str(emp_act)))
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close()               
        
def add_missing_attributes(veh_state):
    if not hasattr(veh_state, 'relev_crosswalks'):
        relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
        veh_state.set_relev_crosswalks(relev_crosswalks)
    #veh_state.has_oncoming_vehicle = False
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_'+constants.CURRENT_FILE_ID+'.db')
    c = conn.cursor()
    if constants.SEGMENT_MAP[veh_state.segment_seq[0]] =='left-turn-lane' or constants.SEGMENT_MAP[veh_state.segment_seq[0]] == 'right-turn-lane' \
        or constants.SEGMENT_MAP[veh_state.segment_seq[0]] == 'through-lane-entry':
            q_string = "select * from TRAFFIC_REGIONS_DEF WHERE name='"+veh_state.segment_seq[0]+"' and TRAFFIC_REGIONS_DEF.REGION_PROPERTY='entry_boundary'"
            c.execute(q_string)
            res_reg = c.fetchone()
            x_pts = np.mean(ast.literal_eval(res_reg[4]))
            y_pts = np.mean(ast.literal_eval(res_reg[5]))
            veh_state.origin_pt = (x_pts,y_pts)
    else:
        veh_state.origin_pt = None
    return veh_state

def assign_true_utils():
    ''' this will assign utils for l1 games that is independent of l3 game solutions.
    A heuristic is used to assign the three utils: vehicle inhibitory (time gap), excitatory, and pedestrian inhibitory'''
    ''' Since we already have the l1 tree data in <cache_dir>/l3_trees_BASELINE_NA, that has the agent, relevant agents, and actions information,
    we can get it from there instead of loading it from the database '''
    file_id = '769'
    constants.CURRENT_FILE_ID = file_id
    constants.L3_ACTION_CACHE = os.path.join(constants.CACHE_DIR,'l3_action_trajectories_'+constants.CURRENT_FILE_ID)
    l3_cache_str = constants.CACHE_DIR+"l3_trees_BASELINE_NA\\"+file_id    
    file_keys = os.listdir(l3_cache_str)
    for file_str in file_keys:
        print('processing',file_id,file_str)
        time_ts = file_str.split('_')[1].replace(',','.')
        time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
        payoff_dict = utils.pickle_load(os.path.join(l3_cache_str,file_str))
        all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
        num_players = len(all_agents)
        agent_obj_map = dict()
        for ag_idx,ag in enumerate(all_agents):
            if ag[1] == 0:
                ag_file_key = os.path.join(constants.L3_ACTION_CACHE, str(ag[0])+'-0_'+str(time_ts).replace('.', ','))
                agent_id = ag[0]
            else:
                ag_file_key = os.path.join(constants.L3_ACTION_CACHE, str(ag[0])+'-'+str(ag[1])+'_'+str(time_ts).replace('.', ','))
                agent_id = ag[1]
            if os.path.exists(ag_file_key):
                ag_info = utils.pickle_load(ag_file_key)
            else:
                ag_info = utils.setup_vehicle_state(agent_id,time_ts)
            ag_info = add_missing_attributes(ag_info)
            agent_obj_map[agent_id] = ag_info
            all_agents[ag_idx] = agent_id
        for strat in payoff_dict.keys():
            ttc = np.full((num_players,num_players), fill_value=np.inf)
            for i in np.arange(num_players):
                for j in np.arange(num_players):
                    if i!=j:
                        ttc[i,j] = calc_time_gap(agent_obj_map[all_agents[i]], agent_obj_map[all_agents[j]], strat)
        f=1



def main():
    import sys
    for f in constants.ALL_FILE_IDS:
        constants.CURRENT_FILE_ID = f
        constants.L3_ACTION_CACHE = 'l3_action_trajectories_'+constants.CURRENT_FILE_ID
        traj_util_obj = TrajectoryUtils()
        traj_util_obj.assign_l1_actions()
        #traj_util_obj.update_l1_action_in_eq_data('all')

if __name__ == '__main__':
    assign_true_utils()