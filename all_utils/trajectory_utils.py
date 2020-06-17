'''
Created on Jun 4, 2020

@author: Atrisha
'''
import sqlite3
import constants
import os
from all_utils import utils
from astropy.units import act

log = constants.common_logger

class TrajectoryUtils:
    
    def __init__(self,veh=None):
        if isinstance(veh, int):
            self.veh_id = veh
        else:
            self.veh_state = veh
            
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
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=NULL where EQ_CONFIG_PARMS='"+param_str+"'"
        c.execute(u_string)
        conn.commit()
        q_string = "SELECT * FROM L1_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()
        l1_act_map = {(row[0],row[1]):row[2] for row in res}
        u_string = "update EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=? where TIME=? AND TRACK_ID=? AND EQ_CONFIG_PARMS='"+param_str+"'"
        u_list = []
        for k,v in l1_act_map.items():
            u_list.append((v,k[0],k[1]))
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close() 
        
    def update_next_signal_change_in_eq_data(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
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
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
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
        u_list = []
        ct = 0
        N = len(l1_act_map)
        for k,v in l1_act_map.items():
            ct += 1 
            log.info('processing '+str(ct)+'/'+str(N))
            time_ts,ag_id = v[1],v[0]
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
            
    def assign_l1_actions(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "SELECT TIME,TRACK_ID FROM L1_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()
        time_track_list = [(row[0],row[1]) for row in res]
        conn_trajdb = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
        c_trajdb = conn_trajdb.cursor()
        action_list = []
        ct,N = 0,len(time_track_list)
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
                
        u_string = "update L1_ACTIONS SET L1_ACTION=? where TIME=? AND TRACK_ID=?"
        u_list = []
        for time_ts,agent_id in time_track_list:
            if agent_id == 2 and time_ts == 8.008:
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
                #log.info(str(ag_file_key) + " not found in cache. Setting up")
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
            l2_act = None
            print(str(ct)+'/'+str(N)+' '+str((time_ts,agent_id,emp_act)))
            if len(emp_act) > 0:
                plan_vel_profile = [x[1] for x in vel_profile[:min(len(vel_profile),constants.PLAN_FREQ*constants.DATASET_FPS)]]
                if abs(plan_vel_profile[-1] - plan_vel_profile[0]) >= constants.MAX_LONG_ACC_NORMAL:
                    l2_act = 'AGGRESSIVE'
                else:
                    l2_act = 'NORMAL'
            emp_act = [utils.unreadable(str(agent_id)+'|'+str(0)+'|'+x+'|'+l2_act) for x in emp_act]
            u_list.append((str(emp_act),time_ts,agent_id))
        c.executemany(u_string,u_list)
        conn.commit()
        conn.close()               
        
    



if __name__ == '__main__':
    traj_util_obj = TrajectoryUtils()
    traj_util_obj.update_pedestrian_info_in_eq_data()