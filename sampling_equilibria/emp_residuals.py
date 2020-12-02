'''
Created on Sep 22, 2020

@author: Atrisha
'''
import matplotlib
#matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import constants
from all_utils import utils
from all_utils.utils import kph_to_mps
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from visualizer import visualizer
log = constants.common_logger
from collections import OrderedDict
import itertools
import math
import sys
import ast
import os
import pickle
  
OBJ_CACHE_DIR = "D:\\temp_cache\\"
HORIZON = 5

def pickle_dump(file_key,obj_dump):
    file_key = os.path.join(OBJ_CACHE_DIR,file_key)
    pickle.dump( obj_dump, open( file_key, "wb" ) )

class ResidualGeneration:
    
    def setup_animation(self, file_id):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        conn_traj = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_generated_trajectories_'+file_id+'.db')
        c_traj = conn_traj.cursor()
            
        all_files = os.listdir(OBJ_CACHE_DIR)
        if file_id+'_acts_dict' in all_files :
            acts_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_acts_dict'), "rb" ) )
            self.acts_dict = acts_dict
        else:
            q_string = "select * from L1_ACTIONS"
            c.execute(q_string)
            res = c.fetchall()                
            self.acts_dict = dict()
            for row in res:
                if (round(row[0]),row[1]) not in self.acts_dict:
                    self.acts_dict[(round(row[0]),row[1])] = [utils.get_l1_action_string(int(x[9:11])) for x in ast.literal_eval(row[2])]
            pickle_dump(file_id+'_acts_dict',self.acts_dict)
        if file_id+'_eq_acts_dict' in all_files :
            eq_acts_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_eq_acts_dict'), "rb" ) )
            self.eq_acts_dict = eq_acts_dict
        else:
            q_string = "select EQUILIBRIUM_ACTIONS.TRACK_ID, EQUILIBRIUM_ACTIONS.TIME,EQ_ACTIONS from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.EQ_CONFIG_PARMS='NASH|MAXMIN|BOUNDARY' order by EQUILIBRIUM_ACTIONS.TRACK_ID, EQUILIBRIUM_ACTIONS.TIME"
            c.execute(q_string)
            res = c.fetchall()                
            self.eq_acts_dict = dict()
            for row in res:
                if (round(row[1]),row[0]) not in self.eq_acts_dict:
                    self.eq_acts_dict[(round(row[1]),row[0])] = list(dict.fromkeys([utils.get_l1_action_string(int(x[9:11])) for x in ast.literal_eval(row[2])]))
            pickle_dump(file_id+'_eq_acts_dict',self.eq_acts_dict)
        if file_id+'_traj_info_dict' in all_files and file_id+'_emp_traj_dict' in all_files :
            traj_info_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_traj_info_dict'), "rb" ) )
            self.traj_info_dict = traj_info_dict
            emp_traj_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_emp_traj_dict'), "rb" ) )
            self.emp_traj_dict = emp_traj_dict
        else:   
            q_string = "select * from GENERATED_TRAJECTORY_INFO order by time"
            c_traj.execute(q_string)
            res = c_traj.fetchall()
            self.traj_info_dict = OrderedDict()
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                #if ct == 100:
                #    break
                log.info("loading "+str(ct)+'/'+str(N)+' '+file_id)
                t = int(round(row[6]))
                emp_act = self.acts_dict[(t,row[2])] if row[3] == 0 else self.acts_dict[(t,row[3])]
                if t not in self.traj_info_dict:
                    self.traj_info_dict[t] = dict()
                if len(emp_act) > 0 and row[4] in emp_act and row[5] == 'NORMAL':
                    self.traj_info_dict[t][(row[2],row[3])] = row[1]
                        
            ct,N = 0,len(self.traj_info_dict)    
            self.emp_traj_dict = dict()
            for k,v in self.traj_info_dict.items():
                ct += 1
                log.info("loading "+str(ct)+'/'+str(N)+' '+file_id)
                q_string = "select * from TRAJECTORIES_0"+file_id+" where time >="+str(k)+" and time <"+str(k+HORIZON)+" order by track_id,time"
                c.execute(q_string)
                res = c.fetchall()
                temp_trajs = dict()
                for row in res[::3]:
                    if row[0] not in temp_trajs:
                        temp_trajs[row[0]] = []
                    temp_trajs[row[0]].append((row[1], row[2], row[6], row[3]))
                self.emp_traj_dict[k] = dict()
                for t_k,t_v in self.traj_info_dict[k].items():
                    if t_k[1] == 0:
                        if t_k[0] in temp_trajs:
                            self.emp_traj_dict[k][t_k] = temp_trajs[t_k[0]]
                    else:
                        if t_k[1] in temp_trajs:
                            self.emp_traj_dict[k][t_k] = temp_trajs[t_k[1]]
                for o_a,o_t in temp_trajs.items():
                    if o_a not in [x[0] for x in self.traj_info_dict[k].keys()] and o_a not in [x[1] for x in self.traj_info_dict[k].keys()]:
                        self.emp_traj_dict[k][(o_a,-1)] = o_t
            ct,N = 0,len(self.traj_info_dict)    
            for k,v in self.traj_info_dict.items():
                ct += 1
                log.info("loading "+str(ct)+'/'+str(N)+' '+file_id)
                for ag,traj_info_id in v.items():
                    q_string = "select * from GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID="+str(traj_info_id)+" order by TRAJECTORY_ID,TIME"
                    c_traj.execute(q_string)
                    res = c_traj.fetchall()
                    if len(res) == 0:
                        self.traj_info_dict[k][ag] = []
                        continue
                    selected_traj_id = res[0][0]
                    self.traj_info_dict[k][ag] = [(row[3],row[4], row[2], row[6]) for row in res if row[0]==selected_traj_id and k <= row[2] < k+HORIZON]
                    #if ag == (8,0) and k==1:
                    #    f=1
            
            #self.traj_info_dict = self.emp_traj_dict
            #plot_traffic_regions()
            pickle_dump(file_id+'_traj_info_dict',self.traj_info_dict)
            pickle_dump(file_id+'_emp_traj_dict',self.emp_traj_dict)
        self.frame_list = []
        for k,v in self.emp_traj_dict.items():
            if len(v) == 0:
                continue
            max_len = max([len(t) for _,t in v.items()])
            this_frame_list = [(x,k)for x in np.arange(max_len)]
            self.frame_list.extend(this_frame_list)
        self.frame_list.sort(key=lambda tup: tup[1])
        conn.close()
        conn_traj.close()
        
    def __init__(self, file_id):
        self.setup_animation(file_id)
        
    def update_residuals(self, frame, emp_res_dict):
        time_ts = frame
        #print(frame)
        emp_X, emp_Y, gen_X, gen_Y = [],[],[],[]
        in_scene = []
        for ag,traj in self.emp_traj_dict[time_ts].items():
            principle_agent = ag[0] if (ag[1] == 0 or ag[1] == -1) else ag[1]
            if principle_agent not in in_scene:
                emp_X = [x[0] for x in traj[0:]]
                emp_Y = [x[1] for x in traj[0:]]
                if ag[1] == 0:
                    this_act = str(self.acts_dict[(time_ts,ag[0])]) if (time_ts,ag[0]) in self.acts_dict else 'x'
                elif ag[1] != -1:
                    this_act = str(self.acts_dict[(time_ts,ag[1])]) if (time_ts,ag[1]) in self.acts_dict else 'x'
                gen_traj = None
                for eq_ag,eq_traj in self.traj_info_dict[time_ts].items(): 
                    if eq_ag[0] == principle_agent and (eq_ag[1] == 0 or eq_ag[1] == -1):
                        gen_traj = eq_traj 
                        break
                    elif eq_ag[1] == principle_agent:
                        gen_traj = eq_traj 
                        break
                if gen_traj is None or len(gen_traj) == 0 or abs(traj[0][2]-time_ts) > 1:
                    continue
                end_idx = min(len(gen_traj),0+(HORIZON*10+1)) 
                gen_X = [x[0] for x in gen_traj[0:end_idx]]
                gen_Y = [x[1] for x in gen_traj[0:end_idx]]
                if this_act not in emp_res_dict:
                    emp_res_dict[this_act] = []
                if len(gen_X) != len(emp_X):
                    min_len = min(len(gen_X),len(emp_X))
                    gen_X = gen_X[:min_len]
                    gen_Y = gen_Y[:min_len]
                    emp_X = emp_X[:min_len]
                    emp_Y = emp_Y[:min_len]
                if this_act not in emp_res_dict:
                    emp_res_dict[this_act] = []
                traj_residual_X = [e-g for e,g in zip(emp_X,gen_X)]
                traj_residual_Y = [e-g for e,g in zip(emp_Y,gen_Y)]
                traj_residuals = list(zip(traj_residual_X,traj_residual_Y))
                max_delta = max([math.hypot(x[0],x[1]) for x in list(traj_residuals)])
                #if max_delta > 15:
                    #print(ag,time_ts,max_delta)
                    #print(this_act)
                '''
                    plt.figure()
                    visualizer.plot_traffic_regions()
                    plt.plot(gen_X,gen_Y,c='red')
                    plt.plot(emp_X,emp_Y,c='blue')
                    plt.show()
                '''
                emp_res_dict[this_act].append(traj_residuals)
                in_scene.append(principle_agent)
        return emp_res_dict
      
    
def insert_trajectory_errors():
    ins_list = []
    emp_res_dict = dict()
    for file_id in constants.ALL_FILE_IDS:
        res_gen = ResidualGeneration(file_id)
        for frame in list(set([x[1] for x in res_gen.frame_list])):
            res_gen.update_residuals(frame, emp_res_dict)
    all_traj_residuals = dict()
    for k,v in emp_res_dict.items():
        act = k
        if act not in all_traj_residuals:
            all_traj_residuals[act] = dict()
        for traj in v:
            for idx,traj_res in enumerate(traj):
                if idx not in all_traj_residuals[act]:
                    all_traj_residuals[act][idx] = []
                all_traj_residuals[act][idx].append(traj_res)
        
        plt.figure()
        X = list(all_traj_residuals[act].keys())
        Y = list(all_traj_residuals[act].values())
        Y = [[math.hypot(x2[0],x2[1]) for x2 in x] for x in Y]
        all_pts = []
        for idx,r_arr in enumerate(Y):
            for r in r_arr:
                all_pts.append((idx,r))
                       
        
        std_vals = [np.std(x) for x in Y]
        mean_vals = [np.mean(x) for x in Y]
        plt.plot(X,[mean_vals[i]+std_vals[i] for i in np.arange(len(mean_vals))],'-',c='red')
        
        
        med_vals = [np.median(x) for x in Y]
        plt.plot(X,mean_vals,'-',c='blue')
        act = ast.literal_eval(act)
        for idx,x in enumerate(X):
            ins_list.append((act[0],round(x/10,1),mean_vals[idx],med_vals[idx],std_vals[idx]))
        for file_id in constants.ALL_FILE_IDS:
            print('inserting',file_id)
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_generated_trajectories_'+file_id+'.db')
            c = conn.cursor()
            q_string = "DELETE FROM TRAJECTORY_ERRORS"
            c.execute(q_string)
            conn.commit()
            i_string = 'INSERT INTO TRAJECTORY_ERRORS VALUES (?,?,?,?,?)'
            c.executemany(i_string,ins_list)
            conn.commit()
    
        plt.plot([x[0] for x in all_pts],[x[1] for x in all_pts],'.')
        plt.title(act)
        #plt.show()
    f=1
       
if __name__ == '__main__':  
    insert_trajectory_errors()