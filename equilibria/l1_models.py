'''
Created on Oct 20, 2020

@author: Atrisha
'''
from equilibria import equilibria_core, cost_evaluation 
from all_utils import utils,db_utils,trajectory_utils
import itertools
import constants
import numpy as np
import matplotlib.pyplot as plt
from planning_objects import VehicleState
from collections import OrderedDict
import pandas as pd
from collections import Counter
from equilibria.equilibria_objects import *
import math
import sqlite3
import seaborn as sn
import ast
from scipy.optimize import curve_fit
from sklearn import tree
from all_utils.lp_utils import solve_lp
import csv
import os

import constants
from subprocess import check_output
import math



class Rules():
    
    def __init__(self, file_id):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        q_string = "select * from ACTIONS where IS_RULE='Y'"
        c.execute(q_string)
        res = c.fetchall()                
        ''' (segment,signal,lead vehicle,task, pedestrian,state_string(lead,pedestrian,oncoming))):action'''
        self.rules = {(row[0],row[3],row[4],row[5],row[7],row[8]):row[1] for row in res}
        
    def resolve_rule(self,veh_state,pedestrian_info):
        relev_pedestrians = utils.get_relevant_pedestrians(veh_state, pedestrian_info)
        key_tuples = [[constants.SEGMENT_MAP[veh_state.current_segment]], [veh_state.signal,'*']]
        state_str = []
        if veh_state.leading_vehicle is not None:
            key_tuples.append(['Y','*'])
        else:
            key_tuples.append(['N','*'])
        key_tuples.append([veh_state.task])
        if relev_pedestrians is not None:
            key_tuples.append(['Y','*'])
        else:
            key_tuples.append(['N','*'])
        if veh_state.has_oncoming_vehicle:
            oncoming_veh = ['Y','*']
        else:
            oncoming_veh = ['N','*']
        state_str = [','.join(x) for x in itertools.product(key_tuples[2],key_tuples[4],oncoming_veh)]
        all_keys = list(itertools.product(key_tuples[0],key_tuples[1],key_tuples[2],key_tuples[3],key_tuples[4],state_str))
        action = []
        for k in all_keys:
            if k in self.rules:
                action.append(self.rules[k])
        if len(action) > 1:
            all_wait_actions = all(a in constants.WAIT_ACTIONS for a in action)
            if all_wait_actions:
                chosen_action = None
                for a in reversed(constants.WAIT_ACTIONS):
                    if a in action:
                        chosen_action = a
                        break
                if chosen_action is None:
                    sys.exit('No action in '+str(action)+' found in the list of wait actions')
                else:
                    action = [chosen_action]
            else:
                if 'wait_for_lead_to_cross' in action and ('Y' in oncoming_veh or relev_pedestrians is not None):
                    action = ['wait_for_lead_to_cross']
                elif 'wait-for-pedestrian' in action and relev_pedestrians is not None:
                    action = ['wait-for-pedestrian']
                else:
                    if 'follow_lead_into_intersection' in action and ('Y' in oncoming_veh or relev_pedestrians is not None):
                        action.remove('follow_lead_into_intersection')
        return action

class L1ModelCore():
    
    def __init__(self,file_id,with_baseline_weights,mixed_weights):
        self.mixed_weights = mixed_weights
        self.with_baseline_weights = with_baseline_weights
        self.file_id = file_id
        self.l3_cache_str = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\l3_trees\\"+self.file_id
        self.weighted_l3_cache_str = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\weighted_l3_trees\\"+self.file_id
        self.nash_eq_res_cache = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\nash_eq_res\\"+self.file_id
        if not os.path.exists(self.weighted_l3_cache_str):
            os.makedirs(self.weighted_l3_cache_str)
        if not os.path.exists(self.nash_eq_res_cache):
            os.makedirs(self.nash_eq_res_cache)
        if constants.L3_ACTION_CACHE is None:
            constants.L3_ACTION_CACHE = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\l3_action_trajectories_"+file_id
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        q_string = "select * from L1_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()                
        acts_dict = dict()
        for row in res:
            if (round(row[0]),row[1]) not in acts_dict:
                acts_dict[(round(row[0]),row[1])] = [utils.get_l1_action_string(int(x[9:11])) for x in ast.literal_eval(row[2])]
        self.acts_dict = acts_dict
        time_ts = 0
        agents = []
        self.rule_obj = Rules(file_id)
        file_keys = os.listdir(self.l3_cache_str)
        self.file_keys = file_keys
        self.load_state_info()
        self.load_emp_and_rule_acts()
    
    def load_state_info(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "select * from EQUILIBRIUM_ACTIONS"
        c.execute(q_string)
        res = c.fetchall()
        self.state_info = {(row[3],row[4]):tuple(row[5:9] + row[11:14]) for row in res}
    
    def load_emp_and_rule_acts(self):
        #confusion_dict = dict()
        ct,N = 0,len(self.file_keys)
        self.emp_acts_dict, self.rule_acts_dict, self.all_agent_obj_dict = dict(), dict(), dict()
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',ct,'/',N)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            self.emp_acts_dict[file_str] = emp_acts
            self.rule_acts_dict[file_str] = rule_acts
            
    def get_emp_and_rule_actions(self,file_str,payoff_dict):
        time_ts = file_str.split('_')[1].replace(',','.')
        time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
        pedestrian_info = utils.setup_pedestrian_info(time_ts)
        agent_object_dict = dict()
        all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
        for ag in all_agents:
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
            self.add_missing_attributes(ag_info)
            agent_object_dict[ag] = ag_info
        if 'TURN' in agent_object_dict[ag].task:
            for ag,ag_obj in agent_object_dict.items():
                if ag != agent_id and 'TURN' not in ag_obj.task:
                    if agent_object_dict[ag].leading_vehicle is None or (agent_object_dict[ag].leading_vehicle is not None and ag != agent_object_dict[ag].leading_vehicle.id):
                        agent_object_dict[ag].has_oncoming_vehicle = True
                        break
        emp_acts, rule_acts = [],[]
        self.agent_object_dict = agent_object_dict
        self.all_agent_obj_dict[file_str] = agent_object_dict
        for ag in all_agents:
            ag_obj = agent_object_dict[ag]
            rule_action, emp_action = None, None
            rule_action = self.rule_obj.resolve_rule(ag_obj, pedestrian_info)
            agent_id = ag[0] if ag[1]==0 else ag[1]
            if (round(time_ts),agent_id) in self.acts_dict:
                emp_action = self.acts_dict[(round(time_ts),agent_id)]
            emp_acts.append([self.file_id+str(ag[0]).zfill(3)+str(ag[1]).zfill(3)+str(constants.L1_ACTION_CODES[x]).zfill(2)+'02' for x in emp_action])
            rule_acts.append([self.file_id+str(ag[0]).zfill(3)+str(ag[1]).zfill(3)+str(constants.L1_ACTION_CODES[x]).zfill(2)+'02' for x in rule_action])
        emp_acts = list(itertools.product(*emp_acts))
        rule_acts = list(itertools.product(*rule_acts))
        return (emp_acts,rule_acts)
     
class InverseGameLearningModel(L1ModelCore):
    
    def add_missing_attributes(self,veh_state):
        if not hasattr(veh_state, 'relev_crosswalks'):
            relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
            veh_state.set_relev_crosswalks(relev_crosswalks)
        veh_state.has_oncoming_vehicle = False
        
    
            
    def formulate_and_solve_lp(self,all_agents,all_obj_mat,all_constr_mat,on_rule_list):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if on_rule_list[idx]:
                all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
            else:
                obj_mat = all_obj_mat[:,idx]
                constr_mat = [x[:,idx] for x in all_constr_mat]
                num_params = 2 if self.agent_object_dict[ag].relev_pedestrians is None else 3
                solns = solve_lp(obj_mat, constr_mat, num_params)
                all_solns.append((ag,solns))
        return all_solns
    
    
    def convert_to_nfg(self):
        
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM UTILITY_WEIGHTS"
        c.execute(q_string)
        res = c.fetchall()
        if len(res) == 0:
            sys.exit('UTILITY_WEIGHTS weights table is not populated ')
        weights_dict = dict()
        for row in res:
            if row[12] is not None:
                if tuple(row[3:11]) not in weights_dict:
                    weights_dict[tuple(row[3:11])] = [row[12:]]
                else:
                    weights_dict[tuple(row[3:11])].append(row[12:])
        weights_dict = {k: np.mean(np.asarray(v), axis=0) for k,v in weights_dict.items()}
        ct,N = 0,len(self.file_keys)
        ''' UTILITY_WEIGHTS TABLE also stores the agent state info information, so we can retrieve that instead of building it again '''
        q_string = "SELECT * FROM UTILITY_WEIGHTS"
        c.execute(q_string)
        res = c.fetchall()
        agent_info = {tuple(row[:3]):tuple(row[3:11]) for row in res}
        for file_str in self.file_keys:
            ct +=1
            print('processing',ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict}
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            #emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            for ag_idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                agent_info_key = (int(self.file_id),ag_id,time_ts)
                ag_state_key = agent_info[agent_info_key] if agent_info_key in agent_info else None
                if (ag_state_key is not None and ag_state_key in weights_dict) and not self.with_baseline_weights:
                    ag_weights = weights_dict[ag_state_key]
                else:
                    ag_weights = np.asarray([0.25, 0.25,  0.25, 0.25, 0.5, 0.5, 0, 0])
                    print('default weights assigned')
                for k in payoff_dict.keys():
                    orig_payoff = payoff_dict[k]
                    if len(rule_acts) == 0:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                        ag_orig_payoff = orig_payoff[:,ag_idx]
                    else:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                        if k[ag_idx] in [x[ag_idx] for x in rule_acts]:
                            ag_orig_payoff = np.append(orig_payoff[:,ag_idx],1)
                        else:
                            ag_orig_payoff = np.append(orig_payoff[:,ag_idx], 0)
                    if self.mixed_weights:
                        if np.sum(high_weights) != 0:
                            high_weights = high_weights/np.sum(high_weights)
                        if np.sum(low_weights) != 0:
                            low_weights = low_weights/np.sum(low_weights)
                        
                    else:
                        #high_weights = np.where(high_weights == max(high_weights), 1, 0)
                        if np.sum(np.exp(10*high_weights)) != 0:
                            high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        #low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        if np.sum(np.exp(10*high_weights)) != 0:
                            low_weights = np.exp(10*low_weights)/np.sum(np.exp(10*low_weights))
                    high_weighted_payoffs = high_weights @ ag_orig_payoff
                    low_weighted_payoffs = low_weights @ ag_orig_payoff
                    low_bounds_transformed_payoffs[k][ag_idx] = low_weighted_payoffs[0]
                    high_bounds_transformed_payoffs[k][ag_idx] = high_weighted_payoffs[0]
            
            if not self.with_baseline_weights:
                eq = EquilibriaCore(len(all_agents),low_bounds_transformed_payoffs,len(low_bounds_transformed_payoffs),None,None)
                out_file_str = os.path.join(self.weighted_l3_cache_str,file_str+'_low.nfg')
                eq.transform_to_nfg_format(out_file_str,all_agents)
            eq = EquilibriaCore(len(all_agents),high_bounds_transformed_payoffs,len(high_bounds_transformed_payoffs),None,None)
            if self.with_baseline_weights:
                out_file_str = os.path.join(self.weighted_l3_cache_str,file_str+'_high_baselineW.nfg')
            else:
                out_file_str = os.path.join(self.weighted_l3_cache_str,file_str+'_high.nfg')
            eq.transform_to_nfg_format(out_file_str,all_agents)
            
                    
    def invoke_pure_strat_nash_eq_calc(self):
        nfg_file_keys = os.listdir(self.weighted_l3_cache_str)
        ct,N = 0,len(nfg_file_keys)
        for file_str in nfg_file_keys:
            ct +=1
            print('processing',ct,'/',N)
            out = check_output('"C:\\Program Files (x86)\\Gambit\\gambit-enumpure.exe" -q "'+self.weighted_l3_cache_str+'\\'+file_str+'"', shell=True).decode()
            outfile_loc = os.path.join(self.nash_eq_res_cache,file_str.split('.')[0]+'.ne')
            text_file = open(outfile_loc, "w")
            text_file.write(out)
            text_file.close()
            
    def calc_confusion_matrix(self):
        ct,N = 0,len(self.file_keys)
        confusion_dict = dict()
        for file_str in self.file_keys:
            ct +=1
            print('processing',ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(len(all_agents))]
            all_acts = [x for sublist in player_actions for x in sublist]
            if len(all_agents) < 2:
                continue
            #emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ne_list = []
            if self.with_baseline_weights:
                ne_res_file_list = [os.path.join(self.nash_eq_res_cache,file_str+'_high_baselineW.ne')]
            else:
                ne_res_file_list = [os.path.join(self.nash_eq_res_cache,file_str+'_high.ne'), os.path.join(self.nash_eq_res_cache,file_str+'_low.ne')]
            for res_f in ne_res_file_list:
                with open(res_f) as f:
                    ne_res_high_bounds = f.readlines()
                    for l in ne_res_high_bounds:
                        code_str = l.rstrip().split(',')[1:]
                        if len(code_str) != len(all_acts):
                            continue
                        ne_tuple = tuple([x for i,x in enumerate(all_acts) if int(code_str[i]) == 1])
                        if ne_tuple not in ne_list:
                            ne_list.append(ne_tuple)
                if len(ne_list) > 0 and len(emp_acts) > 0:
                    for emp_act in emp_acts:
                        for ag_idx in np.arange(len(all_agents)):
                            if all_agents[ag_idx][1] != 0:
                                continue
                            emp_act_str = utils.get_l1_action_string(int(emp_act[ag_idx][9:11]))
                            if emp_act_str not in confusion_dict:
                                confusion_dict[emp_act_str] = []
                            if emp_act[ag_idx] in [x[ag_idx] for x in ne_list]:
                                confusion_dict[emp_act_str].append(emp_act_str)
                            else:
                                non_eq_len = 0
                                for eq_act in set([x[ag_idx] for x in ne_list]):
                                    eq_act_str = utils.get_l1_action_string(int(eq_act[9:11]))
                                    if eq_act_str != emp_act_str:
                                        confusion_dict[emp_act_str].append(eq_act_str)
                                        non_eq_len += 1
                                    else:
                                        confusion_dict[emp_act_str].append(emp_act_str)
                                ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                                confusion_dict[emp_act_str] += [emp_act_str]*(non_eq_len-1)
        confusion_key_list = list(confusion_dict.keys())
        confusion_matrix = []
        for k in confusion_key_list:
            confusion_arr = []
            ctr = Counter(confusion_dict[k])
            _sum = sum(ctr.values())
            for p_k in confusion_key_list:
                th_prob = ctr[p_k]/_sum
                confusion_arr.append(th_prob)
            confusion_matrix.append(confusion_arr)
        df_cm = pd.DataFrame(confusion_matrix, index = confusion_key_list,
                      columns = confusion_key_list)
        plt.figure()
        chart = sn.heatmap(df_cm, annot=True)
        sn.set(font_scale=.50)
        plt.xticks(rotation=75)
        plt.yticks(rotation=0)
        plt.xlabel('Equilibrium action', labelpad=15)
        plt.ylabel('Empirical action')
        b, t = plt.ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        plt.ylim(b, t)
        plt.savefig('sample.png', bbox_inches='tight')
        plt.show()
        f=1
        
    
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            #print('processing',ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            agents_on_rule = [[False]*len(all_agents)]*len(emp_acts)
            if len(emp_acts) > 0 and len(rule_acts) > 0:
                for emp_idx,e in enumerate(emp_acts):
                    for i in np.arange(len(all_agents)):
                        if e[i] in [x[i] for x in rule_acts]:
                            agents_on_rule[emp_idx][i] = True
                        else:
                            emp_act_str = utils.get_l1_action_string(int(e[i][9:11]))
                            all_rule_str = set([utils.get_l1_action_string(int(x[i][9:11])) for x in rule_acts])
                            if emp_act_str in constants.WAIT_ACTIONS and len(all_rule_str & set(constants.WAIT_ACTIONS)) > 0:
                                agents_on_rule[emp_idx][i] = True
            self.agents_on_rule = agents_on_rule
            if len(list(set(emp_acts) & set(rule_acts))) > 0:
                on_rule = True
            else:
                on_rule = False
            obj_utils = [payoff_dict[x] for x in emp_acts if x in payoff_dict]
            if len(obj_utils) == 0:
                continue
            else:
                for emp_idx,obj_utils in enumerate(obj_utils):
                    constr_utils = [v-obj_utils for k,v in payoff_dict.items() if k not in emp_acts]
                    solns = self.formulate_and_solve_lp(all_agents,obj_utils,constr_utils,self.agents_on_rule[emp_idx])
                    for s in solns:
                        ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                        ag_obj = self.agent_object_dict[s[0]]
                        leading_veh_id = ag_obj.leading_vehicle.id if ag_obj.leading_vehicle is not None else None
                        relev_agents = []
                        for a in all_agents:
                            if a[1] != 0 and a[1] != leading_veh_id:
                                relev_agents.append(a[1])
                        relev_agents = 'Y'  if len(relev_agents)>0 else 'N'
                        next_signal_change = utils.get_time_to_next_signal(time_ts, ag_obj.direction, ag_obj.signal)
                        if next_signal_change[0] is not None:
                            if next_signal_change[0] < 10:
                                time_to_change = 'LT 10'
                            elif 10 <= next_signal_change[0] <= 30:
                                time_to_change = '10-30'
                            else:
                                time_to_change = 'GT 30'
                        else:
                            time_to_change = None
                        if ag_obj.speed < .3:
                            speed_fact = 'NEAR STATIONARY'
                        elif .3 <= ag_obj.speed < 3:
                            speed_fact = 'SLOW'
                        elif 3 <= ag_obj.speed <= 14:
                            speed_fact = 'MEDIUM'
                        else:
                            speed_fact = 'HIGH'
                            
                        #ag_state_info = (ag_obj.signal, str((int(next_signal_change[0]),next_signal_change[1])), ag_obj.current_segment, int(ag_obj.speed), leading_veh_id, len(relev_agents), 'Y' if ag_obj.relev_pedestrians is not None else 'N')
                        ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','NASH', s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                        print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()

class InverseQlkRGameLearningModel(InverseGameLearningModel):
    
    
    def formulate_and_solve_lp_qlkr(self,all_agents,all_obj_mat,all_constr_mat,on_rule_list,this_idx):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if idx == this_idx:
                if on_rule_list[idx]:
                    all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
                else:
                    obj_mat = all_obj_mat[:,idx]
                    constr_mat = [x[:,idx] for x in all_constr_mat]
                    num_params = 2 if self.agent_object_dict[ag].relev_pedestrians is None else 3
                    solns = solve_lp(obj_mat, constr_mat, num_params)
                    all_solns.append((ag,solns))
        return all_solns
    
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            #print('processing',ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            agents_on_rule = [[False]*len(all_agents)]*len(emp_acts)
            if len(emp_acts) > 0 and len(rule_acts) > 0:
                for emp_idx,e in enumerate(emp_acts):
                    for i in np.arange(len(all_agents)):
                        if e[i] in [x[i] for x in rule_acts]:
                            agents_on_rule[emp_idx][i] = True
                        else:
                            emp_act_str = utils.get_l1_action_string(int(e[i][9:11]))
                            all_rule_str = set([utils.get_l1_action_string(int(x[i][9:11])) for x in rule_acts])
                            if emp_act_str in constants.WAIT_ACTIONS and len(all_rule_str & set(constants.WAIT_ACTIONS)) > 0:
                                agents_on_rule[emp_idx][i] = True
            self.agents_on_rule = agents_on_rule
            if len(list(set(emp_acts) & set(rule_acts))) > 0:
                on_rule = True
            else:
                on_rule = False
            if len(emp_acts) == 0 or len(rule_acts) == 0:
                continue
            for idx,ag in enumerate(all_agents):
                ''' select the observed action for idx and rule acts for the rest in act_tups '''
                act_tups = []
                for e_a in emp_acts:
                    this_strat = []
                    for idx2 in np.arange(len(all_agents)):
                        if idx2 == idx:
                            this_strat.append(e_a[idx2])
                        else:
                            this_strat.append(rule_acts[0][idx2])
                    act_tups.append(tuple(this_strat))   
                
                obj_utils = [payoff_dict[x] for x in act_tups if x in payoff_dict]
                if len(obj_utils) == 0:
                    continue
                else:
                    for emp_idx,obj_utils in enumerate(obj_utils):
                        constr_utils = [v-obj_utils for k,v in payoff_dict.items() if k not in act_tups]
                        solns = self.formulate_and_solve_lp_qlkr(all_agents,obj_utils,constr_utils,self.agents_on_rule[emp_idx],idx)
                        for s in solns:
                            ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                            ag_obj = self.agent_object_dict[s[0]]
                            leading_veh_id = ag_obj.leading_vehicle.id if ag_obj.leading_vehicle is not None else None
                            relev_agents = []
                            for a in all_agents:
                                if a[1] != 0 and a[1] != leading_veh_id:
                                    relev_agents.append(a[1])
                            relev_agents = 'Y'  if len(relev_agents)>0 else 'N'
                            next_signal_change = utils.get_time_to_next_signal(time_ts, ag_obj.direction, ag_obj.signal)
                            if next_signal_change[0] is not None:
                                if next_signal_change[0] < 10:
                                    time_to_change = 'LT 10'
                                elif 10 <= next_signal_change[0] <= 30:
                                    time_to_change = '10-30'
                                else:
                                    time_to_change = 'GT 30'
                            else:
                                time_to_change = None
                            if ag_obj.speed < .3:
                                speed_fact = 'NEAR STATIONARY'
                            elif .3 <= ag_obj.speed < 3:
                                speed_fact = 'SLOW'
                            elif 3 <= ag_obj.speed <= 14:
                                speed_fact = 'MEDIUM'
                            else:
                                speed_fact = 'HIGH'
                                
                            #ag_state_info = (ag_obj.signal, str((int(next_signal_change[0]),next_signal_change[1])), ag_obj.current_segment, int(ag_obj.speed), leading_veh_id, len(relev_agents), 'Y' if ag_obj.relev_pedestrians is not None else 'N')
                            ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','QlkR', s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                            print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()


class InverseCorrelatedEquilibriaLearningModel(InverseGameLearningModel):
    
    def formulate_and_solve_lp_qlkr(self,all_agents,obj_mat,constr_mat,on_rule,this_idx,file_str):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if idx == this_idx:
                if on_rule:
                    all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
                else:
                    num_params = 2 if self.all_agent_obj_dict[file_str][ag].relev_pedestrians is None else 3
                    solns = solve_lp(obj_mat, constr_mat, num_params)
                    all_solns.append((ag,solns))
        return all_solns
    
    
    def build_emp_distribution(self):
        ct,N = 0,len(self.file_keys)
        emp_act_distr = dict()
        for file_str in self.file_keys:
            ct +=1
            print('processing',ct,'/',N)
            #if ct > 20:
            #    continue
            
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_actions = [''.join(list(set([ str(y[9:11]) for y in x ]))) for x in player_actions]
            if len(all_agents) < 2:
                continue
            if num_players not in emp_act_distr:
                emp_act_distr[num_players] = dict()
            game_in_dict = False
            for k in emp_act_distr[len(all_agents)].keys():
                game_act_code = k
                if set(game_act_code) == set(player_actions):
                    game_in_dict = True
                    for e_a in emp_acts:
                        strat_code = tuple([str(x[9:11]) for x in e_a])
                        emp_act_distr[len(all_agents)][k].append(strat_code)
            if not game_in_dict:
                if len(emp_acts) > 0:
                    emp_act_distr[len(all_agents)][tuple(player_actions)] = []
                    for e_a in emp_acts:
                        strat_code = tuple([str(x[9:11]) for x in e_a])
                        emp_act_distr[len(all_agents)][tuple(player_actions)].append(strat_code)
        for k,v in emp_act_distr.items():
            key_list = list(v.keys())
            for k in key_list:
                ctr = Counter(v[k])
                _sum = sum(ctr.values())
                for ctr_k in ctr.keys():
                    ctr[ctr_k] = ctr[ctr_k]/_sum
                v[k] = ctr
        self.emp_act_distr = emp_act_distr
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            print('processing',ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_action_codes = [''.join(list(set([ str(y[9:11]) for y in x ]))) for x in player_actions]
            if len(all_agents) < 2:
                continue
            if tuple(player_action_codes) not in self.emp_act_distr[num_players]:
                ''' we haven't seen this game matrix before, so continue'''
                continue
                    
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            agents_on_rule = [[False]*len(all_agents)]*len(emp_acts)
            if len(emp_acts) > 0 and len(rule_acts) > 0:
                for emp_idx,e in enumerate(emp_acts):
                    for i in np.arange(len(all_agents)):
                        if e[i] in [x[i] for x in rule_acts]:
                            agents_on_rule[emp_idx][i] = True
                        else:
                            emp_act_str = utils.get_l1_action_string(int(e[i][9:11]))
                            all_rule_str = set([utils.get_l1_action_string(int(x[i][9:11])) for x in rule_acts])
                            if emp_act_str in constants.WAIT_ACTIONS and len(all_rule_str & set(constants.WAIT_ACTIONS)) > 0:
                                agents_on_rule[emp_idx][i] = True
            self.agents_on_rule = agents_on_rule
            for idx,ag in enumerate(all_agents):
                act_tups = []
                replaced_p_a = list(player_actions) 
                replaced_p_a[idx] = [None]*len(replaced_p_a[idx])
                
                obj_strats = []
                for this_ag_strat in player_actions[idx]:
                    other_agent_act_combinations = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                    for idx_2,s in enumerate(other_agent_act_combinations):
                        _s = list(s)
                        _s[idx] = this_ag_strat
                        other_agent_act_combinations[idx_2] = tuple(_s)
                    prob_vect_for_game = self.emp_act_distr[num_players][tuple(player_action_codes)]
                    prob_vect = []
                    for all_s in other_agent_act_combinations:
                        s_code = tuple([x[9:11] for x  in list(all_s)])
                        if s_code in prob_vect_for_game:
                            prob_vect.append(prob_vect_for_game[s_code])
                        else:
                            prob_vect.append(0)
                    _sum = sum(prob_vect)
                    if _sum == 0:
                        continue
                    prob_vect = [x/_sum for x in prob_vect]
                    obj_util_forall_acts = [y[:,idx] for y in [payoff_dict[x] for x in other_agent_act_combinations if x in payoff_dict]]
                    obj_utils = sum([obj_util_forall_acts[x]*prob_vect[x] for x in np.arange(len(prob_vect))])
                    constr_list = None
                    for this_ag_oth_strat in player_actions[idx]:
                        if this_ag_strat != this_ag_oth_strat:
                            other_agent_act_combinations = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                            for idx_2,s in enumerate(other_agent_act_combinations):
                                _s = list(s)
                                _s[idx] = this_ag_oth_strat
                                other_agent_act_combinations[idx_2] = tuple(_s)
                            constr_util_forall_acts = [y[:,idx] for y in [payoff_dict[x] for x in other_agent_act_combinations if x in payoff_dict]]
                            constr_utils = sum([constr_util_forall_acts[x]*prob_vect[x] for x in np.arange(len(prob_vect))])
                            constr_diff = obj_utils - constr_utils
                            if np.any(constr_diff):
                                if constr_list is None:
                                    constr_list = np.array(constr_diff).reshape((1,3)) 
                                else:
                                    constr_list = np.append(constr_list, constr_diff.reshape((1,3)), 0)
                    if constr_list is None:
                        continue
                    constr_list = np.unique(constr_list, axis=0)
                    if any([x[idx] for x in self.agents_on_rule]):
                        on_rule = True
                    else:
                        on_rule = False
                    solns = self.formulate_and_solve_lp_qlkr(all_agents,obj_utils,constr_list,on_rule,idx,file_str)
                    for s in solns:
                        ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                        ag_obj = self.all_agent_obj_dict[file_str][s[0]]
                        leading_veh_id = ag_obj.leading_vehicle.id if ag_obj.leading_vehicle is not None else None
                        relev_agents = []
                        for a in all_agents:
                            if a[1] != 0 and a[1] != leading_veh_id:
                                relev_agents.append(a[1])
                        relev_agents = 'Y'  if len(relev_agents)>0 else 'N'
                        next_signal_change = utils.get_time_to_next_signal(time_ts, ag_obj.direction, ag_obj.signal)
                        if next_signal_change[0] is not None:
                            if next_signal_change[0] < 10:
                                time_to_change = 'LT 10'
                            elif 10 <= next_signal_change[0] <= 30:
                                time_to_change = '10-30'
                            else:
                                time_to_change = 'GT 30'
                        else:
                            time_to_change = None
                        if ag_obj.speed < .3:
                            speed_fact = 'NEAR STATIONARY'
                        elif .3 <= ag_obj.speed < 3:
                            speed_fact = 'SLOW'
                        elif 3 <= ag_obj.speed <= 14:
                            speed_fact = 'MEDIUM'
                        else:
                            speed_fact = 'HIGH'
                            
                        #ag_state_info = (ag_obj.signal, str((int(next_signal_change[0]),next_signal_change[1])), ag_obj.current_segment, int(ag_obj.speed), leading_veh_id, len(relev_agents), 'Y' if ag_obj.relev_pedestrians is not None else 'N')
                        ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','CorrEq', s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                        print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
            
'''
def read_l3_tree():
    file_id = '769'
    N = None
    for k in payoff_dict.keys():
        orig_payoff = payoff_dict[k]
        if N is None:
            N = len(k)
            for ag in k:
                if int(ag[6:9]) == 0:
                    agents.append(int(ag[3:6]))
                else:
                    agents.append(int(ag[6:9]))
        weights = np.reshape(np.asarray([0.25,0.25,0.5]), newshape=(1,3))
        weighted_payoffs = weights @ orig_payoff
        payoff_dict[k] = np.reshape(weighted_payoffs, newshape=(N,))
    for k,v in acts_dict.items():
        if k[1] in agents:
            brk = 1
    
    f=1
    eq = EquilibriaCore(N,payoff_dict,len(payoff_dict),None,None)
    eq.transform_to_nfg_format()
    ne_all = eq.calc_pure_strategy_nash_equilibrium_exhaustive()
    for k,v in ne_all.items():
        print(k,v)
'''
'''
with_baseline_weights = False
mixed_weights = False
l1_model = InverseGameLearningModel('769',with_baseline_weights,mixed_weights)
#l1_model.calculate_and_insert_weights()
l1_model.convert_to_nfg()
l1_model.invoke_pure_strat_nash_eq_calc()
l1_model.calc_confusion_matrix()
'''

'''
with_baseline_weights = False
mixed_weights = False
l1_model = InverseQlkRGameLearningModel('769',with_baseline_weights,mixed_weights)        
l1_model.calculate_and_insert_weights()
'''
'''
with_baseline_weights = False
mixed_weights = False
l1_model = InverseCorrelatedEquilibriaLearningModel('769',with_baseline_weights,mixed_weights)
l1_model.build_emp_distribution()
l1_model.calculate_and_insert_weights()
'''