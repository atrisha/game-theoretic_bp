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
from all_utils.lp_utils import solve_lp, solve_lp_multivar
import csv
import os
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize, Bounds
import constants
from subprocess import check_output, CalledProcessError
import math
import json

class L1_Model_Config():
    
    def __init__(self):
        self.update_only = False
        self.model_type = None
        self.emp_acts_dict = None
        self.rule_acts_dict = None
        self.agent_object_dict = None
        self.old_model = False
        self.ignore_rule = False
        self.ignore_low = False
        self.is_transformed = False
        
    def set_run_type(self,run_type):
        self.run_type = run_type
    
    def set_no_insertion(self,no_insertion):
        self.no_insertion = no_insertion
        
    def set_l1_model_type(self,l1_model_type):
        self.l1_model_type = l1_model_type
    
    def set_baseline_weights_flag(self, with_baseline_weights):
        self.with_baseline_weights = with_baseline_weights
        
    def set_mixed_weights_flag(self, mixed_weights):
        self.mixed_weights = mixed_weights
        
    def set_update_only(self,update_only):
        ''' Only used for inserting weights'''
        self.update_only = update_only
        
    def set_l3_model_def(self,sampling_type,soln_type):
        self.l3_sampling = sampling_type
        self.l3_soln = soln_type
        
    def set_lzero_behavior(self,lzero_behavior):
        ''' Only used for Ql1 models'''
        self.lzero_behavior = lzero_behavior
        
    def set_file_id(self,file_id):
        self.file_id = file_id
        
    def set_l3_weights_estimation_model_type(self,model_type):
        self.model_type = model_type
        
    def set_prev_version_params(self):
        self.is_transformed = True
        self.ignore_rule = True
        self.ignore_low = True
        self.old_model = True
        
    def get_model_parms_str(self):
        model_parms=[]
        if self.l3_soln is not None:
            model_parms.append('l3_soln='+str(self.l3_soln))
        else:
            model_parms.append('l3_soln=NA')
        if self.l3_sampling is not None:
            model_parms.append('l3_sampling='+str(self.l3_sampling))
        else:
            model_parms.append('l3_sampling=NA')
        if self.mixed_weights is not None:
            model_parms.append('mixed_weights='+str(self.mixed_weights))
        else:
            model_parms.append('mixed_weights=NA')
        if self.with_baseline_weights is not None:
            model_parms.append('baseline_weights='+str(self.with_baseline_weights))
        else:
            model_parms.append('baseline_weights=NA')
        if self.old_model is not None:
            model_parms.append('old_model='+str(self.old_model))
        if self.l1_model_type == 'Ql1':
            if self.lzero_behavior is not None:
                model_parms.append('lzero_behavior='+str(self.lzero_behavior))
            else:
                model_parms.append('lzero_behavior=NA')
        
        return ','.join(model_parms)
        
class Rules():
    
    def is_relevant(self,veh_state,pedestrian_info):
        v_id = veh_state.id
        relev_xwalk = utils.get_relevant_crosswalks(v_id)
        is_relevant = False
        if relev_xwalk is not None:
            crosswalk_ids,near_gate = relev_xwalk[0], relev_xwalk[1]
            #current_segment = all_utils.get_current_segment_by_veh_id(v_id,self.eq_context.curr_time)
            for xwalk in crosswalk_ids:
                for ped_state in pedestrian_info:
                    if xwalk in ped_state.crosswalks:
                        if ped_state.crosswalks[xwalk]['location'] == constants.ON_CROSSWALK:
                            is_relevant = True
                        elif ped_state.crosswalks[xwalk]['location'] == constants.BEFORE_CROSSWALK \
                                and ped_state.crosswalks[xwalk]['dist_to_entry'] < constants.PEDESTRIAN_CROSSWALK_DIST_THRESH \
                                    and ((ped_state.crosswalks[xwalk]['next_change'][1] == 'G' and ped_state.crosswalks[xwalk]['next_change'][0] <= constants.PEDESTRIAN_CROSSWALK_TIME_THRESH) \
                                             or ped_state.crosswalks[xwalk]['next_change'][1] == 'R'):
                            ''' before the crosswalk, within the distance threshold, and the signal is green or about to change to green'''
                            is_relevant = True
                        else:
                            is_relevant = False
        return is_relevant
    
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
            is_pedestrian_relevant = self.is_relevant(veh_state,relev_pedestrians)
            if is_pedestrian_relevant:
                key_tuples.append(['Y','*'])
            else:
                key_tuples.append(['N','*'])
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
    
    def __init__(self,l1_model_config):
        self.l1_model_config = l1_model_config
        self.mixed_weights = l1_model_config.mixed_weights
        self.with_baseline_weights = l1_model_config.with_baseline_weights
        self.file_id = l1_model_config.file_id
        file_id = l1_model_config.file_id
        update_only = l1_model_config.update_only
        model_type = l1_model_config.model_type
        self.l3_cache_str = constants.CACHE_DIR+"l3_trees_"+l1_model_config.l3_sampling+"_"+l1_model_config.l3_soln+"\\"+self.file_id
        self.weighted_l3_cache_str = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\weighted_l3_trees\\"+self.file_id
        self.nash_eq_res_cache = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\nash_eq_res\\"+self.file_id
        if not os.path.exists(self.weighted_l3_cache_str):
            os.makedirs(self.weighted_l3_cache_str)
        if not os.path.exists(self.nash_eq_res_cache):
            os.makedirs(self.nash_eq_res_cache)
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
        if l1_model_config.l3_sampling == 'BOUNDARY' and not l1_model_config.old_model:
            self.file_keys = [x for x in self.file_keys if x[-3:]!='low']
        if self.l1_model_config.run_type == 'calculate_and_insert_weights':
            if update_only:
                q_string = "select * from UTILITY_WEIGHTS where MODEL_TYPE = '"+self.l1_model_config.l1_model_type+"' and L3_MODEL_TYPE = '"+self.l1_model_config.model_type+"'"
                c.execute(q_string)
                res = c.fetchall()   
                files_in_db = [str(x[1])+'_'+str.replace(str(x[2]),'.',',') for x in res]
                self.file_keys = [x for x in self.file_keys if x not in files_in_db]
            else:
                q_string = "delete from UTILITY_WEIGHTS where MODEL_TYPE = '"+self.l1_model_config.l1_model_type+"' and L3_MODEL_TYPE = '"+self.l1_model_config.model_type+"'"
                c.execute(q_string)
                conn.commit()
            
        self.load_state_info()
        if self.l1_model_config.emp_acts_dict is None or self.l1_model_config.rule_acts_dict is None or self.l1_model_config.agent_object_dict is None:
            self.load_emp_and_rule_acts()
        else:
            self.emp_acts_dict = self.l1_model_config.emp_acts_dict
            self.rule_acts_dict = self.l1_model_config.rule_acts_dict
            self.agent_object_dict = self.l1_model_config.agent_object_dict
    
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
        self.emp_acts_dict, self.rule_acts_dict, self.agent_object_dict = dict(), dict(), dict()
        for file_str in self.file_keys:
            ct +=1
            #if ct > 30:
            #    continue
            print('processing',self.file_id,ct,'/',N,file_str)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            emp_acts,rule_acts,agent_object_dict = self.get_emp_and_rule_actions(file_str,payoff_dict)
            self.emp_acts_dict[file_str] = emp_acts
            self.rule_acts_dict[file_str] = rule_acts
            self.agent_object_dict[file_str] = agent_object_dict
        self.l1_model_config.emp_acts_dict = self.emp_acts_dict
        self.l1_model_config.rule_acts_dict = self.rule_acts_dict
        self.l1_model_config.agent_object_dict = self.agent_object_dict
            
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
                print('vehicle state not found in cache '+ag_file_key+' ......setting up')
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
        return (emp_acts,rule_acts,agent_object_dict)
     
class InverseNashEqLearningModel(L1ModelCore):
    
    def solve(self):
        filelist = [ f for f in os.listdir(self.weighted_l3_cache_str)]
        for f in filelist:
            os.remove(os.path.join(self.weighted_l3_cache_str, f))
        filelist = [ f for f in os.listdir(self.nash_eq_res_cache)]
        for f in filelist:
            os.remove(os.path.join(self.nash_eq_res_cache, f))
        
        self.convert_to_nfg()
        self.invoke_pure_strat_nash_eq_calc()
        #self.calc_confusion_matrix()
    
    def add_missing_attributes(self,veh_state):
        if not hasattr(veh_state, 'relev_crosswalks'):
            relev_crosswalks = utils.get_relevant_crosswalks(veh_state)
            veh_state.set_relev_crosswalks(relev_crosswalks)
        veh_state.has_oncoming_vehicle = False
        
    
            
    def formulate_and_solve_lp(self,all_agents,all_obj_mat,all_constr_mat,on_rule_list,file_str):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if on_rule_list[idx]:
                all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
            else:
                obj_mat = all_obj_mat[:,idx]
                constr_mat = [x[:,idx] for x in all_constr_mat]
                num_params = 2 if self.agent_object_dict[file_str][ag].relev_pedestrians is None else 3
                solns = solve_lp(obj_mat, constr_mat, num_params)
                all_solns.append((ag,solns))
        return all_solns
    
    
    def convert_to_nfg(self):
        
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ct,N = 0,len(self.file_keys)
        ''' UTILITY_WEIGHTS TABLE also stores the agent state info information, so we can retrieve that instead of building it again '''
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 30:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            if len(all_agents) < 2:
                continue
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict}
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            #emp_acts,rule_acts = self.get_emp_and_rule_actions(file_str,payoff_dict)
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ''' older model did not take into account rules, so this flag is for backward compatibility '''
            if self.l1_model_config.ignore_rule:
                rule_acts = []
            self.weights_info = [[] for i in range(num_players)]
            if not self.l1_model_config.is_transformed:
                for ag_idx,ag in enumerate(all_agents):
                    ag_id = ag[0] if ag[1]==0 else ag[1]
                    ag_info_key = (int(self.file_id),ag_id,time_ts)
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    self.weights_info[ag_idx] = ag_weights
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
            else:
                high_bounds_transformed_payoffs = payoff_dict
            ''' validate that we are indeed running the older model'''
            if self.l1_model_config.is_transformed:
                assert self.with_baseline_weights
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
            print('processing',self.file_id,ct,'/',N)
            try:
                out = check_output('"C:\\Program Files (x86)\\Gambit\\gambit-enumpure.exe" -q "'+self.weighted_l3_cache_str+'\\'+file_str+'"', shell=True).decode()
                outfile_loc = os.path.join(self.nash_eq_res_cache,file_str.split('.')[0]+'.ne')
                returncode = 0
            except CalledProcessError:
                returncode = 1
            if returncode == 0:
                text_file = open(outfile_loc, "w")
                text_file.write(out)
                text_file.close()
            
    def calc_confusion_matrix(self):
        ct,N = 0,len(self.file_keys)
        confusion_dict,confusion_dict_N = dict(),dict()
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        ins_list = []
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N)
            #if ct > 30:
            #    continue
            
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            self.weights_info = [[] for i in range(num_players)]
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
                f=1
                opt_strat = ne_list
                opt_strat = list(set(opt_strat))
                
                all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
                num_players = len(all_agents)
                player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]           
                '''
                eq = equilibria_core.EquilibriaCore(num_players,payoff_dict,len(payoff_dict),player_actions[0],False)
                soln = eq.calc_pure_strategy_nash_equilibrium_exhaustive()
                for s in soln.keys():
                    if s not in opt_strat:
                        f=1
                for s in opt_strat:
                    if s not in list(soln.keys()):
                        f=1
                '''
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    if not self.l1_model_config.old_model:
                        emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                        emp_utils = np.asarray(emp_utils)
                        assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            emp_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                        else:
                            emp_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    if not self.l1_model_config.old_model:
                        assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    if not self.l1_model_config.old_model:
                        soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                        soln_utils = np.asarray(soln_utils)
                        assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            soln_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                        else:
                            soln_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                for idx,ag in enumerate(all_agents):
                    ag_id = ag[0] if ag[1]==0 else ag[1]
                    ag_id = ag[0] if ag[1]==0 else ag[1]
                    ag_info_key = (int(self.file_id),ag_id,time_ts)
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    self.weights_info[idx] = ag_weights
                    #try:
                    if not self.l1_model_config.old_model or not self.l1_model_config.is_transformed:
                        emp_utils = np.asarray(emp_utils)
                        soln_utils = np.asarray(soln_utils)
                        self.weights_info[idx] = np.asarray(self.weights_info[idx])
                        try:
                            soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(self.weights_info[idx].tolist()) if self.weights_info[idx] is not None and not self.l1_model_config.old_model  else None)
                        except:
                            f=1
                    else:
                        soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils) if emp_utils is not None else None,None,json.dumps(soln_utils) if soln_utils is not None else None,None)
                    '''
                    except:
                        if self.weights_info[idx] is not None:
                            a = self.weights_info[idx].tolist()
                        if soln_utils is not None:
                            b = soln_utils.tolist()
                        if rule_utils is not None:
                            c = rule_utils.tolist()
                        if emp_utils is not None:
                            d = emp_utils.tolist()
                        f=1
                    '''
                    #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                    ins_list.append(soln_vect)
                
                if len(ne_list) > 0 and len(emp_acts) > 0:
                    for emp_act in emp_acts:
                        for ag_idx in np.arange(len(all_agents)):
                            #if all_agents[ag_idx][1] != 0:
                            #    continue
                            emp_act_str = utils.get_l1_action_string(int(emp_act[ag_idx][9:11]))
                            if emp_act_str not in confusion_dict:
                                confusion_dict[emp_act_str] = []
                            if emp_act_str not in confusion_dict_N:
                                confusion_dict_N[emp_act_str] = 0
                            if emp_act[ag_idx] in [x[ag_idx] for x in ne_list]:
                                confusion_dict[emp_act_str].append(emp_act_str)
                                confusion_dict_N[emp_act_str] += 1
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
                                confusion_dict_N[emp_act_str] += 1
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
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
        df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(confusion_dict_N[x])+')' for x in confusion_key_list],
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
        plt.savefig(str(constants.CURRENT_FILE_ID)+'_NASH_'+self.l1_model_config.model_type+'_'+str(self.with_baseline_weights)+'_'+str(self.mixed_weights)+'.png', bbox_inches='tight')
        #plt.show()
        
        
    
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            #print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
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
                    solns = self.formulate_and_solve_lp(all_agents,obj_utils,constr_utils,self.agents_on_rule[emp_idx],file_str)
                    for s in solns:
                        ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                        ag_obj = self.agent_object_dict[file_str][s[0]]
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
                        ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','NASH',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                        print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()



class InverseQlkRGameLearningModel(InverseNashEqLearningModel):
    
    
    def keywithmaxval(self,d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=np.asarray(list(d.values()))
        k=list(d.keys())
        max_v = max(v)
        indices = np.where(v == max_v)[0]
        return [k[x] for x in indices],max_v
    
    def solve(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        ins_list = []
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            if len(rule_acts) == 0:
                continue
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            weights_info = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                    ag_weights = weights_dict[ag_info_map[ag_info_key]]
                else:
                    ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                weights_info[idx] = ag_weights
                low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict}
                high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
                for k in payoff_dict.keys():
                    orig_payoff = payoff_dict[k]
                    
                    high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                    low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                    ag_orig_payoff = orig_payoff[:,idx]
                    if self.mixed_weights:
                        if np.sum(high_weights) != 0:
                            high_weights = high_weights/np.sum(high_weights)
                        if np.sum(low_weights) != 0:
                            low_weights = low_weights/np.sum(low_weights)
                        
                    else:
                        high_weights = np.where(high_weights == max(high_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    low_weights = np.exp(10*low_weights)/np.sum(np.exp(10*low_weights))
                    high_weighted_payoffs = high_weights @ ag_orig_payoff
                    low_weighted_payoffs = low_weights @ ag_orig_payoff
                    low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                    high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                opt_strat = []
                for r in rule_acts:
                    this_player_strats = []
                    for this_player_action in player_actions[idx]:
                        _r = list(r)
                        _r[idx] = this_player_action
                        this_player_strats.append(tuple(_r))
                    this_player_payoffdict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items() if k in this_player_strats}
                    if len(this_player_payoffdict) > 0:
                        s_star, v_star = self.keywithmaxval(this_player_payoffdict)
                        opt_strat.extend(s_star)
                        for s in s_star:
                            this_s_act_code = s[idx][9:11]
                            if this_s_act_code not in player_acts_solns[idx]: 
                                player_acts_solns[idx].append(this_s_act_code)
                    this_player_payoffdict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items() if k in this_player_strats}
                    if len(this_player_payoffdict) > 0:
                        s_star, v_star = self.keywithmaxval(this_player_payoffdict)
                        opt_strat.extend(s_star)
                        for s in s_star:
                            this_s_act_code = s[idx][9:11]
                            if this_s_act_code not in player_acts_solns[idx]: 
                                player_acts_solns[idx].append(this_s_act_code)
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                opt_strat = list(set(opt_strat))
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                    emp_utils = np.asarray(emp_utils)
                    assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                    soln_utils = np.asarray(soln_utils)
                    assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None else None)
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
            
                        
    def calc_confusion_matrix(self,model_str=None):
        self.confusion_dict = {utils.get_l1_action_string(int(k)):[utils.get_l1_action_string(int(x)) for x in v] for k,v in self.confusion_dict.items()}
        self.confusion_dict_N = {utils.get_l1_action_string(int(k)):v for k,v in self.confusion_dict_N.items()}
        confusion_key_list = list(self.confusion_dict.keys())
        confusion_matrix = []
        for k in confusion_key_list:
            confusion_arr = []
            ctr = Counter(self.confusion_dict[k])
            _sum = sum(ctr.values())
            for p_k in confusion_key_list:
                th_prob = ctr[p_k]/_sum
                confusion_arr.append(th_prob)
            confusion_matrix.append(confusion_arr)
        df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(self.confusion_dict_N[x])+')' for x in confusion_key_list],
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
        model_str = 'qlkr_' if model_str is None else model_str
        plt.savefig(str(constants.CURRENT_FILE_ID)+'_'+model_str+'_'+self.l1_model_config.model_type+'_'+str(self.with_baseline_weights)+'_'+str(self.mixed_weights)+'.png', bbox_inches='tight')
        #plt.show()
        f=1
                    
    
    
    def formulate_and_solve_lp_qlkr(self,all_agents,all_obj_mat,all_constr_mat,on_rule_list,this_idx,file_str):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if idx == this_idx:
                if on_rule_list[idx]:
                    all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
                else:
                    obj_mat = all_obj_mat[:,idx]
                    constr_mat = [x[:,idx] for x in all_constr_mat]
                    num_params = 2 if self.agent_object_dict[file_str][ag].relev_pedestrians is None else 3
                    solns = solve_lp(obj_mat, constr_mat, num_params)
                    all_solns.append((ag,solns))
        return all_solns
    
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            #print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
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
                        solns = self.formulate_and_solve_lp_qlkr(all_agents,obj_utils,constr_utils,self.agents_on_rule[emp_idx],idx,file_str)
                        for s in solns:
                            ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                            ag_obj = self.agent_object_dict[file_str][s[0]]
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
                            ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','QlkR',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                            print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
        

class InverseMaxMaxResponse(InverseQlkRGameLearningModel):
    
    
    def calculate_and_insert_weights(self):
        
        def _sum_utils(W):
            _max_oth_strats = max([W@x for x in self.curr_strat_subsets.values()])
            _max_emps_strat = max([W@x for x in self.curr_objstrat_subsets.values()])
            return _max_emps_strat - _max_oth_strats
        
        def _stopping_callback(xk,result_state):
            if result_state.nit >= 100:
                return True
            else:
                return False
            
        
        def _minimizing_problem(W):
            _obj_list = []
            ''' we need to change the sign since this is originally a maximization problem'''
            for x in self.curr_objstrat_subsets.values():
                _obj_list.append(W@x)
            return max(_obj_list)
        
        def _maximizing_problem(W):
            _obj_list = []
            for x in self.curr_objstrat_subsets.values():
                _obj_list.append(W@x)
            return -max(_obj_list)
            
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N,file_str)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            if len(all_agents) < 2:
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
            if len(list(set(emp_acts) & set(rule_acts))) > 0:
                on_rule = True
            else:
                on_rule = False
            if len(emp_acts) == 0 or len(rule_acts) == 0:
                continue
            for idx,ag in enumerate(all_agents):
                ''' select the observed action idx and the action that maximizes the utility of idx's observed action for the rest in act_tups '''
                act_tups = [dict()]*len(emp_acts)
                for emp_idx,e_a in enumerate(emp_acts):
                    ''' get the strategies with idx's observed action '''
                    strat_subsets = {k:v[:,idx] for k,v in payoff_dict.items() if k[idx] == e_a[idx]}
                    act_tups[emp_idx][e_a[idx]] = strat_subsets
                
                
                if len(act_tups) == 0:
                    continue
                else:
                    for emp_idx,obj_vect in enumerate(act_tups):
                        solns = []
                        if not self.agents_on_rule[emp_idx][idx]:
                            cons_list = []
                            for pl_act in player_actions[idx]:
                                if pl_act not in obj_vect:
                                    self.curr_strat_subsets = {k:v[:,idx] for k,v in payoff_dict.items() if  k[idx] != pl_act}
                                    self.curr_objstrat_subsets = obj_vect[emp_acts[emp_idx][idx]]
                                    nlc = NonlinearConstraint(_sum_utils, 0, np.inf)
                                    cons_list.append(nlc)
                            if len(self.curr_objstrat_subsets) == 0:
                                continue
                            if any([np.isnan(x).any() or np.isinf(x).any() for x in self.curr_strat_subsets.values()]) or any([np.isnan(x).any() or np.isinf(x).any() for x in self.curr_objstrat_subsets.values()]):
                                continue
                            lc = LinearConstraint(np.asarray([1,1,1]), 1, 1)
                            cons_list.append(lc)
                            res_obj_high = minimize(fun=_maximizing_problem, x0=np.asarray([0.25, 0.25, 0.5]), bounds=Bounds(lb=np.asarray([0,0,0]),ub=np.asarray([1,1,1])), method='trust-constr', constraints=cons_list, callback = _stopping_callback)
                            res_obj_high.x = np.round(res_obj_high.x,1)
                            res_obj_high.x = [x/np.sum(res_obj_high.x) for x in res_obj_high.x]
                            res_obj_low = minimize(fun=_minimizing_problem, x0=np.asarray([0.25, 0.25, 0.5]), bounds=Bounds(lb=np.asarray([0,0,0]),ub=np.asarray([1,1,1])), method='trust-constr', constraints=cons_list, callback = _stopping_callback)
                            res_obj_low.x = np.round(res_obj_low.x,1)
                            res_obj_low.x = [x/np.sum(res_obj_low.x) for x in res_obj_low.x]
                            solns = [(ag,[(res_obj_high.x[0],res_obj_low.x[0]),(res_obj_high.x[1],res_obj_low.x[1]),(res_obj_high.x[2],res_obj_low.x[2]),(0,0)])]
                        else:
                            solns = [(ag,[(0,0),(0,0),(0,0),(1,1)])]
                        for s in solns:
                            ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                            ag_obj = self.agent_object_dict[file_str][s[0]]
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
                            ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','maxmax',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                            print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
    
    def solve(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        ins_list = []
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ''' older model did not take into account rules, so this flag is for backward compatibility '''
            if self.l1_model_config.ignore_rule:
                rule_acts = []
            
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            weights_info = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if not self.l1_model_config.is_transformed:
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    weights_info[idx] = ag_weights
                    for k in payoff_dict.keys():
                        orig_payoff = payoff_dict[k]
                        if len(rule_acts) == 0:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                            ag_orig_payoff = orig_payoff[:,idx]
                        else:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                            if k[idx] in [x[idx] for x in rule_acts]:
                                ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                            else:
                                ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                        if not self.mixed_weights:
                            high_weights = np.where(high_weights == max(high_weights), 1, 0)
                            #if np.sum(np.exp(10*high_weights)) != 0:
                            #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                            low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        high_weighted_payoffs = high_weights @ ag_orig_payoff
                        high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                        low_weighted_payoffs = low_weights @ ag_orig_payoff
                        low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                else:
                    high_bounds_transformed_payoffs = payoff_dict
                opt_strat = []
                this_player_payoff_dict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items()}
                opt_strat_high,opt_val_high = self.keywithmaxval(this_player_payoff_dict)
                opt_strat.extend(opt_strat_high)
                player_acts_solns[idx] = list(set([x[idx][9:11] for x in opt_strat_high]))
                if not self.l1_model_config.ignore_low:
                    this_player_payoff_dict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items()}
                    opt_strat_low,opt_val_low = self.keywithmaxval(this_player_payoff_dict)
                    opt_strat.extend(opt_strat_low)
                    for l_s in list(set([x[idx][9:11] for x in opt_strat_low])):
                        if l_s not in player_acts_solns[idx]:
                            player_acts_solns[idx].append(l_s)
                
                opt_strat = list(set(opt_strat))
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    if not self.l1_model_config.old_model:
                        emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                        emp_utils = np.asarray(emp_utils)
                        assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            emp_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                        else:
                            emp_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    if not self.l1_model_config.old_model:
                        assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    if not self.l1_model_config.old_model:
                        soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                        soln_utils = np.asarray(soln_utils)
                        assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            soln_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                        else:
                            soln_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                if not self.l1_model_config.old_model or not self.l1_model_config.is_transformed:
                    emp_utils = np.asarray(emp_utils)
                    soln_utils = np.asarray(soln_utils)
                    weights_info[idx] = np.asarray(weights_info[idx])
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None and not self.l1_model_config.old_model else None)
                else:
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils) if emp_utils is not None else None,None,json.dumps(soln_utils) if soln_utils is not None else None,None)
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
        
        
        
        
class InverseBestResponseWithTrueBelief(InverseQlkRGameLearningModel):
    
    def calculate_and_insert_weights(self):
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
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
            if len(list(set(emp_acts) & set(rule_acts))) > 0:
                on_rule = True
            else:
                on_rule = False
            if len(emp_acts) == 0 or len(rule_acts) == 0:
                continue
            for idx,ag in enumerate(all_agents):
                ''' select the observed action for idx and as well as for the rest in act_tups '''
                act_tups = []
                for e_a in emp_acts:
                    this_strat = []
                    for idx2 in np.arange(len(all_agents)):
                        if idx2 == idx:
                            this_strat.append(e_a[idx2])
                        else:
                            this_strat.append(e_a[idx2])
                    act_tups.append(tuple(this_strat))   
                
                obj_utils = [payoff_dict[x] for x in act_tups if x in payoff_dict]
                if len(obj_utils) == 0:
                    continue
                else:
                    for emp_idx,obj_utils in enumerate(obj_utils):
                        constr_utils = [v-obj_utils for k,v in payoff_dict.items() if k not in act_tups]
                        solns = self.formulate_and_solve_lp_qlkr(all_agents,obj_utils,constr_utils,self.agents_on_rule[emp_idx],idx,file_str)
                        for s in solns:
                            ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                            ag_obj = self.agent_object_dict[file_str][s[0]]
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
                            ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','brtb',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                            print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
        
    
    def calc_confusion_matrix(self,model_str=None):
        self.confusion_dict = {utils.get_l1_action_string(int(k)):[utils.get_l1_action_string(int(x)) for x in v] for k,v in self.confusion_dict.items()}
        self.confusion_dict_N = {utils.get_l1_action_string(int(k)):v for k,v in self.confusion_dict_N.items()}
        confusion_key_list = list(self.confusion_dict.keys())
        confusion_matrix = []
        for k in confusion_key_list:
            confusion_arr = []
            ctr = Counter(self.confusion_dict[k])
            _sum = sum(ctr.values())
            for p_k in confusion_key_list:
                th_prob = ctr[p_k]/_sum
                confusion_arr.append(th_prob)
            confusion_matrix.append(confusion_arr)
        df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(self.confusion_dict_N[x])+')' for x in confusion_key_list],
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
        model_str = 'brtb_' if model_str is None else model_str
        model_type = self.l1_model_config.model_type
        plt.savefig(str(constants.CURRENT_FILE_ID)+'_'+model_str+'_'+model_type+'_'+str(self.with_baseline_weights)+'_'+str(self.mixed_weights)+'.png', bbox_inches='tight')
        #plt.show()
        f=1
    
    def solve(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        ins_list = []
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            if len(rule_acts) == 0:
                continue
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            weights_info = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                    ag_weights = weights_dict[ag_info_map[ag_info_key]]
                else:
                    ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                weights_info[idx] = ag_weights
                low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict}
                high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
                for k in payoff_dict.keys():
                    orig_payoff = payoff_dict[k]
                    if len(rule_acts) == 0:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                        ag_orig_payoff = orig_payoff[:,idx]
                    else:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                        if k[idx] in [x[idx] for x in rule_acts]:
                            ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                        else:
                            ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                    if self.mixed_weights:
                        if np.sum(high_weights) != 0:
                            high_weights = high_weights/np.sum(high_weights)
                        if np.sum(low_weights) != 0:
                            low_weights = low_weights/np.sum(low_weights)
                        
                    else:
                        high_weights = np.where(high_weights == max(high_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    low_weights = np.exp(10*low_weights)/np.sum(np.exp(10*low_weights))
                    high_weighted_payoffs = high_weights @ ag_orig_payoff
                    low_weighted_payoffs = low_weights @ ag_orig_payoff
                    low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                    high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                opt_strat = []
                for r in emp_acts:
                    this_player_strats = []
                    for this_player_action in player_actions[idx]:
                        _r = list(r)
                        _r[idx] = this_player_action
                        this_player_strats.append(tuple(_r))
                    this_player_payoffdict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items() if k in this_player_strats}
                    if len(this_player_payoffdict) == 0:
                        continue
                    s_star, v_star = self.keywithmaxval(this_player_payoffdict)
                    opt_strat.extend(s_star)
                    for s in s_star:
                        this_s_act_code = s[idx][9:11]
                        if this_s_act_code not in player_acts_solns[idx]: 
                            player_acts_solns[idx].append(this_s_act_code)
                    this_player_payoffdict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items() if k in this_player_strats}
                    s_star, v_star = self.keywithmaxval(this_player_payoffdict)
                    opt_strat.extend(s_star)
                    for s in s_star:
                        this_s_act_code = s[idx][9:11]
                        if this_s_act_code not in player_acts_solns[idx]: 
                            player_acts_solns[idx].append(this_s_act_code)
                opt_strat = list(set(opt_strat))
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                    emp_utils = np.asarray(emp_utils)
                    assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                    soln_utils = np.asarray(soln_utils)
                    assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None else None)
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
    
class InverseMaxMinResponse(InverseMaxMaxResponse):
    
    def calculate_and_insert_weights(self):
        
        def _sum_utils(W):
            _max_oth_strats = min([W@x for x in self.curr_strat_subsets.values()])
            _max_emps_strat = min([W@x for x in self.curr_objstrat_subsets.values()])
            return _max_emps_strat - _max_oth_strats
        
        def _stopping_callback(xk,result_state):
            if result_state.nit >= 100:
                return True
            else:
                return False
            
        
        def _minimizing_problem(W):
            _obj_list = []
            ''' we need to change the sign since this is originally a maximization problem'''
            for x in self.curr_objstrat_subsets.values():
                _obj_list.append(W@x)
            return max(_obj_list)
        
        def _maximizing_problem(W):
            _obj_list = []
            for x in self.curr_objstrat_subsets.values():
                _obj_list.append(W@x)
            return -max(_obj_list)
            
        ct,N = 0,len(self.file_keys)
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            if len(all_agents) < 2:
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
            if len(list(set(emp_acts) & set(rule_acts))) > 0:
                on_rule = True
            else:
                on_rule = False
            if len(emp_acts) == 0 or len(rule_acts) == 0:
                continue
            for idx,ag in enumerate(all_agents):
                ''' select the observed action idx and the action that maximizes the utility of idx's observed action for the rest in act_tups '''
                act_tups = [dict()]*len(emp_acts)
                for emp_idx,e_a in enumerate(emp_acts):
                    ''' get the strategies with idx's observed action '''
                    strat_subsets = {k:v[:,idx] for k,v in payoff_dict.items() if k[idx] == e_a[idx]}
                    act_tups[emp_idx][e_a[idx]] = strat_subsets
                
                
                if len(act_tups) == 0:
                    continue
                else:
                    for emp_idx,obj_vect in enumerate(act_tups):
                        solns = []
                        if not self.agents_on_rule[emp_idx][idx]:
                            cons_list = []
                            for pl_act in player_actions[idx]:
                                if pl_act not in obj_vect:
                                    self.curr_strat_subsets = {k:v[:,idx] for k,v in payoff_dict.items() if  k[idx] != pl_act}
                                    self.curr_objstrat_subsets = obj_vect[emp_acts[emp_idx][idx]]
                                    nlc = NonlinearConstraint(_sum_utils, 0, np.inf)
                                    cons_list.append(nlc)
                            if len(self.curr_objstrat_subsets) == 0:
                                continue
                            if any([np.isnan(x).any() or np.isinf(x).any() for x in self.curr_strat_subsets.values()]) or any([np.isnan(x).any() or np.isinf(x).any() for x in self.curr_objstrat_subsets.values()]):
                                continue
                            lc = LinearConstraint(np.asarray([1,1,1]), 1, 1)
                            cons_list.append(lc)
                            res_obj_high = minimize(fun=_maximizing_problem, x0=np.asarray([0.25, 0.25, 0.5]), bounds=Bounds(lb=np.asarray([0,0,0]),ub=np.asarray([1,1,1])), method='trust-constr', constraints=cons_list, callback = _stopping_callback)
                            res_obj_high.x = np.round(res_obj_high.x,1)
                            res_obj_high.x = [x/np.sum(res_obj_high.x) for x in res_obj_high.x]
                            res_obj_low = minimize(fun=_minimizing_problem, x0=np.asarray([0.25, 0.25, 0.5]), bounds=Bounds(lb=np.asarray([0,0,0]),ub=np.asarray([1,1,1])), method='trust-constr', constraints=cons_list, callback = _stopping_callback)
                            res_obj_low.x = np.round(res_obj_low.x,1)
                            res_obj_low.x = [x/np.sum(res_obj_low.x) for x in res_obj_low.x]
                            solns = [(ag,[(res_obj_high.x[0],res_obj_low.x[0]),(res_obj_high.x[1],res_obj_low.x[1]),(res_obj_high.x[2],res_obj_low.x[2]),(0,0)])]
                        else:
                            solns = [(ag,[(0,0),(0,0),(0,0),(1,1)])]
                        for s in solns:
                            ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                            ag_obj = self.agent_object_dict[file_str][s[0]]
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
                            ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','maxmin',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                            print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
        
    def solve(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        ins_list = []
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(self.l1_model_config.l1_model_type,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            
            ct +=1
            #if ct > 30:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ''' older model did not take into account rules, so this flag is for backward compatibility '''
            if self.l1_model_config.ignore_rule:
                rule_acts = []
            
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            weights_info = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if not self.l1_model_config.is_transformed:
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    weights_info[idx] = ag_weights
                    for k in payoff_dict.keys():
                        orig_payoff = payoff_dict[k]
                        if len(rule_acts) == 0:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                            ag_orig_payoff = orig_payoff[:,idx]
                        else:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                            if k[idx] in [x[idx] for x in rule_acts]:
                                ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                            else:
                                ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                        if not self.mixed_weights:
                            high_weights = np.where(high_weights == max(high_weights), 1, 0)
                            #if np.sum(np.exp(10*high_weights)) != 0:
                            #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                            low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        high_weighted_payoffs = high_weights @ ag_orig_payoff
                        high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                        low_weighted_payoffs = low_weights @ ag_orig_payoff
                        low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                else:
                    high_bounds_transformed_payoffs = payoff_dict
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                eq = equilibria_core.EquilibriaCore(num_players,high_bounds_transformed_payoffs,len(high_bounds_transformed_payoffs),player_actions[idx],False)
                soln = eq.calc_max_min_response()
                if not self.l1_model_config.ignore_low:
                    eq = equilibria_core.EquilibriaCore(num_players,low_bounds_transformed_payoffs,len(low_bounds_transformed_payoffs),player_actions[idx],False)
                    soln_low = eq.calc_max_min_response()
                    soln.update(soln_low)
                opt_strat = list(soln.keys())
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    if not self.l1_model_config.old_model:
                        emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                        emp_utils = np.asarray(emp_utils)
                        assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            emp_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                        else:
                            emp_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    if not self.l1_model_config.old_model:
                        assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    if not self.l1_model_config.old_model:
                        soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                        soln_utils = np.asarray(soln_utils)
                        assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            soln_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                        else:
                            soln_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                
                if not self.l1_model_config.old_model  or not self.l1_model_config.is_transformed:
                    emp_utils = np.asarray(emp_utils)
                    soln_utils = np.asarray(soln_utils)
                    weights_info[idx] = np.asarray(weights_info[idx])
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None and not self.l1_model_config.old_model else None)
                else:
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils) if emp_utils is not None else None,None,json.dumps(soln_utils) if soln_utils is not None else None,None)
                    
            
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                
            
                player_acts_solns[idx] = list(set([x[idx][9:11] for x in soln.keys()]))
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
        
        
        
class Ql1Model(InverseQlkRGameLearningModel):
    
    def solve(self):
        self.lzero_behavior = self.l1_model_config.lzero_behavior
        if self.lzero_behavior == 'maxmax':
            self.solve_maxmax()
        else:
            self.solve_maxmin()
            
    def solve_maxmin(self):
        ins_list = []
        lzero_behavior = self.lzero_behavior
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(lzero_behavior,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ''' older model did not take into account rules, so this flag is for backward compatibility '''
            if self.l1_model_config.ignore_rule:
                rule_acts = []
            
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            weights_info = [[] for i in range(num_players)]
            
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if not self.l1_model_config.is_transformed:
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    weights_info[idx] = ag_weights
                    for k in payoff_dict.keys():
                        orig_payoff = payoff_dict[k]
                        if len(rule_acts) == 0:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                            ag_orig_payoff = orig_payoff[:,idx]
                        else:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                            if k[idx] in [x[idx] for x in rule_acts]:
                                ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                            else:
                                ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                        if not self.mixed_weights:
                            high_weights = np.where(high_weights == max(high_weights), 1, 0)
                            #if np.sum(np.exp(10*high_weights)) != 0:
                            #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        else:
                            if np.sum(high_weights) != 0:
                                high_weights = high_weights/np.sum(high_weights)
                            if np.sum(low_weights) != 0:
                                low_weights = low_weights/np.sum(low_weights)   
                        high_weighted_payoffs = high_weights @ ag_orig_payoff
                        high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                        low_weighted_payoffs = low_weights @ ag_orig_payoff
                        low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                else:
                    high_bounds_transformed_payoffs = payoff_dict
            lzero_soln_strats = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                eq = equilibria_core.EquilibriaCore(num_players,high_bounds_transformed_payoffs,len(high_bounds_transformed_payoffs),player_actions[idx],False)
                soln = eq.calc_max_min_response()
                if not self.l1_model_config.ignore_low:
                    eq = equilibria_core.EquilibriaCore(num_players,low_bounds_transformed_payoffs,len(low_bounds_transformed_payoffs),player_actions[idx],False)
                    soln_low = eq.calc_max_min_response()
                    soln.update(soln_low)
                opt_strat = list(soln.keys())
                lzero_soln_strats[idx].extend([x[idx] for x in opt_strat])
            lzero_soln_strats = [list(set(x)) for x in lzero_soln_strats]
            lzero_soln_strats = list(itertools.product(*lzero_soln_strats))
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                opt_strat = []
                for s in lzero_soln_strats:
                    strat_subsets = []
                    for pl_strat in player_actions[idx]:
                        _s = list(s)
                        _s[idx] = pl_strat
                        _s = tuple(_s)
                        if _s not in strat_subsets:
                            strat_subsets.append(_s)
                    this_player_payoff_dict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items() if k in strat_subsets}
                    l_one_soln_strat,l_one_opt_val = self.keywithmaxval(this_player_payoff_dict)
                    for l_str in l_one_soln_strat:
                        if l_str not in opt_strat:
                            opt_strat.append(l_str)
                    if not self.l1_model_config.ignore_low:
                        this_player_payoff_dict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items() if k in strat_subsets}
                        l_one_soln_strat,l_one_opt_val = self.keywithmaxval(this_player_payoff_dict)
                        for l_str in l_one_soln_strat:
                            if l_str not in opt_strat:
                                opt_strat.append(l_str)
                
                ''' build the confusion dict '''
                player_acts_solns[idx] = list(set([x[idx][9:11] for x in opt_strat]))
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    if not self.l1_model_config.old_model:
                        emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                        emp_utils = np.asarray(emp_utils)
                        assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            emp_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                        else:
                            emp_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    if not self.l1_model_config.old_model:
                        assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    if not self.l1_model_config.old_model:
                        soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                        soln_utils = np.asarray(soln_utils)
                        assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            soln_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                        else:
                            soln_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                
                
                if not self.l1_model_config.old_model or not self.l1_model_config.is_transformed:
                    emp_utils = np.asarray(emp_utils)
                    soln_utils = np.asarray(soln_utils)
                    weights_info[idx] = np.asarray(weights_info[idx])
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None and not self.l1_model_config.old_model else None)
                else:
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils) if emp_utils is not None else None,None,json.dumps(soln_utils) if soln_utils is not None else None,None)
                
                
                
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()
        
    def solve_maxmax(self):
        ins_list = []
        lzero_behavior = self.lzero_behavior
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        ins_string = 'INSERT INTO L1_SOLUTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        self.confusion_dict, self.confusion_dict_N = dict(), dict()
        weights_dict = utils.load_weights_map(lzero_behavior,self.l1_model_config.model_type)
        ag_info_map = utils.load_agent_info_map(self.file_id)
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            ''' older model did not take into account rules, so this flag is for backward compatibility '''
            if self.l1_model_config.ignore_rule:
                rule_acts = []
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            player_acts_solns = [[] for i in range(num_players)]
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
            weights_info = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                ag_info_key = (int(self.file_id),ag_id,time_ts)
                if not self.l1_model_config.is_transformed:
                    if (ag_info_key in ag_info_map and ag_info_map[ag_info_key] in weights_dict) and not self.l1_model_config.with_baseline_weights:
                        ag_weights = weights_dict[ag_info_map[ag_info_key]]
                    else:
                        ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    weights_info[idx] = ag_weights
                    for k in payoff_dict.keys():
                        orig_payoff = payoff_dict[k]
                        if len(rule_acts) == 0:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                            ag_orig_payoff = orig_payoff[:,idx]
                        else:
                            high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                            low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                            if k[idx] in [x[idx] for x in rule_acts]:
                                ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                            else:
                                ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                        if not self.mixed_weights:
                            high_weights = np.where(high_weights == max(high_weights), 1, 0)
                            #if np.sum(np.exp(10*high_weights)) != 0:
                            #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        else:
                            if np.sum(high_weights) != 0:
                                high_weights = high_weights/np.sum(high_weights)
                            if np.sum(low_weights) != 0:
                                low_weights = low_weights/np.sum(low_weights)   
                        high_weighted_payoffs = high_weights @ ag_orig_payoff
                        high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
                        low_weighted_payoffs = low_weights @ ag_orig_payoff
                        low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                else:
                    high_bounds_transformed_payoffs = payoff_dict
            lzero_soln_strats = [[] for i in range(num_players)]
            for idx,ag in enumerate(all_agents):
                this_player_payoff_dict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items()}
                opt_strat,opt_val = self.keywithmaxval(this_player_payoff_dict)
                lzero_soln_strats[idx].extend([x[idx] for x in opt_strat])
                if not self.l1_model_config.ignore_low:
                    this_player_payoff_dict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items()}
                    opt_strat,opt_val = self.keywithmaxval(this_player_payoff_dict)
                    lzero_soln_strats[idx].extend([x[idx] for x in opt_strat])
            lzero_soln_strats = [list(set(x)) for x in lzero_soln_strats]
            lzero_soln_strats = list(itertools.product(*lzero_soln_strats))
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                opt_strat = []
                for s in lzero_soln_strats:
                    strat_subsets = []
                    for pl_strat in player_actions[idx]:
                        _s = list(s)
                        _s[idx] = pl_strat
                        _s = tuple(_s)
                        if _s not in strat_subsets:
                            strat_subsets.append(_s)
                    this_player_payoff_dict = {k:v[idx] for k,v in high_bounds_transformed_payoffs.items() if k in strat_subsets}
                    l_one_soln_strat,l_one_opt_val = self.keywithmaxval(this_player_payoff_dict)
                    for l_str in l_one_soln_strat:
                        if l_str not in opt_strat:
                            opt_strat.append(l_str)
                    if not self.l1_model_config.ignore_low:
                        this_player_payoff_dict = {k:v[idx] for k,v in low_bounds_transformed_payoffs.items() if k in strat_subsets}
                        l_one_soln_strat,l_one_opt_val = self.keywithmaxval(this_player_payoff_dict)
                        for l_str in l_one_soln_strat:
                            if l_str not in opt_strat:
                                opt_strat.append(l_str)
                
                ''' build the confusion dict '''
                player_acts_solns[idx] = list(set([x[idx][9:11] for x in opt_strat]))
                this_player_emp_acts = list(set([x[idx][9:11] for x in emp_acts]))
                model_parms_str = self.l1_model_config.get_model_parms_str()
                emp_utils, rule_utils, soln_utils = None, None, None
                if len(emp_acts) > 0:
                    if not self.l1_model_config.old_model:
                        emp_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in emp_acts]
                        emp_utils = np.asarray(emp_utils)
                        assert emp_utils.shape == (len(emp_acts),3,num_players) , str(emp_utils.shape) + str((len(emp_acts),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            emp_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                        else:
                            emp_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in emp_acts]
                if len(rule_acts) > 0:
                    rule_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in rule_acts]
                    rule_utils = np.asarray(rule_utils)
                    if not self.l1_model_config.old_model:
                        assert rule_utils.shape == (len(rule_acts),3,num_players) , str(rule_utils.shape) + str((len(rule_acts),3,num_players))
                if len(opt_strat) > 0:
                    if not self.l1_model_config.old_model:
                        soln_utils = [payoff_dict[x] if x in payoff_dict else np.full((3, num_players), np.nan, dtype=float) for x in opt_strat]
                        soln_utils = np.asarray(soln_utils)
                        assert soln_utils.shape == (len(opt_strat),3,num_players) , str(soln_utils.shape) + str((len(opt_strat),3,num_players))
                    else:
                        if self.l1_model_config.is_transformed:
                            soln_utils = [payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                        else:
                            soln_utils = [np.asarray([0.25,0.25,0.5])@payoff_dict[x] if x in payoff_dict else [None]*num_players for x in opt_strat]
                
                
                if not self.l1_model_config.old_model or not self.l1_model_config.is_transformed:
                    emp_utils = np.asarray(emp_utils)
                    soln_utils = np.asarray(soln_utils)
                    weights_info[idx] = np.asarray(weights_info[idx])
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils.tolist()) if emp_utils is not None else None,json.dumps(rule_utils.tolist()) if rule_utils is not None else None,json.dumps(soln_utils.tolist()) if soln_utils is not None else None,json.dumps(weights_info[idx].tolist()) if weights_info[idx] is not None and not self.l1_model_config.old_model else None)
                else:
                    soln_vect = (int(self.file_id),ag_id,time_ts,self.l1_model_config.l1_model_type,model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,json.dumps(emp_utils) if emp_utils is not None else None,None,json.dumps(soln_utils) if soln_utils is not None else None,None)
                
                
                
                #soln_vect = (int(self.file_id),ag_id,time_ts,'Ql1',model_parms_str,str(emp_acts) if emp_acts is not None else None,str(rule_acts) if rule_acts is not None else None,str(opt_strat) if opt_strat is not None else None,emp_utils,rule_utils,soln_utils,weights_info[idx])
                ins_list.append(soln_vect)
                for e_a in this_player_emp_acts:
                    if e_a not in self.confusion_dict:
                        self.confusion_dict[e_a] = []
                    if e_a not in self.confusion_dict_N:
                        self.confusion_dict_N[e_a] = 0
                    if e_a in player_acts_solns[idx]:
                        self.confusion_dict[e_a].append(e_a)
                        self.confusion_dict_N[e_a] += 1
                    else:
                        for s_a in player_acts_solns[idx]:
                            self.confusion_dict[e_a].append(s_a)
                        ''' since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization'''
                        self.confusion_dict[e_a] += [e_a]*(len(player_acts_solns[idx])-1)
                        self.confusion_dict_N[e_a] += 1
        if not self.l1_model_config.no_insertion:
            c.executemany(ins_string,ins_list)
            conn.commit()
            conn.close()

class InverseCorrelatedEquilibriaLearningModel(InverseNashEqLearningModel):
    
    def calc_confusion_matrix(self):
        self.confusion_dict = {utils.get_l1_action_string(int(k)):[utils.get_l1_action_string(int(x)) for x in v] for k,v in self.confusion_dict.items()}
        self.confusion_dict_N = {utils.get_l1_action_string(int(k)):v for k,v in self.confusion_dict_N.items()}
        confusion_key_list = list(self.confusion_dict.keys())
        confusion_matrix = []
        for k in confusion_key_list:
            confusion_arr = []
            ctr = Counter(self.confusion_dict[k])
            _sum = sum(ctr.values())
            for p_k in confusion_key_list:
                th_prob = ctr[p_k]/_sum
                confusion_arr.append(th_prob)
            confusion_matrix.append(confusion_arr)
        df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(self.confusion_dict_N[x])+')' for x in confusion_key_list],
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
        plt.savefig('corr_'+str(self.with_baseline_weights)+'_'+str(self.mixed_weights)+'.png', bbox_inches='tight')
        plt.show()
        f=1
    
    def solve(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+self.file_id+'\\uni_weber_'+self.file_id+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM UTILITY_WEIGHTS where MODEL_TYPE='CorrEq'"
        c.execute(q_string)
        res = c.fetchall()
        if len(res) == 0:
            sys.exit('UTILITY_WEIGHTS weights table is not populated ')
        weights_dict = dict()
        for row in res:
            if row[12] is not None:
                if tuple(row[3:11]) not in weights_dict:
                    weights_dict[tuple(row[3:11])] = [row[13:]]
                else:
                    weights_dict[tuple(row[3:11])].append(row[13:])
        weights_dict = {k: np.mean(np.asarray(v), axis=0) for k,v in weights_dict.items()}
        q_string = "SELECT * FROM UTILITY_WEIGHTS where MODEL_TYPE='CorrEq'"
        c.execute(q_string)
        res = c.fetchall()
        agent_info = {tuple(row[:3]):tuple(row[3:11]) for row in res}
        self.confusion_dict,self.confusion_dict_N = dict(),dict()
        ct,N = 0,len(self.file_keys)
        for file_str in self.file_keys:
            ct +=1
            #if ct > 20:
            #    continue
            print('processing',self.file_id,ct,'/',N)
            time_ts = file_str.split('_')[1].replace(',','.')
            time_ts = round(float(time_ts),len(time_ts.split('.')[1])) if '.' in time_ts else round(float(time_ts),0)
            payoff_dict = utils.pickle_load(os.path.join(self.l3_cache_str,file_str))
            all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
            if len(all_agents) < 2:
                print('less than 2 continuing')
                continue
            emp_acts,rule_acts = self.emp_acts_dict[file_str], self.rule_acts_dict[file_str]
            num_players = len(all_agents)
            player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]
            all_indices = list(itertools.product(*[list(np.arange(len(x))) for x in player_actions]))
            all_vars = ['p'+''.join([str(y) for y in list(x)]) for x in all_indices]
            player_acts_solns = [[] for i in range(num_players)]
            high_constr_list,low_constr_list = None, None
            low_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict}
            high_bounds_transformed_payoffs = {k:[None]*len(all_agents) for k in payoff_dict} 
                
            for idx,ag in enumerate(all_agents):
                ag_id = ag[0] if ag[1]==0 else ag[1]
                agent_info_key = (int(self.file_id),ag_id,time_ts)
                ag_state_key = agent_info[agent_info_key] if agent_info_key in agent_info else None
                if (ag_state_key is not None and ag_state_key in weights_dict) and not self.with_baseline_weights:
                    ag_weights = weights_dict[ag_state_key]
                else:
                    ag_weights = np.asarray([0.25, 0.25,  0.5, 0, 0.25, 0.25, 0.5, 0])
                    print('default weights assigned')
                for k in payoff_dict.keys():
                    orig_payoff = payoff_dict[k]
                    if len(rule_acts) == 0:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2], 0), newshape=(1,3))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6], 0), newshape=(1,3))
                        ag_orig_payoff = orig_payoff[:,idx]
                    else:
                        high_weights = np.reshape(np.take(ag_weights, [0,1,2,3], 0), newshape=(1,4))
                        low_weights = np.reshape(np.take(ag_weights, [4,5,6,7], 0), newshape=(1,4))
                        if k[idx] in [x[idx] for x in rule_acts]:
                            ag_orig_payoff = np.append(orig_payoff[:,idx],1)
                        else:
                            ag_orig_payoff = np.append(orig_payoff[:,idx], 0)
                    if self.mixed_weights:
                        if np.sum(high_weights) != 0:
                            high_weights = high_weights/np.sum(high_weights)
                        if np.sum(low_weights) != 0:
                            low_weights = low_weights/np.sum(low_weights)
                        
                    else:
                        high_weights = np.where(high_weights == max(high_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    high_weights = np.exp(10*high_weights)/np.sum(np.exp(10*high_weights))
                        low_weights = np.where(low_weights == max(low_weights), 1, 0)
                        #if np.sum(np.exp(10*high_weights)) != 0:
                        #    low_weights = np.exp(10*low_weights)/np.sum(np.exp(10*low_weights))
                    high_weighted_payoffs = high_weights @ ag_orig_payoff
                    low_weighted_payoffs = low_weights @ ag_orig_payoff
                    low_bounds_transformed_payoffs[k][idx] = low_weighted_payoffs[0]
                    high_bounds_transformed_payoffs[k][idx] = high_weighted_payoffs[0]
            
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
                    high_obj_utils = [high_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                    low_obj_utils = [high_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                    var_vect = ['p'+''.join([str(player_actions[a_idx].index(y)) for a_idx,y in enumerate(list(x))]) for x in other_agent_act_combinations]
                    
                    for this_ag_oth_strat in player_actions[idx]:
                        if this_ag_strat != this_ag_oth_strat:
                            other_agent_act_combinations = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                            for idx_2,s in enumerate(other_agent_act_combinations):
                                _s = list(s)
                                _s[idx] = this_ag_oth_strat
                                other_agent_act_combinations[idx_2] = tuple(_s)
                            constr_utils = [high_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                            constr_diff = [x1 - x2 for (x1, x2) in zip(high_obj_utils, constr_utils)]
                            if np.any(constr_diff):
                                if high_constr_list is None:
                                    high_constr_list = [(var_vect,constr_diff)] 
                                else:
                                    high_constr_list.append((var_vect,constr_diff))
                            constr_utils = [low_bounds_transformed_payoffs[x][idx] for x in other_agent_act_combinations]
                            constr_diff = [x1 - x2 for (x1, x2) in zip(low_obj_utils, constr_utils)]
                            if np.any(constr_diff):
                                if low_constr_list is None:
                                    low_constr_list = [(var_vect,constr_diff)] 
                                else:
                                    low_constr_list.append((var_vect,constr_diff))
            solns_dict = dict()
            obj_vals = []
            for act_code in all_vars:
                val = high_bounds_transformed_payoffs[tuple([player_actions[i][int(x)] for i,x in enumerate(act_code[1:])])]
                obj_vals.append(val)
            if high_constr_list is not None:
                solns = solve_lp_multivar(all_vars,obj_vals,high_constr_list) 
                solns_dict['high'] = solns
            obj_vals = []
            for act_code in all_vars:
                val = low_bounds_transformed_payoffs[tuple([player_actions[i][int(x)] for i,x in enumerate(act_code[1:])])]
                obj_vals.append(val)
            if low_constr_list is not None:
                solns = solve_lp_multivar(all_vars,obj_vals,low_constr_list) 
                solns_dict['low'] = solns
            
            for hl_i, solns in solns_dict.items():
                if solns is not None:
                    print('max_prob',max(solns))
                    for i,s in enumerate(solns):
                        if s >= 0.01:
                            print(s)
                            strat = [player_actions[p_i][int(x)] for p_i,x in enumerate(list(all_vars[i][1:]))]
                            int_prob = int(round(s,2)*100)
                            for idx,ag in enumerate(all_agents):
                                this_player_emp_acts = list(set([x[idx] for x in emp_acts]))
                                for e_a in this_player_emp_acts:
                                    if e_a[9:11] not in self.confusion_dict:
                                        self.confusion_dict[e_a[9:11]] = []
                                    if e_a[9:11] not in self.confusion_dict_N:
                                        self.confusion_dict_N[e_a[9:11]] = 0
                                    self.confusion_dict[e_a[9:11]].extend([strat[idx][9:11]]*int_prob)
                                    self.confusion_dict_N[e_a[9:11]] += 1 
                else:
                    print('no solutions')
                
    
    def formulate_and_solve_lp_est_probs(self, all_agents,obj_mat,constr_mat,on_rule,this_idx,file_str):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if idx == this_idx:
                num_params = 2 if self.agent_object_dict[file_str][ag].relev_pedestrians is None else 3
                solns = solve_lp(obj_mat, constr_mat, num_params)
                all_solns.append((ag,solns))
        return all_solns
        
    
    def formulate_and_solve_lp_est_weights(self,all_agents,obj_mat,constr_mat,on_rule,this_idx,file_str):
        all_solns = []
        for idx,ag in enumerate(all_agents):
            if idx == this_idx:
                if on_rule:
                    all_solns.append((ag,[(0,0),(0,0),(0,0),(1,1)]))
                else:
                    num_params = 2 if self.agent_object_dict[file_str][ag].relev_pedestrians is None else 3
                    solns = solve_lp(obj_mat, constr_mat, num_params)
                    all_solns.append((ag,solns))
        return all_solns
    
    
    def build_emp_distribution(self):
        ct,N = 0,len(self.file_keys)
        emp_act_distr = dict()
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N)
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
        q_string = "INSERT INTO UTILITY_WEIGHTS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for file_str in self.file_keys:
            ct +=1
            print('processing',self.file_id,ct,'/',N)
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
                    solns = self.formulate_and_solve_lp_est_weights(all_agents,obj_utils,constr_list,on_rule,idx,file_str)
                    for s in solns:
                        ag_id = s[0][0] if s[0][1] == 0 else s[0][1]
                        ag_obj = self.agent_object_dict[file_str][s[0]]
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
                        ins_list.append((int(self.file_id), ag_id, time_ts, ag_obj.signal, next_signal_change[1], time_to_change, ag_obj.current_segment, speed_fact, 'Y' if leading_veh_id is not None else 'N', relev_agents, 'Y' if ag_obj.relev_pedestrians is not None else 'N','CorrEq',self.l1_model_config.model_type, s[1][0][0], s[1][1][0], s[1][2][0], s[1][3][0], s[1][0][1], s[1][1][1], s[1][2][1], s[1][3][1]))
                        print(file_str,ct,'/',N)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()


def insert_weights_all_models(traj_type):
    
    #file_id = sys.argv[1]
    #traj_type = sys.argv[2]
    #soln = sys.argv[3]
    for file_id in ['769','770','771','775']:
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        q_string = "DELETE from UTILITY_WEIGHTS WHERE UTILITY_WEIGHTS.L3_MODEL_TYPE='"+traj_type+"'"
        c.execute(q_string)
        conn.commit()
        conn.close()          
        
        soln = 'NA'
        
        constants.CURRENT_FILE_ID = file_id
        l1_model_config = L1_Model_Config()
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('NASH')
        l1_model_config.set_update_only(False)
        l1_model = InverseNashEqLearningModel(l1_model_config)
        l1_model.calculate_and_insert_weights()
        #l1_model.solve()
        #l1_model.calc_confusion_matrix()
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('QlkR')
        l1_model_config.set_update_only(False)
        l1_model = InverseQlkRGameLearningModel(l1_model_config)        
        l1_model.calculate_and_insert_weights()
        #l1_model.solve()
        #l1_model.calc_confusion_matrix()
        
        
        '''
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('CorrEq')
        l1_model_config.set_update_only(False)
        l1_model = InverseCorrelatedEquilibriaLearningModel(l1_model_config)
        l1_model.build_emp_distribution()
        l1_model.calculate_and_insert_weights()
        #l1_model.solve()
        #l1_model.calc_confusion_matrix()
        '''
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('brtb')
        l1_model_config.set_update_only(False)
        l1_model = InverseBestResponseWithTrueBelief(l1_model_config)  
        l1_model.calculate_and_insert_weights() 
        #l1_model.solve()
        #l1_model.calc_confusion_matrix('brtb_')
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmin')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMinResponse(l1_model_config)   
        l1_model.calculate_and_insert_weights()
        #l1_model.solve()
        #l1_model.calc_confusion_matrix('maxmin')
        
        
        #l1_model_config = L1_Model_Config()
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('calculate_and_insert_weights')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmax')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMaxResponse(l1_model_config)   
        l1_model.calculate_and_insert_weights()
        #l1_model.solve()
        #l1_model.calc_confusion_matrix('maxmin')
    
    
def solve_l1_all_models(traj_type,no_insertion):
    
    #file_id = sys.argv[1]
    #traj_type = sys.argv[2]
    #soln = sys.argv[3]
    for file_id in [x for x in constants.ALL_FILE_IDS if int(x) > 775]:
        soln = 'NA'
        constants.CURRENT_FILE_ID = file_id
        
        if not no_insertion:
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            q_string = "DELETE from L1_SOLUTIONS WHERE L1_SOLUTIONS.MODEL_PARMS LIKE '%l3_sampling="+traj_type+",%'"
            c.execute(q_string)
            conn.commit()
            conn.close()
        
        
        l1_model_config = L1_Model_Config()
        l1_model_config.set_no_insertion(no_insertion)
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('NASH')
        l1_model_config.set_update_only(False)
        l1_model = InverseNashEqLearningModel(l1_model_config)
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        l1_model.calc_confusion_matrix()
        
        
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('QlkR')
        l1_model_config.set_update_only(False)
        l1_model = InverseQlkRGameLearningModel(l1_model_config)        
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        l1_model.calc_confusion_matrix('QlkR_')
        
        
        '''
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('CorrEq')
        l1_model_config.set_update_only(False)
        l1_model = InverseCorrelatedEquilibriaLearningModel(l1_model_config)
        l1_model.build_emp_distribution()
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        l1_model.calc_confusion_matrix()
        '''
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('brtb')
        l1_model_config.set_update_only(False)
        l1_model = InverseBestResponseWithTrueBelief(l1_model_config)  
        #l1_model.calculate_and_insert_weights() 
        l1_model.solve()
        l1_model.calc_confusion_matrix('brtb_')
        
        
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmin')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMinResponse(l1_model_config)   
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        l1_model.calc_confusion_matrix('Ql0maxmin')
    
    
    
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmax')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMaxResponse(l1_model_config)   
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        l1_model.calc_confusion_matrix('Ql0maxmax')
        
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_lzero_behavior('maxmin')
        l1_model_config.set_l1_model_type('Ql1')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model = Ql1Model(l1_model_config)
        l1_model.solve()
        l1_model.calc_confusion_matrix('Ql1maxmin')
        
        l1_model_config.set_baseline_weights_flag(False)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_lzero_behavior('maxmax')
        l1_model_config.set_l1_model_type('Ql1')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model = Ql1Model(l1_model_config)
        l1_model.solve()
        l1_model.calc_confusion_matrix('Ql1maxmax')
    
    
    

def insert_weights_maxmax():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('771')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('maxmax')
    l1_model_config.set_update_only(True)
    l1_model = InverseMaxMaxResponse(l1_model_config)   
    l1_model.calculate_and_insert_weights()
    #l1_model.solve()
    #l1_model.calc_confusion_matrix('maxmax')

def insert_weights_maxmin():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('771')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('maxmin')
    l1_model_config.set_update_only(True)
    l1_model = InverseMaxMinResponse(l1_model_config)   
    l1_model.calculate_and_insert_weights()
    #l1_model.solve()
    #l1_model.calc_confusion_matrix('maxmin')
    
def solve_Ql1():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('769')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_lzero_behavior('maxmin')
    l1_model_config.set_l1_model_type('Ql1')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model = Ql1Model(l1_model_config)
    l1_model.solve()
    #l1_model.calc_confusion_matrix('ql1maxma_l3baseline')

def solve_maxmax():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('769')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('maxmax')
    l1_model = InverseMaxMaxResponse(l1_model_config)   
    #l1_model.calculate_and_insert_weights()
    l1_model.solve()
    
def solve_maxmin():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('769')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('maxmin')
    l1_model = InverseMaxMinResponse(l1_model_config)   
    #l1_model.calculate_and_insert_weights()
    l1_model.solve()

def solve_nash():
    file_id, traj_type, soln, traj_type = '769','SAMPLING_EQ', 'NA','SAMPLING_EQ'
    constants.CURRENT_FILE_ID = file_id
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id(file_id)
    l1_model_config.set_run_type('solve')
    l1_model_config.set_l3_model_def(traj_type,soln)
    l1_model_config.set_l3_weights_estimation_model_type(traj_type)
    l1_model_config.set_l1_model_type('NASH')
    l1_model_config.set_update_only(False)
    l1_model = InverseNashEqLearningModel(l1_model_config)
    #l1_model.calculate_and_insert_weights()
    #l1_model.solve()
    l1_model.calc_confusion_matrix()
    
def solve_qlkr():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('769')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('QlkR')
    l1_model = InverseQlkRGameLearningModel(l1_model_config)   
    #l1_model.calculate_and_insert_weights()
    l1_model.solve()
    
def solve_brtb():
    l1_model_config = L1_Model_Config()
    l1_model_config.set_baseline_weights_flag(False)
    l1_model_config.set_mixed_weights_flag(True)
    l1_model_config.set_file_id('769')
    l1_model_config.set_l3_model_def('SAMPLING_EQ', 'NA')
    l1_model_config.set_l3_weights_estimation_model_type('SAMPLING_EQ')
    l1_model_config.set_l1_model_type('brtb')
    l1_model = InverseBestResponseWithTrueBelief(l1_model_config)   
    #l1_model.calculate_and_insert_weights()
    l1_model.solve()


def solve_l1_all_models_old_params():
    
    file_id = sys.argv[1]
    #traj_type = sys.argv[2]
    #soln = sys.argv[3]
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
    c = conn.cursor()
    q_string = "delete FROM L1_SOLUTIONS WHERE MODEL='maxmin' and L1_SOLUTIONS.MODEL_PARMS LIKE '%old_model%';"
    #c.execute(q_string)
    #conn.commit()
    conn.close()           
    
    l1_model_config = L1_Model_Config()
    models = [('BASELINE','NA'),('BOUNDARY','BR'),('BOUNDARY','MAXMIN'),('GAUSSIAN','BR'),('GAUSSIAN','MAXMIN')]
    for traj_type,soln in models[0:1]:
        
        if traj_type == 'BOUNDARY' or traj_type == 'GAUSSIAN':
            l1_model_config.set_prev_version_params()
        else:
            l1_model_config.set_prev_version_params()
            l1_model_config.is_transformed = False
        
        
        l1_model_config.set_baseline_weights_flag(True)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('NASH')
        l1_model_config.set_update_only(False)
        l1_model = InverseNashEqLearningModel(l1_model_config)
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        #l1_model.calc_confusion_matrix()
        '''
        l1_model_config.set_baseline_weights_flag(True)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmin')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMinResponse(l1_model_config)   
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        #l1_model.calc_confusion_matrix('maxmin')
        '''
        
        '''
        l1_model_config.set_baseline_weights_flag(True)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_run_type('solve')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model_config.set_l1_model_type('maxmax')
        l1_model_config.set_update_only(False)
        l1_model = InverseMaxMaxResponse(l1_model_config)   
        #l1_model.calculate_and_insert_weights()
        l1_model.solve()
        #l1_model.calc_confusion_matrix('maxmin')
        
        l1_model_config.set_baseline_weights_flag(True)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_lzero_behavior('maxmin')
        l1_model_config.set_l1_model_type('Ql1')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model = Ql1Model(l1_model_config)
        l1_model.solve()
        
        l1_model_config.set_baseline_weights_flag(True)
        l1_model_config.set_mixed_weights_flag(True)
        l1_model_config.set_file_id(file_id)
        l1_model_config.set_l3_model_def(traj_type,soln)
        l1_model_config.set_lzero_behavior('maxmax')
        l1_model_config.set_l1_model_type('Ql1')
        l1_model_config.set_l3_weights_estimation_model_type(traj_type)
        l1_model = Ql1Model(l1_model_config)
        l1_model.solve()
        '''
if __name__ == '__main__':   
    for traj_type in ['BASELINE','BASELINEUTILS','SAMPLING_EQ']: 
        insert_weights_all_models(traj_type)
    for traj_type in ['BASELINE','BASELINEUTILS','SAMPLING_EQ']: 
        solve_l1_all_models(traj_type,False)


