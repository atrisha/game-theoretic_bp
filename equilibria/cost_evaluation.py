'''
Created on Feb 6, 2020

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3
from all_utils import utils
import sys
import constants
import ast
import math
import pickle
import os.path
from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import scipy.special,scipy.stats
import visualizer
from equilibria import equilibria_core
from motion_planners import motion_planner
from scipy.interpolate import CubicSpline
from motion_planners.planning_objects import TrajectoryDef
from collections import OrderedDict
from all_utils import db_utils
import concurrent.futures
import threading

import logging
logging.basicConfig(format='%(levelname)-8s %(funcName)s-2s  %(module)s: %(message)s',level=logging.INFO)








class CostEvaluation():
    
    def __init__(self,eq_context):
        self.eq_context = eq_context
        
    
    def calc_l3_equilibrium_payoffs(self,inhibitory,excitatory,traj_id_list,strategy_tuple,traffic_signal):
        num_agents = len(traj_id_list)
        all_traj_viability = eval_trajectory_viability(traj_id_list)
        len_of_trajectories = [len(x) for x in all_traj_viability]
        #print('N=',len_of_trajectories)
        ''' complexity weighted '''
        #chosen_trajectory_id_combinations = list(itertools.product(*[np.random.choice(np.arange(x),size = min(5,x), replace=False, p=all_traj_viability[_i]) for _i,x in enumerate(len_of_trajectories)]))
        chosen_trajectory_ids = [np.random.choice([t[0] for t in x],size = min(5,len(x)), replace=False, p=[t[1] for t in x]) for x in all_traj_viability]
        chosen_trajectory_id_combinations = list(itertools.product(*chosen_trajectory_ids))
        ''' random '''
        #chosen_trajectory_id_combinations = list(itertools.product(*[np.random.choice(np.arange(x),size = max(1,x//5,x//4), replace=False) for _i,x in enumerate(len_of_trajectories)]))
        '''all - not possible (just for reference)'''
        #chosen_trajectory_id_combinations = list(itertools.product(*[np.arange(x) for _i,x in enumerate(len_of_trajectories)]))
        #print('N*=',len(chosen_trajectory_id_combinations))
        all_possible_payoffs = self.calc_l3_payoffs(chosen_trajectory_id_combinations,chosen_trajectory_ids,num_agents,strategy_tuple)
        
        all_possible_payoffs_vals = np.asarray([v for v in all_possible_payoffs.values()])
        all_possible_payoffs_keys = np.asarray([v for v in all_possible_payoffs.keys()])
        ''' calculate max,min,mean,sd respectively'''
        payoff_stats = np.full(shape=(4,num_agents),fill_value=np.inf)
        ''' we will only use the min and the max index for now '''
        payoff_stats_trajectories = np.full(shape=(4,num_agents),fill_value=np.inf)
        
        payoff_stats[0,] = np.amax(all_possible_payoffs_vals,axis = 0)
        for i in np.arange(num_agents):
            _idx = np.where(all_possible_payoffs_vals[:,i]==payoff_stats[0,i])
            payoff_stats_trajectories[0,i] = all_possible_payoffs_keys[_idx[0][0]][i]
        
        payoff_stats[1,] = np.amin(all_possible_payoffs_vals,axis = 0)
        for i in np.arange(num_agents):
            _idx = np.where(all_possible_payoffs_vals[:,i]==payoff_stats[1,i])
            payoff_stats_trajectories[1,i] = all_possible_payoffs_keys[_idx[0][0]][i]
        
        payoff_stats[2,] = np.mean(all_possible_payoffs_vals,axis = 0)
        for i in np.arange(num_agents):
            payoff_stats_trajectories[2,i] = all_possible_payoffs_keys[utils.find_nearest_in_array(all_possible_payoffs_vals[:,i],payoff_stats[2,i])][i]
        
        payoff_stats[3,] = np.std(all_possible_payoffs_vals,axis = 0)
        payoff_stats_trajectories = payoff_stats_trajectories.astype(int)
        ''' equilibria_core for l3 actions'''
        #visualizer.plot_payoff_grid(all_possible_payoffs_vals)
        #print('calculating l3 equilibria_core')
        eq = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(all_possible_payoffs)
        br = equilibria_core.calc_best_response(all_possible_payoffs)
        ns_dicts = {'max':{tuple(payoff_stats_trajectories[0,:]) : payoff_stats[0,:] },
                    'min':{tuple(payoff_stats_trajectories[1,:]) : payoff_stats[1,:] },
                    'mean':{tuple(payoff_stats_trajectories[2,:]) : payoff_stats[2,:] },
                    'sd':{tuple(payoff_stats_trajectories[3,:]) : payoff_stats[3,:] }
                    }
        
        return ns_dicts,eq,br     

        
    ''' given a strategy combination, evaluate the vector of payoffs for each agent.'''
    def eval_inhibitory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        disp_arr_x,disp_arr_y = [],[] 
        num_agents = len(strategy_tuple)
        all_possible_payoffs_inh = dict(all_possible_payoffs)
        for traj_idx_tuple in all_possible_payoffs.keys():
            ''' pair-wise min distance matrix '''
            dist_among_agents = np.full(shape=(num_agents,num_agents),fill_value=np.inf)
            dist = np.full(shape=(num_agents),fill_value=np.inf)
            for i in np.arange(num_agents):
                for j in np.arange(i,num_agents):
                    if i != j:
                        s_x,r_x = traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[j][traj_idx_tuple[j]][:,2]
                        ''' for now calculate the plan payoffs instead of payoffs for the next 1 second '''
                        #slice_len = min(s_x.shape[0],r_x.shape[0])
                        slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_x.shape[0],r_x.shape[0]))
                        s_x,r_x = s_x[:slice_len],r_x[:slice_len]
                        s_y,r_y = traj_dict_list[i][traj_idx_tuple[i]][:,3],traj_dict_list[j][traj_idx_tuple[j]][:,3]
                        #slice_len = min(s_y.shape[0],r_y.shape[0])
                        slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_y.shape[0],r_y.shape[0]))
                        s_y,r_y = s_y[:slice_len],r_y[:slice_len]
                        ''' original trajectory is at 10hz. sample at 1hz for speedup'''
                        if len(s_x) > 3:
                            s_x,r_x,s_y,r_y = s_x[0::9],r_x[0::9],s_y[0::9],r_y[0::9]
                        _d = np.hypot(s_x-r_x,s_y-r_y)
                        dist_among_agents[i,j] = min(_d)
            ''' to be safe, make the matrix symmetric '''
            dist_among_agents = np.minimum(dist_among_agents,dist_among_agents.T)
            ''' find the minimum distance for a vehicle action given all other agent actions '''
            dist = np.amin(dist_among_agents,axis=1)
            #payoffs = dist_payoffs(dist) + constants.L2_ACTION_PAYOFF_ADDITIVE
            payoffs = dist_payoffs(dist)
            all_possible_payoffs_inh[traj_idx_tuple] = payoffs
        return all_possible_payoffs_inh
        
    def eval_excitatory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        num_agents = len(strategy_tuple)
        payoff_stats_trajectories = np.full(shape=(4,num_agents),fill_value=np.inf)
        all_possible_payoffs_exc = dict(all_possible_payoffs)
        payoff_stats = np.full(shape=(4,num_agents),fill_value=np.inf)
        for traj_idx_tuple in all_possible_payoffs.keys():
            payoffs = np.full(shape=(num_agents,),fill_value=np.inf)
            for i in np.arange(num_agents):
                s_V = traj_dict_list[i][traj_idx_tuple[i]][:,5]
                s_traj = list(zip(traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[i][traj_idx_tuple[i]][:,3]))
                traj_len = utils.calc_traj_len(s_traj)
                
                ''' for now calculate the plan payoffs '''
                payoffs[i] = progress_payoffs_dist(traj_len)
            all_possible_payoffs_exc[traj_idx_tuple] = payoffs
        return all_possible_payoffs_exc
    
    
    def eval_pedestrian_inh_by_action(self,action):
        sv_id = int(action[3:6])
        ra_id = int(action[6:9])
        l1_act_code = int(action[9:11])
        l1_act_code = int(action[11:13])
        v_id = sv_id if ra_id==0 else ra_id
        relev_xwalk = utils.get_relevant_crosswalks(v_id)
        payoff = 1
        if relev_xwalk is not None:
            crosswalk_ids,near_gate = relev_xwalk[0], relev_xwalk[1]
            #current_segment = all_utils.get_current_segment_by_veh_id(v_id,self.eq_context.curr_time)
            pedestrian_info = self.eq_context.eval_config.pedestrian_info
            for xwalk in crosswalk_ids:
                for ped_state in pedestrian_info:
                    if xwalk in ped_state.crosswalks:
                        if ped_state.crosswalks[xwalk]['location'] == constants.ON_CROSSWALK:
                            payoff = payoff*1 if l1_act_code == 11 else payoff*0
                        elif ped_state.crosswalks[xwalk]['location'] == constants.BEFORE_CROSSWALK \
                                and ped_state.crosswalks[xwalk]['dist_to_entry'] < constants.PEDESTRIAN_CROSSWALK_DIST_THRESH \
                                    and ((ped_state.crosswalks[xwalk]['next_change'][1] == 'G' and ped_state.crosswalks[xwalk]['next_change'][0] <= constants.PEDESTRIAN_CROSSWALK_TIME_THRESH) \
                                             or ped_state.crosswalks[xwalk]['next_change'][1] == 'R'):
                            ''' before the crosswalk, within the distance threshold, and the signal is green or about to change to green'''
                            payoff = payoff*1 if l1_act_code == 11 else payoff*0
                        else:
                            payoff = payoff*1
        return payoff
       
    def eval_pedestrian_inhibitory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        payoff_vect = np.full(shape=(len(strategy_tuple),),fill_value=0)
        all_possible_payoffs_inh_ped = dict(all_possible_payoffs)
        for idx,action in enumerate(strategy_tuple):
            payoff = self.eval_pedestrian_inh_by_action(action)
            payoff_vect[idx] = payoff
        for traj_idx_tuple in all_possible_payoffs.keys():
            all_possible_payoffs_inh_ped[traj_idx_tuple] = payoff_vect
        return all_possible_payoffs_inh_ped
       
    def calc_l3_payoffs(self,eq,strategy_tuple,l3_utility_dict=None):
        if eq.eval_config.l3_eq is not None and l3_utility_dict is None:
            raise ValueError('L3 utility dict cannot be None when L3 Equilibria is set to None')
    
        if eq.eval_config.l3_eq is None:
            ''' payoffs will be calculated just from the baselines '''
            baseline_ids = [[int(x.split('-')[1])] for x in strategy_tuple]
            all_possible_payoffs = {k:np.zeros(shape=len(k)) for k in list(itertools.product(*[v for v in baseline_ids]))}
            traj_dict_list = [{t:eq.l1l2_trajectory_cache[k[0]] for t in k} for k in baseline_ids]
        else:
            all_possible_payoffs = l3_utility_dict
            traj_ids = list(all_possible_payoffs.keys())
            traj_dict_list = [dict() for n in np.arange(eq.eval_config.num_agents)]
            for k in traj_ids:
                for ag_idx,t in enumerate(k):
                    t_id = int(t.split('-')[1])
                    try:
                        traj_dict_list[ag_idx].update({t:eq.l3_trajectory_cache[int(strategy_tuple[ag_idx].split('-')[1])][t_id]})
                    except KeyError:
                        brk=1
            
        all_possible_payoffs_inh,all_possible_payoffs_exc = 0,0
        
        
        '''
        if len(eval_config.traj_dict) > 0:
            traj_dict_list = [{t:np.vstack(eval_config.traj_dict[k[0]]) for t in k} for k in eval_config.strat_traj_ids]
        else:
            traj_dict_list = all_utils.load_traj_from_db(chosen_trajectory_ids,baseline_only)
        '''
        
        
        if constants.INHIBITORY:
            #if N_size > 300:
            #    logging.info(str(u_ct)+'/'+str(N_size)+' evaluating inhibitory')
            #future_inh = executor.submit(self.eval_inhibitory, traj_dict_list, all_possible_payoffs, strategy_tuple)
            all_possible_payoffs_inh = self.eval_inhibitory(traj_dict_list, all_possible_payoffs, strategy_tuple)
        if constants.EXCITATORY:
            #if N_size > 300:
            #    logging.info(str(u_ct)+'/'+str(N_size)+' evaluating excitatory')
            #future_exc = executor.submit(self.eval_excitatory, traj_dict_list, all_possible_payoffs, strategy_tuple)
            all_possible_payoffs_exc = self.eval_excitatory(traj_dict_list, all_possible_payoffs, strategy_tuple)
        if constants.INHIBITORY_PEDESTRIAN:
            #if N_size > 300:
            #    logging.info(str(u_ct)+'/'+str(N_size)+' evaluating pedestrian_inhibitory')
            #future_pedinh = executor.submit(self.eval_pedestrian_inhibitory, traj_dict_list, all_possible_payoffs, strategy_tuple)
            all_possible_payoffs_inh_ped = self.eval_pedestrian_inhibitory(traj_dict_list, all_possible_payoffs, strategy_tuple)
        #all_possible_payoffs_inh = future_inh.result()
        #all_possible_payoffs_exc = future_exc.result()
        #all_possible_payoffs_inh_ped = future_pedinh.result()
        for k,v in all_possible_payoffs.items():
            if constants.INHIBITORY and constants.EXCITATORY:
                all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.INHIBITORY_PAYOFF_WEIGHT * all_possible_payoffs_inh[k])
                all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.EXCITATORY_PAYOFF_WEIGHT * all_possible_payoffs_exc[k])
                all_possible_payoffs[k] = ((1-constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT) * all_possible_payoffs[k]) + (constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT * all_possible_payoffs_inh_ped[k])
            else:
                if constants.EXCITATORY:
                    all_possible_payoffs[k] = all_possible_payoffs[k] +  all_possible_payoffs_exc[k]
                if constants.INHIBITORY:
                    all_possible_payoffs[k] = all_possible_payoffs[k] + all_possible_payoffs_inh[k]
        
        return all_possible_payoffs
    
    def calc_maxmin_payoff(self,s_traj,r_traj,s_act_key,r_act_key):
        if constants.INHIBITORY and r_traj is not None:
            slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_traj.shape[0],r_traj.shape[0]))
            s_traj,r_traj = s_traj[:slice_len],r_traj[:slice_len]
            s_x,s_y = s_traj[:,1], s_traj[:,2]
            r_x,r_y = r_traj[:,1], r_traj[:,2]
            _d = np.hypot(s_x-r_x,s_y-r_y)
            min_dist = np.amin(_d)
            inh_payoff = dist_payoffs(min_dist)
        else:
            inh_payoff = 1
        if constants.EXCITATORY:
            traj_len = utils.calc_traj_len(s_traj)
            ''' for now calculate the plan payoffs '''
            exc_payoff = progress_payoffs_dist(traj_len)
        if constants.INHIBITORY_PEDESTRIAN:
            ped_inh_payoff = self.eval_pedestrian_inh_by_action(s_act_key)
        final_payoff = ( (1-constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT) * ((constants.INHIBITORY_PAYOFF_WEIGHT*inh_payoff) + (constants.EXCITATORY_PAYOFF_WEIGHT*exc_payoff)) ) + \
                            (constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT*ped_inh_payoff)
        return final_payoff

def eval_trajectory_viability(traj_id_list):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    cplx_list = []
    for t_l in traj_id_list:
        q_string = "SELECT * FROM GENERATED_TRAJECTORY_COMPLEXITY where GENERATED_TRAJECTORY_COMPLEXITY.COMPLEXITY is not null and TRAJECTORY_ID in "+str(tuple(t_l))
        c.execute(q_string)
        complexity_list = [(x[0],round(float(x[1]),5)) for x in c.fetchall()]
        _sum = np.sum([x[1] for x in complexity_list])
        complexity_list = [(x[0],x[1]/_sum) for x in complexity_list]
        ''' now complement the complexities to calculate viability'''
        _sum = np.sum([1-x[1] for x in complexity_list])
        complexity_list = [(x[0],(1-x[1])/_sum) for x in complexity_list]
        cplx_list.append(complexity_list)
    return cplx_list
    


def dist_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.DIST_COST_MEAN) / (constants.DIST_COST_SD * math.sqrt(2)))

def progress_payoffs_velocity(dist_arr):
    return scipy.special.erf((dist_arr - constants.SPEED_COST_MEAN) / (constants.SPEED_COST_SD * math.sqrt(2)))

def progress_payoffs_dist(traj_len):
    if traj_len >= 100:
        return 1
    else:
        return traj_len/100

def eval_regulatory(traj_list,time_ts,dist_arr,traffic_signal,strat_str):
    g = 1
    

def calc_baseline_traj_payoffs(eval_config):
    cost_eval = CostEvaluation(eval_config)
    num_agents = eval_config.num_agents
    chosen_trajectory_ids = eval_config.strat_traj_ids
    chosen_trajectory_id_combinations = list(itertools.product(*chosen_trajectory_ids))
    all_possible_payoffs = cost_eval.calc_l3_payoffs(chosen_trajectory_id_combinations,chosen_trajectory_ids,num_agents,eval_config,True)
    return all_possible_payoffs
    


    
def eval_trajectory_complexity_unloaded(traj_list,strategy_tuple):
    f = 1
    all_traj_compl = []
    for strat,file_str in zip(strategy_tuple, [x[0][1] for x in traj_list]):
        traj_complexity = []
        trajectories = utils.load_traj_from_str(file_str)
        for traj in trajectories:
            compl = utils.eval_complexity(traj,strat)
            traj_complexity.append(compl)
        _sum = sum(traj_complexity)
        traj_complexity = [x/_sum for x in traj_complexity]
        all_traj_compl.append(traj_complexity)
    return all_traj_compl    
        
        
        
        
        
    

  

def unreadable(act_str):
    tokens = act_str.split('|')
    assert(len(tokens)==4)
    l1_action = str(constants.L1_ACTION_CODES[tokens[2]]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[tokens[3]]).zfill(2)
    agent = str(tokens[0]).zfill(3)
    relev_agent = str(tokens[1]).zfill(3)
    unreadable_str = '769'+agent+relev_agent+l1_action+l2_action
    return unreadable_str

