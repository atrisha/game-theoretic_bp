'''
Created on Feb 6, 2020

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
import equilibria_core
from scipy.interpolate import CubicSpline
from planning_objects import TrajectoryDef
from collections import OrderedDict
import db_utils








class CostEvaluation():
    
       
    def calc_l3_payoffs(self,eq,strategy_tuple):
        if eq.eval_config.l3_eq is None:
            ''' payoffs will be calculated just from the baselines '''
            baseline_ids = [[int(x.split('-')[1])] for x in strategy_tuple]
            all_possible_payoffs = {k:np.zeros(shape=len(k)) for k in list(itertools.product(*[v for v in baseline_ids]))}
            traj_dict_list = [{t:eq.trajectory_cache[k[0]] for t in k} for k in baseline_ids]
        else:
            ''' payoffs will be calculated based on some equilibrium'''
            all_possible_payoffs = eq.l3_utility_dict
        all_possible_payoffs_inh,all_possible_payoffs_exc = 0,0
        
        
        '''
        if len(eval_config.traj_dict) > 0:
            traj_dict_list = [{t:np.vstack(eval_config.traj_dict[k[0]]) for t in k} for k in eval_config.strat_traj_ids]
        else:
            traj_dict_list = utils.load_traj_from_db(chosen_trajectory_ids,baseline_only)
        '''
        if constants.INHIBITORY:
            all_possible_payoffs_inh = eval_inhibitory(traj_dict_list, all_possible_payoffs, strategy_tuple)
        if constants.EXCITATORY:
            all_possible_payoffs_exc = eval_excitatory(traj_dict_list, all_possible_payoffs, strategy_tuple)
        for k,v in all_possible_payoffs.items():
            if constants.INHIBITORY and constants.EXCITATORY:
                all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.INHIBITORY_PAYOFF_WEIGHT * all_possible_payoffs_inh[k])
                all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.EXCITATORY_PAYOFF_WEIGHT * all_possible_payoffs_exc[k])
            else:
                if constants.EXCITATORY:
                    all_possible_payoffs[k] = all_possible_payoffs[k] +  all_possible_payoffs_exc[k]
                if constants.INHIBITORY:
                    all_possible_payoffs[k] = all_possible_payoffs[k] + all_possible_payoffs_inh[k]
        
        return all_possible_payoffs


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
    

def calc_l3_equilibrium_payoffs(inhibitory,excitatory,traj_id_list,strategy_tuple,traffic_signal):
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
    all_possible_payoffs = calc_l3_payoffs(chosen_trajectory_id_combinations,chosen_trajectory_ids,num_agents,strategy_tuple)
    
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
        
        
        
        
        
''' given a strategy combination, evaluate the vector of payoffs for each agent.'''
def eval_inhibitory(traj_dict_list, all_possible_payoffs, strategy_tuple):
    disp_arr_x,disp_arr_y = [],[] 
    num_agents = len(strategy_tuple)
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
                    _d = np.hypot(s_x-r_x,s_y-r_y)
                    dist_among_agents[i,j] = min(_d)
        ''' to be safe, make the matrix symmetric '''
        dist_among_agents = np.minimum(dist_among_agents,dist_among_agents.T)
        ''' find the minimum distance for a vehicle action given all other agent actions '''
        dist = np.amin(dist_among_agents,axis=1)
        #payoffs = dist_payoffs(dist) + constants.L2_ACTION_PAYOFF_ADDITIVE
        payoffs = dist_payoffs(dist)
        all_possible_payoffs[traj_idx_tuple] = payoffs
    return all_possible_payoffs
    
def eval_excitatory(traj_dict_list, all_possible_payoffs, strategy_tuple):
    num_agents = len(strategy_tuple)
    payoff_stats_trajectories = np.full(shape=(4,num_agents),fill_value=np.inf)
    payoff_stats = np.full(shape=(4,num_agents),fill_value=np.inf)
    for traj_idx_tuple in all_possible_payoffs.keys():
        payoffs = np.full(shape=(num_agents,),fill_value=np.inf)
        for i in np.arange(num_agents):
            s_V = traj_dict_list[i][traj_idx_tuple[i]][:,5]
            s_traj = list(zip(traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[i][traj_idx_tuple[i]][:,3]))
            traj_len = utils.calc_traj_len(s_traj)
            
            ''' for now calculate the plan payoffs '''
            payoffs[i] = progress_payoffs_dist(traj_len)
        all_possible_payoffs[traj_idx_tuple] = payoffs
    return all_possible_payoffs
    
   


  

def unreadable(act_str):
    tokens = act_str.split('|')
    assert(len(tokens)==4)
    l1_action = str(constants.L1_ACTION_CODES[tokens[2]]).zfill(2)
    l2_action = str(constants.L2_ACTION_CODES[tokens[3]]).zfill(2)
    agent = str(tokens[0]).zfill(3)
    relev_agent = str(tokens[1]).zfill(3)
    unreadable_str = '769'+agent+relev_agent+l1_action+l2_action
    return unreadable_str

