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


class EvalConfig:
    
    def __init__(self,eval_type):
        self.eval_type = eval_type
        self.traj_dict = dict()
        strat_traj_ids = []
        
    def set_l3_traj_dict(self,traj_dict):
        self.traj_dict = traj_dict
        
    def set_num_agents(self,num_agents):
        self.num_agents = num_agents
        
    def set_traffic_signal(self,traffic_signal):
        self.traffic_signal = traffic_signal
        
    def set_curr_strat_tuple(self,strat_tuple):
        self.strat_tuple = strat_tuple
        
    def set_curr_beliefs(self,strat_beliefs):
        self.belief_tuple = strat_beliefs
        
    def set_curr_strat_traj_ids(self, curr_traj_ids):
        self.strat_traj_ids = curr_traj_ids


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
    num_agents = eval_config.num_agents
    chosen_trajectory_ids = eval_config.strat_traj_ids
    chosen_trajectory_id_combinations = list(itertools.product(*chosen_trajectory_ids))
    all_possible_payoffs = calc_l3_payoffs(chosen_trajectory_id_combinations,chosen_trajectory_ids,num_agents,eval_config,True)
    return all_possible_payoffs
    
def calc_l3_payoffs(chosen_trajectory_id_combinations,chosen_trajectory_ids,num_agents,eval_config,baseline_only):
    strategy_tuple = eval_config.strat_tuple
    all_possible_payoffs = dict(zip(chosen_trajectory_id_combinations, [[0]*num_agents]*len(chosen_trajectory_id_combinations)))
    all_possible_payoffs_inh,all_possible_payoffs_exc = 0,0
    if len(eval_config.traj_dict) > 0:
        traj_dict_list = [{t:np.vstack(eval_config.traj_dict[k[0]]) for t in k} for k in eval_config.strat_traj_ids]
    else:
        traj_dict_list = utils.load_traj_from_db(chosen_trajectory_ids,baseline_only)
    if constants.INHIBITORY:
        all_possible_payoffs_inh = eval_inhibitory(traj_dict_list, all_possible_payoffs, strategy_tuple)
    if constants.EXCITATORY:
        all_possible_payoffs_exc = eval_excitatory(traj_dict_list, all_possible_payoffs, strategy_tuple)
    for k in chosen_trajectory_id_combinations:
        if constants.INHIBITORY and constants.EXCITATORY:
            all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.INHIBITORY_PAYOFF_WEIGHT * all_possible_payoffs_inh[k])
            all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.EXCITATORY_PAYOFF_WEIGHT * all_possible_payoffs_exc[k])
        else:
            if constants.EXCITATORY:
                all_possible_payoffs[k] = all_possible_payoffs[k] +  all_possible_payoffs_exc[k]
            if constants.INHIBITORY:
                all_possible_payoffs[k] = all_possible_payoffs[k] + all_possible_payoffs_inh[k]
    
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

def eval_complexity(traj,strat_str):
    if isinstance(traj, pd.DataFrame):
        rv = traj['v'].to_numpy()
        rx = traj['x'].to_numpy()
        ry = traj['y'].to_numpy()
        ra = traj['a'].to_numpy()
        time = traj['time'].to_numpy()
    else:
        assert(traj.shape[1]==9)
        rv,rx,ry,ra,time = traj[:,6],traj[:,3],traj[:,4],traj[:,7],traj[:,2]
    dt = constants.LP_FREQ
    traj_def = TrajectoryDef(strat_str)
    '''
    plt.plot(time,rv,'g')
    plt.plot(time,ra,'r')
    plt.show()
    #linear_plan_x = utils.linear_planner(sx, vxs, axs, gx, vxg, axg, max_accel, max_jerk, dt)
    '''
    hpx = [rx[0],rx[len(rx)//3],rx[2*len(rx)//3],rx[-1]]
    hpy = [ry[0],ry[len(ry)//3],ry[2*len(ry)//3],ry[-1]]
    _d = OrderedDict(sorted(list(zip(hpx,hpy)),key=lambda tup: tup[0]))
    hpx,hpy = list(_d.keys()),list(_d.values())
    try:
        cs = CubicSpline(hpx, hpy)
    except ValueError:
        print(hpx,hpy)
        raise
    path_residuals = sum([abs(cs(x)-ry[_i]) for _i,x in enumerate(rx)])
    new_vels = utils.generate_baseline_trajectory(time,[(x,cs(x)) for x in rx],rv[0],ra[0],traj_def.max_acc_long/2,traj_def.max_jerk/2,rv[-1],dt,traj_def.acc)
    slice_len = min(len(new_vels),len(rv))
    vel_residuals = sum([abs(x[0]-x[1]) for x in zip(rv[:slice_len],  new_vels[:slice_len])])
    compl = math.hypot(path_residuals, vel_residuals)
    return compl
    
def eval_trajectory_complexity_unloaded(traj_list,strategy_tuple):
    f = 1
    all_traj_compl = []
    for strat,file_str in zip(strategy_tuple, [x[0][1] for x in traj_list]):
        traj_complexity = []
        trajectories = utils.load_traj_from_str(file_str)
        for traj in trajectories:
            compl = eval_complexity(traj,strat)
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
                    s_x,r_x = traj_dict_list[i][traj_idx_tuple[i]][:,1],traj_dict_list[j][traj_idx_tuple[j]][:,1]
                    ''' for now calculate the plan payoffs instead of payoffs for the next 1 second '''
                    #slice_len = min(s_x.shape[0],r_x.shape[0])
                    slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_x.shape[0],r_x.shape[0]))
                    s_x,r_x = s_x[:slice_len],r_x[:slice_len]
                    s_y,r_y = traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[j][traj_idx_tuple[j]][:,2]
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
            s_V = traj_dict_list[i][traj_idx_tuple[i]][:,4]
            s_traj = list(zip(traj_dict_list[i][traj_idx_tuple[i]][:,1],traj_dict_list[i][traj_idx_tuple[i]][:,2]))
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

