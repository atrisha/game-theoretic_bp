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
import equilibria
from scipy.interpolate import CubicSpline
from planning_objects import TrajectoryDef

def dist_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.DIST_COST_MEAN) / (constants.DIST_COST_SD * math.sqrt(2)))

def progress_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.SPEED_COST_MEAN) / (constants.SPEED_COST_SD * math.sqrt(2)))

def eval_regulatory(traj_list,curr_time,dist_arr,traffic_signal,strat_str):
    g = 1

def calc_total_payoffs(inhibitory,excitatory,traj_list,strategy_tuple,traffic_signal):
    num_agents = len(traj_list)
    all_traj_complexity = eval_trajectory_complexity(traj_list,strategy_tuple)
    len_of_trajectories = [len(x) for x in traj_list]
    #print('N=',len_of_trajectories)
    ''' complexity weighted '''
    all_trajectory_indices = list(itertools.product(*[np.random.choice(np.arange(x),size = min(5,x), replace=False, p=all_traj_complexity[_i]) for _i,x in enumerate(len_of_trajectories)]))
    ''' random '''
    #all_trajectory_indices = list(itertools.product(*[np.random.choice(np.arange(x),size = max(1,x//5,x//4), replace=False) for _i,x in enumerate(len_of_trajectories)]))
    '''all - not possible (just for reference)'''
    #all_trajectory_indices = list(itertools.product(*[np.arange(x) for _i,x in enumerate(len_of_trajectories)]))
    #print('N*=',len(all_trajectory_indices))
    all_possible_payoffs = dict(zip(all_trajectory_indices, [0]*len(all_trajectory_indices)))
    all_possible_payoffs_inh,all_possible_payoffs_exc = 0,0
    if inhibitory:
        all_possible_payoffs_inh = eval_inhibitory(traj_list, strategy_tuple,all_trajectory_indices)
    if excitatory:
        all_possible_payoffs_exc = eval_excitatory(traj_list, strategy_tuple,all_trajectory_indices)
    for k in all_trajectory_indices:
        if inhibitory and excitatory:
            all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.INHIBITORY_PAYOFF_WEIGHT * all_possible_payoffs_inh[k])
            all_possible_payoffs[k] = all_possible_payoffs[k] + (constants.EXCITATORY_PAYOFF_WEIGHT * all_possible_payoffs_exc[k])
        else:
            if excitatory:
                all_possible_payoffs[k] = all_possible_payoffs[k] +  all_possible_payoffs_exc[k]
            if inhibitory:
                all_possible_payoffs[k] = all_possible_payoffs[k] + all_possible_payoffs_inh[k]
    
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
        payoff_stats_trajectories[1,i] = all_possible_payoffs_keys[np.where(all_possible_payoffs_vals[:,i]==payoff_stats[1,i])[0][0]][i]
    payoff_stats[2,] = np.mean(all_possible_payoffs_vals,axis = 0)
    for i in np.arange(num_agents):
        payoff_stats_trajectories[2,i] = all_possible_payoffs_keys[utils.find_nearest_in_array(all_possible_payoffs_vals[:,i],payoff_stats[2,i])][i]
    payoff_stats[3,] = np.std(all_possible_payoffs_vals,axis = 0)
    ''' equilibria for l3 actions'''
    #visualizer.plot_payoff_grid(all_possible_payoffs_vals)
    #print('calculating l3 equilibria')
    eq = equilibria.calc_pure_strategy_nash_equilibrium_exhaustive(all_possible_payoffs)
    eq_traj_indices = eq
    
    return payoff_stats,payoff_stats_trajectories        
    
def eval_trajectory_complexity(traj_list,strategy_tuple):
    f = 1
    all_traj_compl = []
    for strat,trajectories in zip(strategy_tuple, traj_list):
        traj_complexity = []
        for traj in trajectories:
            rv = traj['v'].to_numpy()
            rx = traj['x'].to_numpy()
            ry = traj['y'].to_numpy()
            ra = traj['a'].to_numpy()
            time = traj['time'].to_numpy()
            dt = constants.LP_FREQ
            traj_def = TrajectoryDef(strat)
            l1_action = traj_def.l1_action_readable
            '''
            plt.plot(time,rv,'g')
            plt.plot(time,ra,'r')
            plt.show()
            #linear_plan_x = utils.linear_planner(sx, vxs, axs, gx, vxg, axg, max_accel, max_jerk, dt)
            '''
            hpx = [rx[0],rx[len(rx)//3],rx[2*len(rx)//3],rx[-1]]
            hpy = [ry[0],ry[len(ry)//3],ry[2*len(ry)//3],ry[-1]]
            if hpx[-1] < hpx[0]:
                hpx.reverse()
                hpy.reverse()
            cs = CubicSpline(hpx, hpy)
            path_residuals = sum([abs(cs(x)-ry[_i]) for _i,x in enumerate(rx)])
            new_vels = utils.generate_baseline_trajectory(time,[(x,cs(x)) for x in rx],rv[0],ra[0],traj_def.max_acc_long,traj_def.max_jerk,rv[-1],dt,l1_action)
            slice_len = min(len(new_vels),len(rv))
            vel_residuals = sum([abs(x[0]-x[1]) for x in zip(rv[:slice_len],  new_vels[:slice_len])])
            compl = math.hypot(path_residuals, vel_residuals)
            traj_complexity.append(compl)
            '''
            plt.plot(rx, ry, 'b', rx, cs(rx), 'r')
            plt.show()
            '''        
        _sum = sum(traj_complexity)
        traj_complexity = [x/_sum for x in traj_complexity]
        all_traj_compl.append(traj_complexity)
    return all_traj_compl    
        
        
        
        
        
''' given a strategy combination, evaluate the vector of payoffs for each agent.'''
def eval_inhibitory(traj_list,a_c,all_trajectory_indices):
    disp_arr_x,disp_arr_y = [],[] 
    num_agents = len(traj_list)
    len_of_trajectories = [len(x) for x in traj_list]
    all_possible_payoffs = dict()
    for traj_idx_tuple in all_trajectory_indices:
        trajectory_dataframes = []
        for i in np.arange(num_agents):
            _data_frames = traj_list[i][traj_idx_tuple[i]]
            trajectory_dataframes.append(_data_frames)
        ''' pair-wise min distance matrix '''
        dist_among_agents = np.full(shape=(num_agents,num_agents),fill_value=np.inf)
        dist = np.full(shape=(num_agents),fill_value=np.inf)
        for i in np.arange(num_agents):
            for j in np.arange(i,num_agents):
                if i != j:
                    s_x,r_x = trajectory_dataframes[i]['x'],trajectory_dataframes[j]['x']
                    ''' for now calculate the plan payoffs instead of payoffs for the next 1 second '''
                    #slice_len = min(s_x.shape[0],r_x.shape[0])
                    slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_x.shape[0],r_x.shape[0]))
                    s_x,r_x = s_x[:slice_len],r_x[:slice_len]
                    s_y,r_y = trajectory_dataframes[i]['y'],trajectory_dataframes[j]['y']
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
    
def eval_excitatory(traj_list,a_c,all_trajectory_indices):
    num_agents = len(traj_list)
    payoff_stats_trajectories = np.full(shape=(4,num_agents),fill_value=np.inf)
    payoff_stats = np.full(shape=(4,num_agents),fill_value=np.inf)
    len_of_trajectories = [len(x) for x in traj_list]
    all_possible_payoffs = dict()
    for traj_idx_tuple in all_trajectory_indices:
        payoffs = np.full(shape=(num_agents,),fill_value=np.inf)
        for i in np.arange(num_agents):
            _data_frames = traj_list[i][traj_idx_tuple[i]]
            s_V= _data_frames['v'].as_matrix()
            ''' for now calculate the plan payoffs '''
            payoffs[i] = progress_payoffs(np.mean(s_V)) + constants.L2_ACTION_PAYOFF_ADDITIVE
        all_possible_payoffs[traj_idx_tuple] = payoffs
    return all_possible_payoffs
    
   
def print_readable(eq):
    readable_eq = []
    for s in eq:
        readable_eq.append(s[3:6]+'_'+s[6:9]+'_'+constants.L1_ACTION_CODES_2_NAME[int(s[9:11])]+'_'+constants.L2_ACTION_CODES_2_NAME[int(s[11:13])])
    return readable_eq

''' calculate equilibria in a brute force method '''
def calc_equilibria(cache_dir,pattern,payoff_type,curr_time):
    print(pattern)
    traffic_signal = utils.get_traffic_signal(curr_time, 'ALL',pattern[0:3])
    dir = cache_dir
    data_index = ['time', 'x', 'y', 'yaw', 'v', 'a', 'j']
    action_dict = dict()
    agent_ids = []
    ag_ct = 0
    for f in listdir(dir):
        if re.match(pattern, f):
            agent_string,time_ts = f.split('_')
            agent_id = agent_string[3:6]
            relev_agent_id = agent_string[6:9]
            full_agent_id = agent_id+relev_agent_id
            if full_agent_id not in agent_ids:
                agent_ids.append(full_agent_id)
            action_string = agent_string[9:]
            agent_key = agent_id+relev_agent_id
            if agent_key not in action_dict:
                action_dict[agent_key] = dict()
                action_dict[agent_key]['actions'] = ['769'+agent_key+action_string+'_'+time_ts]
                action_dict[agent_key]['id'] = ag_ct
                ag_ct += 1
            else:
                action_dict[agent_key]['actions'].append('769'+agent_key+action_string+'_'+time_ts)
        else:
            match = False
    all_action_combinations = list(itertools.product(*[v['actions'] for v in action_dict.values()]))
    #fig, ax = plt.subplots()
    traj_ct = 0
    traj_dict = dict()
    pay_off_dict = dict()
    num_agents = len(agent_ids)
    payoff_trajectories_indices_dict = dict()
    for a_c in all_action_combinations:
        traj_list = []
        for a in a_c:
            if a not in traj_dict:
                traj = utils.pickle_load(os.path.join(dir, a))
                p_d_list = []
                for i in np.arange(traj.shape[0]):
                    _t_data,_t_type = traj[i][0],traj[i][1]
                    if isinstance(_t_data,np.ndarray):
                        s = pd.DataFrame(_t_data, index=data_index, dtype = np.float).T
                        p_d_list.append(s)
                
                    
                traj_dict[a] = p_d_list
                traj_list.append(p_d_list)
            else:
                traj_list.append(traj_dict[a])
            #print(a_c,a,traj[1])
                
        ''' these strategies are not valid since once of them has no trajectory'''
        if 0 in [len(x) for x in traj_list]:
            continue
        #print('calculating payoffs')
        payoffs,traj_indices = calc_total_payoffs(True,False,traj_list,a_c,traffic_signal)
        #print('calculating payoffs....DONE')
        if a_c not in pay_off_dict:
            pay_off_dict[a_c] = payoffs
            payoff_trajectories_indices_dict[a_c] = traj_indices
        traj_ct += 1
        print(traj_ct)
    seq = ['max','min','mean']
    equilibria_actions = []
    for i in np.arange(len(seq)):
        if payoff_type is not None and payoff_type != seq[i]:
            continue
        _t_p_dict = dict()
        for k,v in pay_off_dict.items():
            _t_p_dict[k] = v[i,:]
        eq = equilibria.calc_pure_strategy_nash_equilibrium_exhaustive(_t_p_dict)
        
        print(seq[i],'equilibria are')
        for e in eq:
            print(print_readable(e), [round(float(p),4) for p in _t_p_dict[e]])
            traj_indices = payoff_trajectories_indices_dict[e]
            equilibria_actions.append((e,traj_indices[i,:],_t_p_dict[e]))
            traj_xs,traj_ys = [],[]
            for j in np.arange(num_agents):
                traj_xs.append(traj_dict[e[j]][int(traj_indices[0,j])]['x'])
                traj_ys.append(traj_dict[e[j]][int(traj_indices[0,j])]['y'])
            #visualizer.show_animation(traj_xs, traj_ys)
            
            
    #print(len(all_action_combinations))
    return equilibria_actions


                

#calc_eqs_for_hopping_trajectories()