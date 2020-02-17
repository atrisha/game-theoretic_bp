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

def dist_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.DIST_COST_MEAN) / (constants.DIST_COST_SD * math.sqrt(2)))

def progress_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.SPEED_COST_MEAN) / (constants.SPEED_COST_SD * math.sqrt(2)))

''' given a strategy combination, evaluate the vector of payoffs for each agent.'''
def eval_inhibitory(traj_list,a_c):
    disp_arr_x,disp_arr_y = [],[] 
    num_agents = len(traj_list)
    len_of_trajectories = [len(x) for x in traj_list]
    all_trajectory_indices = list(itertools.product(*[np.arange(x) for x in len_of_trajectories]))
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
                    s_x,r_x = trajectory_dataframes[i]['x'].as_matrix(),trajectory_dataframes[j]['x'].as_matrix()
                    ''' for now calculate the plan payoffs instead of payoffs for the next 1 second '''
                    slice_len = min(s_x.shape[0],r_x.shape[0])
                    #slice_len = min(constants.PLAN_FREQ/constants.LP_FREQ,s_x.shape[0],r_x.shape[0])
                    s_x,r_x = s_x[:slice_len],r_x[:slice_len]
                    s_y,r_y = trajectory_dataframes[i]['y'].as_matrix(),trajectory_dataframes[j]['y'].as_matrix()
                    slice_len = min(s_y.shape[0],r_y.shape[0])
                    s_y,r_y = s_y[:slice_len],r_y[:slice_len]
                    _d = np.hypot(s_x-r_x,s_y-r_y)
                    dist_among_agents[i,j] = min(_d)
        ''' to be safe, make the matrix symmetric '''
        dist_among_agents = np.minimum(dist_among_agents,dist_among_agents.T)
        ''' find the minimum distance for a vehicle action given all other agent actions '''
        dist = np.amin(dist_among_agents,axis=1)
        payoffs = dist_payoffs(dist)
        all_possible_payoffs[traj_idx_tuple] = payoffs
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
    return payoff_stats,payoff_stats_trajectories
    
def eval_excitatory(traj_arr,a_c):
    payoffs = np.empty(shape=(len(traj_arr)))
    for i,s_v in enumerate(a_c):
        s_V= traj_arr[i]['v'].as_matrix()
        ''' for now calculate the plan payoffs '''
        payoffs[i] = progress_payoffs(np.mean(traj_arr[i]['v'].as_matrix()))
    return np.reshape(payoffs,newshape=(len(a_c),1))
        

def calc_equilibria(cache_dir,pattern,payoff_type):
    print(pattern)
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
    payoff_str = ''
    for a_c in all_action_combinations:
        traj_list = []
        for a in a_c:
            if a not in traj_dict:
                traj = utils.pickle_load(os.path.join(dir, a))
                #traj = np.array(traj[0])
                p_d_list = []
                for i in np.arange(traj.shape[0]):
                    _t_data,_t_type = traj[i][0],traj[i][1]
                    if isinstance(_t_data,np.ndarray):
                        s = pd.DataFrame(_t_data, index=data_index, dtype = np.float).T
                        p_d_list.append(s)
                #p_d_list = pd.DataFrame(traj[0], index=data_index).T
                    
                traj_dict[a] = p_d_list
                traj_list.append(p_d_list)
            else:
                traj_list.append(traj_dict[a])
            #print(a_c,a,traj[1])
                
        #traj_arr = np.array(traj_list)
        #print(a_c)
        if a_c == ('7690110000202_1000', '7690110010401_1000', '7690110030301_1000', '7690110200401_1000'):
            brk = 1
        inhibitory_payoffs,traj_indices = eval_inhibitory(traj_list,a_c)
        #excitatory_payoffs = 0 * eval_excitatory(traj_list, a_c)
        #print(a_c,np.concatenate((inhibitory_payoffs,excitatory_payoffs),axis=1))
        #pay_offs = np.reshape(inhibitory_payoffs+excitatory_payoffs,newshape=(4,1))
        #print(a_c,(inhibitory_payoffs+excitatory_payoffs).tolist())
        if a_c not in pay_off_dict:
            #pay_off_dict[a_c] = [x[0] for x in np.reshape((inhibitory_payoffs+excitatory_payoffs),newshape=(4,1)).tolist()]
            pay_off_dict[a_c] = inhibitory_payoffs
            payoff_trajectories_indices_dict[a_c] = traj_indices
        traj_ct += 1
        #print(traj_ct)
        '''
        for t in traj_list:
            t_arr = np.asarray(t)
            eval_inhibitory(t_arr)
            if t_arr.shape[0] > 1:
                t_arr = t_arr[0,:]
                t_X = t_arr[1,]
                t_Y = t_arr[2,]
            else:
                t_X = t_arr[0,1]
                t_Y = t_arr[0,2]
            #ax.plot(t_X,t_Y,'-')
        #print(traj_ct)
        ax.ticklabel_format(useOffset=False)
        plt.show()
        '''
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
            print(e)
            traj_indices = payoff_trajectories_indices_dict[e]
            equilibria_actions.append((e,traj_indices[i,:]))
            traj_xs,traj_ys = [],[]
            for j in np.arange(num_agents):
                traj_xs.append(traj_dict[e[j]][int(traj_indices[0,j])]['x'])
                traj_ys.append(traj_dict[e[j]][int(traj_indices[0,j])]['y'])
            #visualizer.show_animation(traj_xs, traj_ys)
            
            
    #print(len(all_action_combinations))
    return equilibria_actions

def calc_eqs_for_hopping_trajectories():
    ''' since this is the time horizon'''
    for i in np.arange(13):
        if i!= 0:
            pattern = '769011......._'+str(i)+'0..'
        else:
            pattern = '769011......._'+str(i)
        calc_equilibria(pattern)
                

