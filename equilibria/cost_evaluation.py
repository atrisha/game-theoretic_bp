'''
Created on Feb 6, 2020

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3
from all_utils import utils, stat_utils
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
from matplotlib.patches import Ellipse
import time
import threading
from distributed.worker import weight

log = constants.common_logger




def calc_time_gap(ag_obj,ra_obj):
    ttc = None
    if (ag_obj.leading_vehicle is not None and ag_obj.leading_vehicle.id == ra_obj.id) or ag_obj.task == ra_obj.task:
        ttc =  np.inf
    else:
        ag_time_to_collision_region,ra_time_to_collision_region = None,None
        if ag_obj.task == 'LEFT_TURN' or ag_obj.task == 'RIGHT_TURN' :
            if ag_obj.origin_pt is not None and ag_obj.segment_seq[0] in constants.TURN_TIME_MAP:
                slope,intercept = utils.get_slope_intercept(constants.TURN_TIME_MAP[ag_obj.segment_seq[0]])
                dist_from_origin_pt = math.hypot(ag_obj.x-ag_obj.origin_pt[0], ag_obj.y-ag_obj.origin_pt[1])
                ag_time_to_collision_region = slope*dist_from_origin_pt + intercept
        if ra_obj.task == 'LEFT_TURN' or ra_obj.task == 'RIGHT_TURN' :
            if ra_obj.origin_pt is not None and ra_obj.segment_seq[0] in constants.TURN_TIME_MAP:
                slope,intercept = utils.get_slope_intercept(constants.TURN_TIME_MAP[ra_obj.segment_seq[0]])
                dist_from_origin_pt = math.hypot(ra_obj.x-ra_obj.origin_pt[0], ra_obj.y-ra_obj.origin_pt[1])
                ra_time_to_collision_region = slope*dist_from_origin_pt + intercept
        if ag_obj.task == 'STRAIGHT':
            ''' ra_obj has to be left or right turning, so find the common conflict region '''
            if ra_obj.task == 'LEFT_TURN' or ra_obj.task == 'RIGHT_TURN' :
                if ag_obj.segment_seq[0] == 'ln_s_2' and ag_obj.segment_seq[-1] == 'ln_w_-1':
                    conflict_pt = utils.get_conflict_points(['ln_s_1', 'prep-turn_s', 'exec-turn_s', 'ln_w_-1'], ag_obj.signal, ra_obj.segment_seq, ra_obj.signal)
                elif ra_obj.segment_seq[0] == 'ln_s_2' and ra_obj.segment_seq[-1] == 'ln_w_-1':
                    conflict_pt = utils.get_conflict_points(ag_obj.segment_seq, ag_obj.signal, ['ln_s_1', 'prep-turn_s', 'exec-turn_s', 'ln_w_-1'], ra_obj.signal)
                else:
                    conflict_pt = utils.get_conflict_points(ag_obj.segment_seq, ag_obj.signal, ra_obj.segment_seq, ra_obj.signal)
                if conflict_pt is not None:
                    ''' agents has not crossed the conflict point'''
                    if (utils.zero_pi_angle_bet_lines([(ag_obj.x,ag_obj.y),((ag_obj.nose_x,ag_obj.nose_y))], [(ag_obj.x,ag_obj.y),conflict_pt]) < 90 or math.hypot(ag_obj.x-conflict_pt[0],ag_obj.y-conflict_pt[1]) <= constants.CAR_LENGTH):
                        
                        
                        if ag_obj.long_acc != 0:
                            if ag_obj.speed**2 - (4*ag_obj.long_acc/2*(-math.hypot(ag_obj.x-conflict_pt[0],ag_obj.y-conflict_pt[1]))) < 0:
                                ag_time_to_collision_region = np.inf
                            else:
                                t = utils.solve_quadratic(ag_obj.long_acc/2, ag_obj.speed, -math.hypot(ag_obj.x-conflict_pt[0],ag_obj.y-conflict_pt[1]))
                                ag_time_to_collision_region = max(t[0],t[1])
                        else:
                            t = math.hypot(ag_obj.x-conflict_pt[0],ag_obj.y-conflict_pt[1]) / ag_obj.speed if ag_obj.speed != 0 else np.inf
                            ag_time_to_collision_region = t
                        
                    else:
                        ag_time_to_collision_region = np.inf
                else:
                    ag_time_to_collision_region = np.inf
            else:
                ttc =  np.inf
        if ra_obj.task == 'STRAIGHT':
            if ag_obj.task == 'LEFT_TURN' or ag_obj.task == 'RIGHT_TURN' :
                if ag_obj.segment_seq[0] == 'ln_s_2' and ag_obj.segment_seq[-1] == 'ln_w_-1':
                    conflict_pt = utils.get_conflict_points(['ln_s_1', 'prep-turn_s', 'exec-turn_s', 'ln_w_-1'], ag_obj.signal, ra_obj.segment_seq, ra_obj.signal)
                elif ra_obj.segment_seq[0] == 'ln_s_2' and ra_obj.segment_seq[-1] == 'ln_w_-1':
                    conflict_pt = utils.get_conflict_points(ag_obj.segment_seq, ag_obj.signal, ['ln_s_1', 'prep-turn_s', 'exec-turn_s', 'ln_w_-1'], ra_obj.signal)
                else:
                    conflict_pt = utils.get_conflict_points(ag_obj.segment_seq, ag_obj.signal, ra_obj.segment_seq, ra_obj.signal)
                if conflict_pt is not None:
                    ''' agents has not crossed the conflict point'''
                    if (utils.zero_pi_angle_bet_lines([(ra_obj.x,ra_obj.y),((ra_obj.nose_x,ra_obj.nose_y))], [(ra_obj.x,ra_obj.y),conflict_pt]) < 90 or math.hypot(ra_obj.x-conflict_pt[0],ra_obj.y-conflict_pt[1]) <= constants.CAR_LENGTH):
                        if ra_obj.long_acc != 0:
                            if ra_obj.speed**2 - (4*ra_obj.long_acc/2*(-math.hypot(ra_obj.x-conflict_pt[0],ra_obj.y-conflict_pt[1]))) < 0:
                                ra_time_to_collision_region = np.inf
                            else:
                                t = utils.solve_quadratic(ra_obj.long_acc/2, ra_obj.speed, -math.hypot(ra_obj.x-conflict_pt[0],ra_obj.y-conflict_pt[1]))
                                ra_time_to_collision_region = max(t[0],t[1])
                        else:
                            t = math.hypot(ra_obj.x-conflict_pt[0],ra_obj.y-conflict_pt[1]) / ra_obj.speed if ra_obj.speed != 0 else np.inf
                            ra_time_to_collision_region = t
                        
                    else:
                        ra_time_to_collision_region = np.inf
                else:
                    ra_time_to_collision_region = np.inf
            else:
                ttc =  np.inf
        if ttc is None:
            if ag_time_to_collision_region == np.inf or ra_time_to_collision_region == np.inf:
                ttc = np.inf
            else:
                if ag_time_to_collision_region is None or ra_time_to_collision_region is None:
                    brk = 1
                if abs(ag_time_to_collision_region-ra_time_to_collision_region) < 2:
                    ttc = min(ag_time_to_collision_region,ra_time_to_collision_region)
                else:
                    ttc = np.inf
    return ttc    
    

def assign_baseline_utils(strat,ag_obj_list,eq_context):
    util_list = None
    for idx,act in enumerate(strat):
        this_ag_utils = []
        l1_act_code = int(strat[idx][9:11])
        l2_act_code = int(strat[idx][11:13])
        l1_act = utils.get_l1_action_string(l1_act_code)
        l2_act = utils.get_l2_action_string(l2_act_code)
        
        ''' vehicle inhibitory '''
        all_time_gaps = []
        if l1_act in constants.PROCEED_ACTIONS:
            for idx2,act2 in enumerate(strat):
                if idx != idx2:
                    ra_l1_act_code = int(strat[idx2][9:11])
                    ra_l1_act = utils.get_l1_action_string(ra_l1_act_code)
                    if ra_l1_act in constants.PROCEED_ACTIONS:
                        tg = calc_time_gap(ag_obj_list[idx], ag_obj_list[idx2])
                        all_time_gaps.append(tg)
        time_gap = min(all_time_gaps) if len(all_time_gaps) > 0 else np.inf
        veh_inh_util = dist_payoffs(time_gap, (constants.DIST_COST_MEAN,constants.DIST_COST_SD))
        ''' vehicle excitatory '''
        if l1_act in constants.PROCEED_ACTIONS:
            veh_exc_util = 1.0
        else:
            veh_exc_util = 0.0
        ''' pedestrian inhibitory '''
        c_eval = CostEvaluation(eq_context)
        ped_inh_util = c_eval.eval_pedestrian_inh_by_action(strat[idx], (ag_obj_list[idx].x,ag_obj_list[idx].y))
        if util_list is None:
            util_list = np.asarray([veh_inh_util,veh_exc_util,ped_inh_util]).reshape((3,1)) 
        else:
            util_list = np.append(util_list, np.asarray([veh_inh_util,veh_exc_util,ped_inh_util]).reshape((3,1)), axis = 1)
        
    return util_list



class CostEvaluation():
    
    def __init__(self,eq_context):
        self.eq_context = eq_context
        
    def get_dist_gap_params(self,codes):
        codes = tuple(codes)
        act_code_s,act_code_r = codes[0],codes[1]
        if (act_code_s in [1,2,11] and act_code_r in [3,4]) or (act_code_r in [1,2,11] and act_code_s in [3,4]):
            ''' 3.5-9 unsafe and safe'''
            return (6.25,1)
        elif act_code_s in [5,8]:
            ''' 5-10 unsafe and safe'''
            return (7.5,1)
        else:
            ''' 5-15 unsafe and safe'''
            return (10,2.2)
    
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
        #rg_visualizer.plot_payoff_grid(all_possible_payoffs_vals)
        #print('calculating l3 equilibria_core')
        eq = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(all_possible_payoffs)
        br = equilibria_core.calc_best_response(all_possible_payoffs)
        ns_dicts = {'max':{tuple(payoff_stats_trajectories[0,:]) : payoff_stats[0,:] },
                    'min':{tuple(payoff_stats_trajectories[1,:]) : payoff_stats[1,:] },
                    'mean':{tuple(payoff_stats_trajectories[2,:]) : payoff_stats[2,:] },
                    'sd':{tuple(payoff_stats_trajectories[3,:]) : payoff_stats[3,:] }
                    }
        
        return ns_dicts,eq,br     
    
    ''' same as eval_inhibitory; just that the min dist matrix is alredy provided'''
    def eval_inhibitory_from_distmat(self,traj_dict_list, all_possible_payoffs, strategy_tuple, dist_mat):
        disp_arr_x,disp_arr_y = [],[] 
        num_agents = len(strategy_tuple)
        act_code_m = np.empty(shape=(num_agents,num_agents),dtype=(int,2))
        for i in np.arange(num_agents):
            for j in np.arange(num_agents):
                act_code_m[i,j] = (int(strategy_tuple[i].split('-')[0][-4:-2]), int(strategy_tuple[j].split('-')[0][-4:-2]))
        all_possible_payoffs_inh = dict(all_possible_payoffs)
        for traj_idx_tuple in all_possible_payoffs.keys():
            distgap_parm_matrix = np.apply_along_axis(self.get_dist_gap_params,axis=2,arr=act_code_m)
            payoff_m = exp_dist_payoffs(dist_mat, distgap_parm_matrix)
            #combined_payoffs = dist_payoffs(dist) + constants.L2_ACTION_PAYOFF_ADDITIVE
            combined_payoffs = np.amin(payoff_m,axis=1)
            all_possible_payoffs_inh[traj_idx_tuple] = combined_payoffs
        return all_possible_payoffs_inh
    
    ''' given a strategy combination, evaluate the vector of combined_payoffs for each agent.'''
    def eval_inhibitory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        disp_arr_x,disp_arr_y = [],[] 
        num_agents = len(strategy_tuple)
        act_code_m = np.empty(shape=(num_agents,num_agents),dtype=(int,2))
        for i in np.arange(num_agents):
            for j in np.arange(num_agents):
                act_code_m[i,j] = (int(strategy_tuple[i].split('-')[0][-4:-2]), int(strategy_tuple[j].split('-')[0][-4:-2]))
        all_possible_payoffs_inh = dict(all_possible_payoffs)
        for traj_idx_tuple in all_possible_payoffs.keys():
            ''' pair-wise min distance matrix '''
            dist_among_agents = np.full(shape=(num_agents,num_agents),fill_value=np.inf)
            dist = np.full(shape=(num_agents),fill_value=np.inf)
            for i in np.arange(num_agents):
                for j in np.arange(i,num_agents):
                    if i != j:
                        s_x,r_x = traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[j][traj_idx_tuple[j]][:,2]
                        ''' for now calculate the plan combined_payoffs instead of combined_payoffs for the next 1 second '''
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
                        
                        ag_1_act,ag_2_act = utils.get_l1_action_string(act_code_m[i,j][0]), utils.get_l1_action_string(act_code_m[i,j][0])
                        if ag_1_act in constants.WAIT_ACTIONS and ag_2_act in constants.WAIT_ACTIONS:
                            dist_among_agents[i,j] = np.inf
                        elif ag_1_act not in constants.WAIT_ACTIONS and ag_2_act not in constants.WAIT_ACTIONS:
                            dist_among_agents[i,j] = dist_among_agents[i,j]
                        else:
                            if ag_1_act in constants.WAIT_ACTIONS:
                                dist_among_agents[i,j] = np.inf
                            else:
                                dist_among_agents[i,j] = 10
                                
            ''' to be safe, make the matrix symmetric '''
            dist_among_agents = np.minimum(dist_among_agents,dist_among_agents.T)
            ''' find the minimum distance for a vehicle action given all other agent actions '''
            distgap_parm_matrix = np.apply_along_axis(self.get_dist_gap_params,axis=2,arr=act_code_m)
            payoff_m = exp_dist_payoffs(dist_among_agents, distgap_parm_matrix)
            #combined_payoffs = dist_payoffs(dist) + constants.L2_ACTION_PAYOFF_ADDITIVE
            combined_payoffs = np.amin(payoff_m,axis=1)
            all_possible_payoffs_inh[traj_idx_tuple] = combined_payoffs
        return all_possible_payoffs_inh
        
    def eval_excitatory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        num_agents = len(strategy_tuple)
        payoff_stats_trajectories = np.full(shape=(4,num_agents),fill_value=np.inf)
        all_possible_payoffs_exc = dict(all_possible_payoffs)
        payoff_stats = np.full(shape=(4,num_agents),fill_value=np.inf)
        for traj_idx_tuple in all_possible_payoffs.keys():
            combined_payoffs = np.full(shape=(num_agents,),fill_value=np.inf)
            for i in np.arange(num_agents):
                s_V = traj_dict_list[i][traj_idx_tuple[i]][:,5]
                s_traj = list(zip(traj_dict_list[i][traj_idx_tuple[i]][:,2],traj_dict_list[i][traj_idx_tuple[i]][:,3]))
                traj_len = utils.calc_traj_len(s_traj)
                
                ''' for now calculate the plan combined_payoffs '''
                combined_payoffs[i] = progress_payoffs_dist(traj_len)
            all_possible_payoffs_exc[traj_idx_tuple] = combined_payoffs
        return all_possible_payoffs_exc
    
    
    def eval_pedestrian_inh_by_action(self,action,ag_pos):
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
                            if ag_pos is not None:
                                if l1_act_code == 11:
                                    payoff = payoff*1
                                else:
                                    dist_to_ped = math.hypot(ped_state.x-ag_pos[0],ped_state.y-ag_pos[1])
                                    if dist_to_ped <= 2*constants.PEDESTRIAN_CROSSWALK_DIST_THRESH:
                                        payoff = payoff*0
                                    else:
                                        payoff = min(payoff,(dist_to_ped-20)/80,1)
                            else:
                                payoff = payoff*1 if l1_act_code == 11 else payoff*0
                        elif ped_state.crosswalks[xwalk]['location'] == constants.BEFORE_CROSSWALK \
                                and ped_state.crosswalks[xwalk]['dist_to_entry'] < constants.PEDESTRIAN_CROSSWALK_DIST_THRESH \
                                    and ((ped_state.crosswalks[xwalk]['next_change'][1] == 'G' and ped_state.crosswalks[xwalk]['next_change'][0] <= constants.PEDESTRIAN_CROSSWALK_TIME_THRESH) \
                                             or ped_state.crosswalks[xwalk]['next_change'][1] == 'R'):
                            ''' before the crosswalk, within the distance threshold, and the signal is green or about to change to green'''
                            if ag_pos is not None:
                                if l1_act_code == 11:
                                    payoff = payoff*1
                                else:
                                    dist_to_ped = math.hypot(ped_state.x-ag_pos[0],ped_state.y-ag_pos[1])
                                    if dist_to_ped <= 2*constants.PEDESTRIAN_CROSSWALK_DIST_THRESH:
                                        payoff = payoff*0
                                    else:
                                        payoff = min(payoff,(dist_to_ped-20)/80,1)
                            else:
                                payoff = payoff*1 if l1_act_code == 11 else payoff*0
                        else:
                            payoff = payoff*1
        return payoff if payoff == 1 else -1+payoff
       
    def eval_pedestrian_inhibitory(self,traj_dict_list, all_possible_payoffs, strategy_tuple):
        payoff_vect = np.full(shape=(len(strategy_tuple),),fill_value=0)
        all_possible_payoffs_inh_ped = dict(all_possible_payoffs)
        for idx,action in enumerate(strategy_tuple):
            ag_pos = (traj_dict_list[idx][int(action.split('-')[1])][0,2], traj_dict_list[idx][int(action.split('-')[1])][0,3])
            payoff = self.eval_pedestrian_inh_by_action(action,ag_pos)
            payoff_vect[idx] = payoff
        for traj_idx_tuple in all_possible_payoffs.keys():
            all_possible_payoffs_inh_ped[traj_idx_tuple] = payoff_vect
        return all_possible_payoffs_inh_ped
       
    def calc_l3_payoffs(self,eq,strategy_tuple,l3_utility_dict=None):
        if (eq.eval_config.l3_eq is not None and eq.eval_config.l3_eq != 'SAMPLING_EQ') and l3_utility_dict is None:
            raise ValueError('L3 utility dict cannot be None when L3 Equilibria is set to None')
    
        if eq.eval_config.l3_eq is None or eq.eval_config.l3_eq == 'SAMPLING_EQ':
            ''' combined_payoffs will be calculated just from the baselines '''
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
            all_possible_payoffs[k] = np.vstack((all_possible_payoffs_inh[k],all_possible_payoffs_exc[k],all_possible_payoffs_inh_ped[k])) 
        return all_possible_payoffs
    
    
    def calc_l3_sampling_eq_payoffs(self,eq,strategy_tuple,l3_utility_dict=None):
        num_agents = eq.eval_config.num_agents
        if (eq.eval_config.l3_eq is not None and eq.eval_config.l3_eq != 'SAMPLING_EQ') and l3_utility_dict is None:
            raise ValueError('L3 utility dict cannot be None when L3 Equilibria is set to None')
        ag_indexes = dict()
        if eq.eval_config.l3_eq is None or eq.eval_config.l3_eq == 'SAMPLING_EQ':
            ''' combined_payoffs will be calculated just from the baselines '''
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
        
        this_time = self.eq_context.curr_time
        ''' time: {(ag,ra) : [all_pts]} '''
        position_distr = OrderedDict()
        for idx,s in enumerate(strategy_tuple):
            ag_key = (int(s[3:6]),int(s[6:9]))
            ag_indexes[ag_key] = idx
            act = utils.get_l1_action_string(int(s[9:11]))
            traj_id = int(s.split('-')[1])
            traj = traj_dict_list[idx][traj_id]
            #plt.plot(traj[:,2],traj[:,3])
            step_dist = []
            for t_steps in np.arange(0,(constants.PLAN_HORIZON_SECS+1)*(1/constants.LP_FREQ),1/constants.LP_FREQ):
                if int(round(t_steps/10)) not in position_distr:
                    position_distr[int(round(t_steps/10))] = dict()
                if t_steps > 0:
                    if (round(t_steps/10),act) in self.eq_context.eval_config.traj_errs:
                        traj_err = self.eq_context.eval_config.traj_errs[(round(t_steps/10),act)]
                    elif (round(t_steps/10)-.1,act) in self.eq_context.eval_config.traj_errs:
                        traj_err = self.eq_context.eval_config.traj_errs[(round(t_steps/10)-.1,act)]
                    else:
                        traj_err = (0,0,0)
                    d = np.diff(traj[:int(t_steps),2:4], axis=0)
                    segdists = np.sqrt((d ** 2).sum(axis=1))
                    ''' cap the max trajectory error to 10 meters'''
                    try:
                        step_dist.append((max(sum(segdists)-min(10,traj_err[1]),0),sum(segdists)+min(10,traj_err[1])))
                    except IndexError:
                        f=1
                    if t_steps > traj.shape[0]:
                        break
                    f=1
                else:
                    step_dist.append((0,0))
            
            cls = self.eq_context.eval_config.centerline_defs[this_time][ag_key]
            for cl in cls:
                dists_added = []
                cl_x,cl_y = cl[0][0],cl[0][1]
                p_line_lat1, p_line_lat2 = utils.add_parallel(list(zip(cl_x,cl_y)), 1, 1)
                for time_idx,dist in enumerate(step_dist):
                    if ag_key not in position_distr[time_idx]:
                        position_distr[time_idx][ag_key] = []
                    for l in [p_line_lat1,list(zip(cl_x,cl_y)),p_line_lat2]:
                        if dist[0] not in dists_added:
                            if dist[0] > 0:
                                point = utils.find_point_along_line([x[0] for x in l],[x[1] for x in l], None, dist[0])
                            else:
                                point = tuple(l[0])
                            position_distr[time_idx][ag_key].append(point)
                        #plt.plot([point[0]],[point[1]],'o')
                        if dist[1] not in dists_added:
                            if dist[1] > 0:
                                point = utils.find_point_along_line([x[0] for x in l],[x[1] for x in l], None, dist[1])
                            else:
                                point = tuple(l[0])
                            position_distr[time_idx][ag_key].append(point)
                        #plt.plot([point[0]],[point[1]],'o')
                #plt.show() 
        wass_dist_dict = {k:None for k,v in position_distr.items()}
        for k,v in position_distr.items():
            #plt.title(k)
            #ax = plt.gca()
            #rg_visualizer.rg_visualizer.plot_traffic_regions()
            for k1,v1 in v.items():
                mean, cov = stat_utils.get_distribution_params(v1)
                position_distr[k][k1] = (mean,cov)
                #plt.plot([x[0] for x in v1],[x[1] for x in v1],'o')
                #maj_axs, min_axs, phi_hat, C = stat_utils.construct_min_bounding_ellipse(v1)
                #el = Ellipse(xy=C, width=maj_axs, height=min_axs, angle=np.rad2deg(phi_hat), fill=False)
                #ax.add_patch(el)
            #plt.show()
            dist_mat = np.full((num_agents,num_agents), np.inf)
            for ag_key_1 in v.keys():
                for ag_key_2 in v.keys():
                    if ag_key_1 != ag_key_2:
                        ag_1_act = utils.get_l1_action_string(int(strategy_tuple[ag_indexes[ag_key_1]][9:11]))
                        ag_2_act = utils.get_l1_action_string(int(strategy_tuple[ag_indexes[ag_key_2]][9:11]))
                        if ag_1_act in constants.WAIT_ACTIONS and ag_2_act in constants.WAIT_ACTIONS:
                            dist_mat[ag_indexes[ag_key_1],ag_indexes[ag_key_2]] = np.inf
                        elif ag_1_act not in constants.WAIT_ACTIONS and ag_2_act not in constants.WAIT_ACTIONS:
                            wass_dist = stat_utils.calc_wasserstein_distance_multivariate(v[ag_key_1], v[ag_key_2])
                            dist_mat[ag_indexes[ag_key_1],ag_indexes[ag_key_2]] = wass_dist
                        else:
                            if ag_1_act in constants.WAIT_ACTIONS:
                                dist_mat[ag_indexes[ag_key_1],ag_indexes[ag_key_2]] = np.inf
                            else:
                                dist_mat[ag_indexes[ag_key_1],ag_indexes[ag_key_2]] = 10
                    else:
                        dist_mat[ag_indexes[ag_key_1],ag_indexes[ag_key_2]] = np.inf
            wass_dist_dict[k] = dist_mat
        wass_dist_dict = np.stack(wass_dist_dict.values(), axis = 2)
        min_wass_dist = np.amin(wass_dist_dict, axis=(2,1))
        all_possible_payoffs_inh1 = self.eval_inhibitory_from_distmat(traj_dict_list, all_possible_payoffs, strategy_tuple, min_wass_dist)
        f=1
        
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
            all_possible_payoffs_inh = all_possible_payoffs_inh1
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
            all_possible_payoffs[k] = np.vstack((all_possible_payoffs_inh[k],all_possible_payoffs_exc[k],all_possible_payoffs_inh_ped[k])) 
        return all_possible_payoffs
    
    
    
    
    def calc_maxmin_payoff(self,s_traj,r_traj,s_act_key,r_act_key):
        #start_time =time.time()
        s_code = int(s_act_key[3:6])
        if constants.INHIBITORY and r_traj is not None:
            s_act_code,r_act_code = int(s_act_key.split('-')[0][-4:-2]),int(r_act_key.split('-')[0][-4:-2])
            
            distgap_parms = self.get_dist_gap_params((s_act_code, r_act_code))
            slice_len = int(min(5*constants.PLAN_FREQ/constants.LP_FREQ,s_traj.shape[0],r_traj.shape[0]))
            s_traj,r_traj = s_traj[:slice_len],r_traj[:slice_len]
            s_x,s_y = s_traj[:,1], s_traj[:,2]
            r_x,r_y = r_traj[:,1], r_traj[:,2]
            ''' original trajectory is at 10hz. sample at 1hz for speedup'''
            if len(s_x) > 3:
                s_x,r_x,s_y,r_y = s_x[0::9],r_x[0::9],s_y[0::9],r_y[0::9]
            _d = np.hypot(s_x-r_x,s_y-r_y)
            min_dist = np.amin(_d)
            ag_1_act,ag_2_act = utils.get_l1_action_string(s_act_code), utils.get_l1_action_string(r_act_code)
            if ag_1_act in constants.WAIT_ACTIONS and ag_2_act in constants.WAIT_ACTIONS:
                min_dist = np.inf
            elif ag_1_act not in constants.WAIT_ACTIONS and ag_2_act not in constants.WAIT_ACTIONS:
                min_dist = min_dist
            else:
                if ag_1_act in constants.WAIT_ACTIONS:
                    min_dist = np.inf
                else:
                    min_dist = 10
            inh_payoff = exp_dist_payoffs(min_dist,distgap_parms)
        else:
            inh_payoff = 1
        #end_time = time.time()
        #exec_time = str((end_time-start_time))
        #log.info('inhibitory '+exec_time+'s')
        #start_time =time.time()
        if constants.EXCITATORY:
            if self.eq_context.eval_config.l3_eq == 'GAUSSIAN':
                if self.exc_payoff is None:
                    traj_len = utils.calc_traj_len(s_traj)
                    exc_payoff = progress_payoffs_dist(traj_len)
                    self.exc_payoff = exc_payoff
                else:
                    exc_payoff = self.exc_payoff
            else:
                traj_len = utils.calc_traj_len(s_traj)
                exc_payoff = progress_payoffs_dist(traj_len)
        #end_time = time.time()
        #exec_time = str((end_time-start_time))
        #log.info('excitatory '+exec_time+'s')
        #start_time =time.time()
        if constants.INHIBITORY_PEDESTRIAN:
            if self.pedest_payoff is None:
                ag_pos = (s_traj[0,1],s_traj[0,2]) if s_traj is not None and len(s_traj) > 0 else None
                ped_inh_payoff = self.eval_pedestrian_inh_by_action(s_act_key,ag_pos)
                self.pedest_payoff = ped_inh_payoff
            else:
                ped_inh_payoff = self.pedest_payoff
        #end_time = time.time()
        #exec_time = str((end_time-start_time))
        #log.info('pedestrian '+exec_time+'s')
        weights = self.eq_context.weights_info[s_code]
        final_payoff = ( (1-constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT) * ((constants.INHIBITORY_PAYOFF_WEIGHT*inh_payoff) + (constants.EXCITATORY_PAYOFF_WEIGHT*exc_payoff)) ) + \
                            (constants.INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT*ped_inh_payoff)
        #final_payoff = weights[0]*inh_payoff + weights[1]*exc_payoff + weights[2]*ped_inh_payoff 
        return final_payoff

def eval_trajectory_viability(traj_id_list):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+constants.CURRENT_FILE_ID+'\\uni_weber_generated_trajectories_'+constants.CURRENT_FILE_ID+'.db')
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
    


def dist_payoffs(dist_arr,params):
    return scipy.special.erf((dist_arr - params[0]) / (params[1] * math.sqrt(2)))

def exp_dist_payoffs(dist_arr,params):
    if not isinstance(params, np.ndarray):
        return scipy.special.erf((dist_arr - params[0]) / (params[1] * 2))
    else:
        return scipy.special.erf((dist_arr - params[:,:,0]) / (params[:,:,1] * 2))

def progress_payoffs_velocity(dist_arr):
    return scipy.special.erf((dist_arr - constants.SPEED_COST_MEAN) / (constants.SPEED_COST_SD * math.sqrt(2)))

def progress_payoffs_dist(traj_len):
    if traj_len >= 100:
        return 1
    else:
        return traj_len/100

    

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
    unreadable_str = constants.CURRENT_FILE_ID+agent+relev_agent+l1_action+l2_action
    return unreadable_str

