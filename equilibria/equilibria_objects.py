'''
Created on Apr 22, 2020

@author: authorA
'''
import ast
import itertools
import sqlite3
import sys

from constants import *
from equilibria.cost_evaluation import CostEvaluation
from equilibria.equilibria_core import EquilibriaCore
import equilibria.equilibria_core as eq_core
import numpy as np
from planning_objects import *
from all_utils import db_utils
from all_utils import utils
from all_utils.thread_utils import DictProcessor
import copy
import math
import datetime
import time

log = constants.eq_logger


class L3Calculation:

    def __init__(self,maxmin_traj_info,eq_context,l1l2_strat,l3_eq_type):
        self.maxmin_traj_info = maxmin_traj_info
        self.eq_context = eq_context
        self.l1l2_strat = l1l2_strat
        self.l3_eq_type = l3_eq_type
        self.l3_key = 'min_pay' if l3_eq_type == 'MAXMIN' else 'max_pay'
        
        
    def load_trajs(self):
        for ag_key,ag_acts in self.maxmin_traj_info.items():
            traj_info_id = int(ag_key.split('-')[1])
            for traj_id,dets in ag_acts.items():
                traj = self.eq_context.l3_trajectory_cache[traj_info_id][traj_id]
                t_x,s_x,s_y = traj[:,1],traj[:,2],traj[:,3]
                self.maxmin_traj_info[ag_key][traj_id]['trajectory'] = np.asarray(list(zip(t_x,s_x,s_y))).reshape((len(t_x),3))
    
    def calc_maxmin_eq(self):
        num_agents = self.eq_context.eval_config.num_agents
        order_map = {x.split('-')[0]:o_i for o_i,x in enumerate(self.l1l2_strat)}
        self.load_trajs()
        cost_eval = CostEvaluation(self.eq_context)
        cost_eval.pedest_payoff = None
        cost_eval.exc_payoff = None
        if num_agents > 1:
            for ag_key,ag_acts in self.maxmin_traj_info.items():
                for traj_id,dets in ag_acts.items():
                    for oth_ag_key,oth_ag_acts in self.maxmin_traj_info.items():
                        if oth_ag_key != ag_key:
                            for oth_traj_id,oth_dets in oth_ag_acts.items():
                                payoff = cost_eval.calc_maxmin_payoff(dets['trajectory'], oth_dets['trajectory'], ag_key, oth_ag_key)
                                self.maxmin_traj_info[ag_key][traj_id][self.l3_key].append(payoff)
        else:
            for ag_key,ag_acts in self.maxmin_traj_info.items():
                for traj_id,dets in ag_acts.items():
                    payoff = cost_eval.calc_maxmin_payoff(dets['trajectory'], None, ag_key, None)
                    self.maxmin_traj_info[ag_key][traj_id][self.l3_key].append(payoff)
        
        all_sv_act_payoff = []
        all_agents_act_payoff = []
        for ag_key,ag_acts in self.maxmin_traj_info.items():
            ag_pay_dets = []
            for traj_id,dets in ag_acts.items():
                if self.l3_eq_type == 'MAXMIN':
                    dets[self.l3_key] = min(dets[self.l3_key])
                else:
                    dets[self.l3_key] = max(dets[self.l3_key])
                if int(ag_key[6:9]) == 0:
                    all_sv_act_payoff.append((ag_key.split('-')[0]+'-'+str(traj_id), dets[self.l3_key]))
                ag_pay_dets.append((ag_key+'-'+str(traj_id), dets[self.l3_key]))
            ag_pay_dets.sort(key=lambda x: x[1], reverse=True)
            all_agents_act_payoff.append(ag_pay_dets)
        l3_eq_act,l3_eq_pay = [list()]*num_agents,[list()]*num_agents
        for act_p in all_agents_act_payoff:
            idx = order_map[act_p[0][0].split('-')[0]]
            l3_eq_act[idx] = act_p[0][0].split('-')[0]+'-'+act_p[0][0].split('-')[2]
            l3_eq_pay[idx] = act_p[0][1]
        l3_eq = {tuple(l3_eq_act):l3_eq_pay}
        l3_eq_obj = self.eq_context.L3Equilibria(self.l1l2_strat)
        l3_eq_obj.equilibria_actions = l3_eq
        l3_eq_obj.eq_payoff = [l3_eq_pay]
        l3_eq_obj.sv_acts = [x[0] for x in all_sv_act_payoff]
        l3_eq_obj.all_traj_act_payoffs = [[x[1] for x in all_sv_act_payoff]]
        return l3_eq_obj
        
                
        
                            




class L1L2UtilityProcessor:
    
    def __init__(self,l1l2_utility_dict,sv_det,time_ts,eq_obj):
        self.l1l2_utility_dict = l1l2_utility_dict
        self.l1l2_payoffdict = dict()
        self.sv_det = sv_det
        self.time_ts = time_ts
        self.all_l3_eq = dict()
        self.eq_context = eq_obj
    
    def build_l3_utility_table(self,sv_det,time_ts,l1l2_utility_dict_chunk):
        '''
        1. for each trajectory info id get the trajectory ids from baseline and boundary tables
        2. those are the actions
        3. create a power set of those actions
        '''
        l3_utility_dict = dict()
        sv_traj_action_dict = dict()
        for k,v in l1l2_utility_dict_chunk.items():
            l3_utility_dict[k] = dict()
            num_agents = len(k)
            order_map = {x.split('-')[0]:o_i for o_i,x in enumerate(k)}
            traj_info_ids = [(x.split('-')[0],int(x.split('-')[1])) for x in k]
            ids_to_fetch_from_db = []
            all_traj_acts = []
            for ag_key,traj_info_id in traj_info_ids:
                if traj_info_id in self.eq_context.l3_trajectory_cache:
                    all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.eq_context.l3_trajectory_cache[traj_info_id].keys())])
                else:
                    ids_to_fetch_from_db.append((ag_key,traj_info_id))
            if len(ids_to_fetch_from_db) > 0:
                self.eq_context.l3_trajectory_cache.update(utils.load_trajs_for_traj_info_id([x[1] for x in ids_to_fetch_from_db],False,self.eq_context.eval_config.traj_type))
            for ag_key,traj_info_id in ids_to_fetch_from_db:
                all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.eq_context.l3_trajectory_cache[traj_info_id].keys())])
            sv_traj_actions = []
            ordered_traj_acts = [list()]*len(all_traj_acts)
            for traj_acts in all_traj_acts:
                if len(traj_acts) > 0:
                    if traj_acts[0][6:9] == '000':
                        sv_traj_actions = traj_acts
                    ordered_traj_acts[order_map[traj_acts[0].split('-')[0]]] = traj_acts
                
            all_action_combinations = list(itertools.product(*[v for v in ordered_traj_acts]))
            utility_dict = {k:np.zeros(shape=len(k)) for k in all_action_combinations}
            l3_utility_dict[k] = utility_dict
            sv_traj_action_dict[k] = sv_traj_actions
        return l3_utility_dict,sv_traj_action_dict
    
    def build_l3_action_dict(self,sv_det,time_ts,l1l2_utility_dict_chunk):
        l3_traj_action_dict = dict()
        sv_traj_action_dict = dict()
        l3_key = 'min_pay' if self.eq_context.eval_config.l3_eq == 'MAXMIN' else 'max_pay'
        for k,v in l1l2_utility_dict_chunk.items():
            l3_traj_action_dict[k] = dict()
            order_map = {x.split('-')[0]:o_i for o_i,x in enumerate(k)}
            traj_info_ids = [(x.split('-')[0],int(x.split('-')[1])) for x in k]
            ids_to_fetch_from_db = []
            all_traj_acts = []
            for ag_key,traj_info_id in traj_info_ids:
                if traj_info_id in self.eq_context.l3_trajectory_cache:
                    all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.eq_context.l3_trajectory_cache[traj_info_id].keys())])
                else:
                    ids_to_fetch_from_db.append((ag_key,traj_info_id))
                l3_traj_action_dict[k][ag_key+'-'+str(traj_info_id)] = dict()
            if len(ids_to_fetch_from_db) > 0:
                self.eq_context.l3_trajectory_cache.update(utils.load_trajs_for_traj_info_id([x[1] for x in ids_to_fetch_from_db],False,self.eq_context.eval_config.traj_type))
            for ag_key,traj_info_id in ids_to_fetch_from_db:
                all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.eq_context.l3_trajectory_cache[traj_info_id].keys())])
            sv_traj_actions = []
            ordered_traj_acts = [list()]*len(all_traj_acts)
            for traj_acts in all_traj_acts:
                if len(traj_acts) > 0:
                    if traj_acts[0][6:9] == '000':
                        sv_traj_actions = traj_acts
                    ordered_traj_acts[order_map[traj_acts[0].split('-')[0]]] = traj_acts
            for o_acts in ordered_traj_acts:
                for o_act in o_acts:
                    ag_key = o_act.split('-')[0]
                    traj_id = int(o_act.split('-')[1])
                    for ag_key_info_id,akii_det in l3_traj_action_dict[k].items():
                        if ag_key_info_id.split('-')[0] == ag_key:
                            l3_traj_action_dict[k][ag_key_info_id][traj_id] = dict()
                            l3_traj_action_dict[k][ag_key_info_id][traj_id][l3_key] = []
                            l3_traj_action_dict[k][ag_key_info_id][traj_id]['trajectory'] = []
            sv_traj_action_dict[k] = sv_traj_actions
        return l3_traj_action_dict,sv_traj_action_dict
        
    def process(self):
        if self.eq_context.eval_config.l3_eq == 'MAXMIN' or self.eq_context.eval_config.l3_eq == 'BR':
            l3_utility_dict_chunk,sv_traj_action_dict = self.build_l3_action_dict(self.sv_det,self.time_ts,self.l1l2_utility_dict)
        else:
            l3_utility_dict_chunk,sv_traj_action_dict = self.build_l3_utility_table(self.sv_det,self.time_ts,self.l1l2_utility_dict)
        ct_2,chunk_N = 0, len(l3_utility_dict_chunk)
        
        for l1l2_strat,l3_utility_dict in l3_utility_dict_chunk.items():
            ct_2 += 1
            #if self.eq_context.eval_config.l3_eq == 'MAXMIN' or self.eq_context.eval_config.l3_eq == 'BR':
            maxmin_obj = L3Calculation(l3_utility_dict,self.eq_context,l1l2_strat,self.eq_context.eval_config.l3_eq)
            start_time = time.time()
            l3_eq_obj = maxmin_obj.calc_maxmin_eq()
            end_time = time.time()
            exec_time = str((end_time-start_time))
            N_size = (len(l3_utility_dict)-1)*81*len(l3_utility_dict)
            log.info(str(self.run_idx)+':'+str(ct_2)+'/'+str(chunk_N)+' DONE calculating '+self.eq_context.eval_config.l3_eq+' strategy for L3 actions in '+exec_time+'s'+' N='+str(N_size))
            #log.info(str(ct_2)+'/'+str(chunk_N)+' DONE calculating '+self.eq_context.eval_config.l3_eq+' strategy for L3 actions in '+exec_time+'s'+' N='+str(N_size))
            '''
            else:
                l3_payoff = dict()
                if len(l3_utility_dict) >= 3000 and len(l3_utility_dict) < 100000:
                    start_time_th = time.time()
                    dict_processor = DictProcessor.fromFlattenedDict(l3_utility_dict, multithread_l3_utility_execution, (self.eq_context,l1l2_strat,sv_traj_action_dict), 25000)
                    l3_payoff = dict_processor.execute_threads_callback(multithread_l3_utility_execution_callback, l3_payoff)
                    end_time_th = time.time()
                    exec_time_th = str((end_time_th-start_time_th))
                    log.info('l3_multithreading is True '+str(len(l3_utility_dict)) + ' took '+str(exec_time_th)+'s')
                elif len(l3_utility_dict) >= 100000:
                    start_time_th = time.time()
                    dict_proc = DictProcessor.fromFlattenedDict(l3_utility_dict, multithread_l3_utility_execution, (self.eq_context,l1l2_strat,sv_traj_action_dict), 25000)
                    dict_proc.execute_mp_callback(multithread_l3_utility_execution_callback, l3_payoff)
                    end_time_th = time.time()
                    exec_time_th = str((end_time_th-start_time_th))
                    log.info('l3_multiprocessing is True '+str(len(l3_utility_dict)) + ' took '+str(exec_time_th)+'s')
                else:
                    l3_processor_obj = L3UtilityProcessor(l3_utility_dict, l1l2_strat, self.eq_context, sv_traj_action_dict)
                    l3_payoff = l3_processor_obj.process()
                N_l3payoff_table = len(l3_payoff)
                start_time = time.time()
                log.info(str(ct_2)+'/'+str(chunk_N)+' calculating '+self.eq_context.eval_config.l3_eq+' strategy for L3 actions of size '+str(N_l3payoff_table))
                eq_core = EquilibriaCore(self.eq_context.eval_config.num_agents,l3_payoff,N_l3payoff_table)
                l3_eq = eq_core.calc_max_min_response()
                
                l3_eq_obj = self.eq_context.L3Equilibria(l1l2_strat)
                l3_eq_obj.equilibria_actions = l3_eq
                l3_eq_obj.eq_payoff = []
                l3_eq_obj.sv_acts = list(sv_traj_action_dict[l1l2_strat])
                l3_eq_obj.all_traj_act_payoffs = []
                for e,p in l3_eq.items():
                    eq_act_tuple = [x if x[6:9]!='000' else None for x in list(e)]
                    l3_eq_obj.eq_payoff.append(p.tolist())
                    sv_act_payoffs = []
                    for sv_act in sv_traj_action_dict[l1l2_strat]:
                        _act_tup = tuple([x if x is not None else sv_act for x in eq_act_tuple])
                        sv_index = _act_tup.index(sv_act)
                        sv_payoff = round(l3_payoff[_act_tup][sv_index],6)
                        sv_act_payoffs.append(sv_payoff)
                    l3_eq_obj.all_traj_act_payoffs.append(sv_act_payoffs)
                readable_eq = utils.print_readable(e)
            '''
            self.all_l3_eq[l1l2_strat] = l3_eq_obj
            
            if len(l3_eq_obj.eq_payoff) > 0:
                self.l1l2_payoffdict[tuple(x.split('-')[0] for x in l1l2_strat)] = l3_eq_obj.eq_payoff[0] 


class L3UtilityProcessor:      
    
    def __init__(self,l3_utility_dict,l1l2_strat,eq_context,sv_traj_action_dict):
        self.l1l2_strat = l1l2_strat
        self.l3_utility_dict = l3_utility_dict
        self.eq_context = eq_context
        self.sv_traj_action_dict = sv_traj_action_dict
    
    def process(self):
        #log.info(' '.join(['processing chunk',str(ct_1),'/',str(chunks_n),' ',str(ct_2),'/',str(chunk_N)]))
        cost_ev = CostEvaluation(self.eq_context)
        l3_payoff = cost_ev.calc_l3_payoffs(self.eq_context,self.l1l2_strat,self.l3_utility_dict)
        return l3_payoff
    
        
            

def multithread_l1l2_utility_execution_callback(l1l2_processor_obj,params):
    all_l3_eq,payoffdict = params[0],params[1]
    all_l3_eq.update(l1l2_processor_obj.all_l3_eq)
    payoffdict.update(l1l2_processor_obj.l1l2_payoffdict)
    return all_l3_eq,payoffdict
    
def multithread_l3_utility_execution_callback(l3_payoff_chunk,l3_payoff):
    for k,v in l3_payoff_chunk.items():
        l3_payoff[k] = v
    return l3_payoff
    

def multithread_l1l2_utility_execution(l1l2_utility_dict_chunk,params):
    sv_det,time_ts,eq_obj = params[0], params[1], params[2]
    l1l2_utility_dict_chunk_copy = copy.deepcopy(l1l2_utility_dict_chunk)
    eq_obj_copy = copy.deepcopy(eq_obj)
    l1l2_processor_obj = L1L2UtilityProcessor(l1l2_utility_dict_chunk_copy,sv_det,time_ts,eq_obj_copy)
    l1l2_processor_obj.run_idx = 0
    l1l2_processor_obj.process()
    return l1l2_processor_obj
   
def multithread_l3_utility_execution(l3_utility_dict_chunk,params,results_dict = None):
    eq_obj,l1l2_strat,sv_traj_action_dict = params[0], params[1], params[2]
    l3_utility_dict_chunk_copy = copy.deepcopy(l3_utility_dict_chunk)
    eq_obj_copy = copy.deepcopy(eq_obj)
    l1l2_strat_copy = copy.copy(l1l2_strat)
    l3_processor_obj = L3UtilityProcessor(l3_utility_dict_chunk_copy, l1l2_strat_copy, eq_obj_copy, sv_traj_action_dict)
    l3_payoff = l3_processor_obj.process()
    if results_dict is not None:
        ''' for multi processing '''
        for k,v in l3_payoff.items():
            results_dict[k] = v
    return l3_payoff
   

class Equilibria:
    
    class L1L2Equilibrium:
        
        def __init__(self):
            self.equilibria_actions = dict()      
            
            
    class L3Equilibria:
        
        def __init__(self,l1l2_strat):
            self.equilibria_actions = dict()
            self.l1l2_strat = l1l2_strat
    
    def __init__(self,eval_config):
        self.eval_config = eval_config
    
        
                
    
    def calc_empirical_actions(self):
        conn = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT * FROM TRAJECTORIES_0"+constants.CURRENT_FILE_ID+"_EXT WHERE L1_ACTION IS NOT NULL"
        c.execute(q_string)
        res = c.fetchall()
        emp_act_in_db = dict()
        for row in res:
            emp_act_in_db[(row[1],row[0])] = ast.literal_eval(row[3])
        q_string = "SELECT * FROM L1_ACTIONS WHERE L1_ACTION IS NOT NULL"
        c.execute(q_string)
        res = c.fetchall()
        for row in res:
            emp_act_in_db[(row[0],row[1])] = ast.literal_eval(row[2])
        conn_trajdb = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_generated_trajectories_'+constants.CURRENT_FILE_ID+'.db')
        c_trajdb = conn_trajdb.cursor()
        self.empirical_actions = dict()
        ct = 0
        N = len(self.eval_config.traj_metadata.items())
        
        for time_ts,ag_act_dict in self.eval_config.traj_metadata.items():
            self.empirical_actions[time_ts] = dict()
            ct += 1
            logging.info(self.eval_config.direction+'    '+str(ct)+'/'+str(N))
            for ag_id,act_dict in ag_act_dict.items():
                agent_acts = dict()
                agent_id = None
                for k,v in act_dict.items():
                    if k!='raw_data' and k!='relev_agents':
                        agent_id = k
                for k,v in act_dict.items():
                    if k == 'raw_data':
                            continue
                    if k == 'relev_agents':
                        for ra,rv in act_dict[k].items():
                            ac_l = []
                            for l1,l2 in rv.items():
                                for l2_a in l2:
                                    act_str = ((l1,l2_a),None)
                                    ac_l.append(act_str)
                            agent_acts[(agent_id,ra)] = ac_l
                    else:
                        ac_l = []
                        for l1,l2 in v.items():
                            for l2_a in l2:
                                act_str = ((l1,l2_a),None)
                                ac_l.append(act_str)
                        agent_acts[(k,0)] = ac_l
                
                for k,v in agent_acts.items():
                    emp_act_db_key = (time_ts,k[1]) if k[1]!=0 else (time_ts,k[0])
                    if emp_act_db_key not in emp_act_in_db:
                        veh_state = VehicleState()
                        id = k[0] if k[1]==0 else k[1]
                        veh_state.set_id(id)
                        agent_track = utils.get_track(veh_state, time_ts, True)
                        if agent_track is not None and len(agent_track) > 1:
                            selected_time_ts = np.arange(0,min(len(agent_track),(DATASET_FPS*PLAN_FREQ)+(DATASET_FPS*LP_FREQ)),DATASET_FPS*LP_FREQ)
                            ts_sampled_track = [agent_track[int(i)] for i in selected_time_ts]
                            sampled_traj = [(round(x[6],3),x[1],x[2]) for x in ts_sampled_track]
                            acts_with_residual = []
                            for acts in v:
                                if utils.truncate(time_ts,1) != utils.truncate(sampled_traj[0][0],1):
                                    ''' the starting timestamps of the two trajectories are different. (The trajectory arrives in the scene later than the current time) '''
                                    emp_traj_start_ts = sampled_traj[0][0]
                                    q_string = "select distinct time from GENERATED_TRAJECTORY_INFO where GENERATED_TRAJECTORY_INFO.AGENT_ID="+str(id)+" or GENERATED_TRAJECTORY_INFO.RELEV_AGENT_ID="+str(id)+" order by time"
                                    c_trajdb.execute(q_string)
                                    available_times = [x[0] for x in c_trajdb.fetchall()]
                                    if emp_traj_start_ts > available_times[-1]:
                                        ''' there is no appropriate baseline track for this agent for the current time'''
                                        baseline_track = None
                                    else:
                                        ''' find the closest baseline track. we want to extrapolate the first observed action of this agent back to the current time.'''
                                        avl_times_rounded = [int(x) for x in available_times]
                                        if int(emp_traj_start_ts) in avl_times_rounded:
                                            baseline_track_time_ref = available_times[avl_times_rounded.index(int(emp_traj_start_ts))]
                                        else:
                                            baseline_track_time_ref = available_times[utils.find_nearest_in_array(available_times, emp_traj_start_ts)]
                                        baseline_track = utils.get_baseline_trajectory(agent_id, k[1], acts[0][0], acts[0][1], baseline_track_time_ref) 
                                else:
                                    baseline_track = utils.get_baseline_trajectory(agent_id, k[1], acts[0][0], acts[0][1], time_ts) 
                                if baseline_track is not None:
                                    baseline_traj = [(x[2],x[3],x[4]) for x in baseline_track]
                                    _traj_diff = utils.calc_traj_diff(sampled_traj, baseline_traj)
                                    acts_with_residual.append((acts[0],_traj_diff))
                                else:
                                    acts_with_residual.append((acts[0],np.inf))
                            agent_acts[k] = sorted(acts_with_residual, key=lambda tup: tup[1])
                        else:
                            brk = 1
                empirical_acts = []
                for k,v in agent_acts.items():
                    emp_act_db_key = (time_ts,k[1]) if k[1]!=0 else (time_ts,k[0])
                    if emp_act_db_key not in emp_act_in_db:
                        _act_str = []
                        for acts in v:
                            if acts[1] == v[0][1]:
                                #_act_str.append(str(k[0]).zfill(3)+'_'+str(k[1]).zfill(3)+'_'+acts[0][0]+'_'+acts[0][1])
                                _act_str.append(CURRENT_FILE_ID.zfill(3)+str(k[0]).zfill(3)+str(k[1]).zfill(3)+str(L1_ACTION_CODES[acts[0][0]]).zfill(2)+str(L2_ACTION_CODES[acts[0][1]]).zfill(2))
                        empirical_acts.append(_act_str)
                        self.empirical_actions[time_ts][k] = _act_str
                    else:
                        self.empirical_actions[time_ts][k] = emp_act_in_db[(time_ts,k[1])] if k[1]!=0 else emp_act_in_db[(time_ts,k[0])]
                
               
        #if len(emp_act_in_db) == 0:
        emp_act_in_db = dict()
        for ts,v in self.empirical_actions.items():
            for k,acts in v.items():
                if k[1] == 0:
                    emp_act_in_db[(ts,k[0])] = acts
                else:
                    emp_act_in_db[(ts,k[1])] = acts
        u_string = "REPLACE INTO L1_ACTIONS VALUES (?,?,?)"
        u_list = []
        for k,v in emp_act_in_db.items():
            u_list.append((k[0],k[1],str(v)))
        c.executemany(u_string,u_list)
        conn.commit()       
        conn.close()
        
                        
    def to_actstring(self,act_str):
        tokens = act_str.split('|')
        assert(len(tokens)==4)
        l1_action = str(constants.L1_ACTION_CODES[tokens[2]]).zfill(2)
        l2_action = str(constants.L2_ACTION_CODES[tokens[3]]).zfill(2)
        agent = str(tokens[0]).zfill(3)
        relev_agent = str(tokens[1]).zfill(3)
        unreadable_str = constants.CURRENT_FILE_ID+agent+relev_agent+l1_action+l2_action
        return unreadable_str
    
    def build_l3_utility_table(self,sv_det,time_ts,l1l2_utility_dict_chunk):
        '''
        1. for each trajectory info id get the trajectory ids from baseline and boundary tables
        2. those are the actions
        3. create a power set of those actions
        '''
        l3_utility_dict = dict()
        sv_traj_action_dict = dict()
        for k,v in l1l2_utility_dict_chunk.items():
            l3_utility_dict[k] = dict()
            order_map = {x.split('-')[0]:o_i for o_i,x in enumerate(k)}
            traj_info_ids = [(x.split('-')[0],int(x.split('-')[1])) for x in k]
            ids_to_fetch_from_db = []
            all_traj_acts = []
            for ag_key,traj_info_id in traj_info_ids:
                if traj_info_id in self.l3_trajectory_cache:
                    all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.l3_trajectory_cache[traj_info_id].keys())])
                else:
                    ids_to_fetch_from_db.append((ag_key,traj_info_id))
            if len(ids_to_fetch_from_db) > 0:
                self.l3_trajectory_cache.update(utils.load_trajs_for_traj_info_id([x[1] for x in ids_to_fetch_from_db],False,self.eval_config.traj_type))
            for ag_key,traj_info_id in ids_to_fetch_from_db:
                all_traj_acts.append([ag_key+'-'+str(x) for x in list(self.l3_trajectory_cache[traj_info_id].keys())])
            sv_traj_actions = []
            ordered_traj_acts = [list()]*len(all_traj_acts)
            for traj_acts in all_traj_acts:
                if len(traj_acts) > 0:
                    if traj_acts[0][6:9] == '000':
                        sv_traj_actions = traj_acts
                    ordered_traj_acts[order_map[traj_acts[0].split('-')[0]]] = traj_acts
                
            all_action_combinations = list(itertools.product(*[v for v in ordered_traj_acts]))
            utility_dict = {k:np.zeros(shape=len(k)) for k in all_action_combinations}
            l3_utility_dict[k] = utility_dict
            sv_traj_action_dict[k] = sv_traj_actions
        return l3_utility_dict,sv_traj_action_dict

                
    def build_l1l2_utility_table(self,sv_det,time_ts):
        logging.info('building payoff dict...')
        all_acts,all_baseline_ids,sv_actions,all_beliefs = [],[],[],[]
        for k,v in sv_det.items():
            if k == 'raw_data':
                for ag_id_str,ag_act_row in v.items():
                    #ag_id,ra_id,l1_act,l2_act = 
                    acts = [self.to_actstring(str(r[2])+'|'+str(r[3])+'|'+r[4]+'|'+r[5])+'-'+str(r[1]) for r in ag_act_row]
                    emp_dict_key = (int(ag_id_str.split('-')[0]), int(ag_id_str.split('-')[1]))
                    #if emp_dict_key in self.empirical_actions:
                    all_acts.append(acts)
                    if 'BELIEF' in self.eval_config.l1_eq:
                        emp_acts = self.empirical_actions[time_ts][emp_dict_key]
                        beliefs = [1/len(emp_acts) if r.split('-')[0] in emp_acts else 0 for r in acts]
                        all_beliefs.append(beliefs)
                    for r in ag_act_row:
                        if r[3] == 0:
                            sv_actions = [r.split('-')[0] for r in acts]
                        all_baseline_ids.append(str(r[1]))
        all_action_combinations = list(itertools.product(*[v for v in all_acts]))
        if 'BELIEF' in self.eval_config.l1_eq:
            all_belief_combinations = list(itertools.product(*[v for v in all_beliefs]))
            belief_dict = {tuple(x.split('-')[0] for x in k):v for k,v in zip(all_action_combinations,all_belief_combinations)}
        utility_dict = {k:np.zeros(shape=len(k)) for k in all_action_combinations}
        self.eval_config.set_num_agents(len(all_action_combinations[0]))
        if self.eval_config.l3_eq is None:
            self.l1l2_trajectory_cache = utils.load_trajs_for_traj_info_id(all_baseline_ids,True,self.eval_config.traj_type)
        logging.info('building payoff dict...DONE')
        if 'BELIEF' in self.eval_config.l1_eq:
            return utility_dict,sv_actions,belief_dict
        else:
            return utility_dict,sv_actions,None
    
    
    def assign_utility_to_table(self,l1l2_utility_dict,processed_dict):
        payoffdict = dict()
        ct = 0
        N = len(l1l2_utility_dict)
        for l1l2_strat,l1l2_utility in l1l2_utility_dict.items():
            ct +=1
            logging.info('processing '+str(ct)+'/'+str(N)+' l1l2')
            cost_ev = CostEvaluation(self)
            l3_payoff = cost_ev.calc_l3_payoffs(self,l1l2_strat)
            assert(len(l3_payoff)==1)
            ''' the l1l2 payoff table is constructed with the payoffs from lower levels '''
            payoffdict[tuple([k.split('-')[0] for k in l1l2_strat])] = next(iter(l3_payoff.values()))
        for k,v in payoffdict.items():
            processed_dict[k] = v
        return payoffdict
        
    def construct_oth_ag_eq_info(self,e,p,time_ts,sv_id,payoffdict):
        oth_ag_eq_info = dict()
        for oth_ag_idx,e_st in enumerate(e):
            if e_st[6:9]!='000':
                oth_ag_id = int(e_st[6:9])
                if (sv_id,oth_ag_id) in self.empirical_actions[time_ts] and len(self.empirical_actions[time_ts][(sv_id,oth_ag_id)]) > 0:
                    oth_ag_emp_act = e_st[:-4]+self.empirical_actions[time_ts][(sv_id,oth_ag_id)][0][-4:]
                else:
                    continue
                oth_ag_eq_act = e_st
                oth_ag_eq_payoff = round(p[oth_ag_idx],6)
                replaced_strat = list(e)
                replaced_strat[oth_ag_idx] = oth_ag_emp_act
                replaced_strat = tuple(replaced_strat)
                oth_ag_emp_payoff = round(payoffdict[replaced_strat][oth_ag_idx],6)
                oth_ag_payoffdiff = round(oth_ag_eq_payoff-oth_ag_emp_payoff,6)
                if oth_ag_id not in oth_ag_eq_info:
                    oth_ag_eq_info[oth_ag_id] = []
                oth_ag_eq_info[oth_ag_id].append((oth_ag_eq_payoff,oth_ag_emp_payoff))
        return oth_ag_eq_info
    
    def calc_l1l2_equilibrium(self):
        ct,N = 0,len(self.eval_config.traj_metadata)
        for time_ts,m_data in self.eval_config.traj_metadata.items():
            ct += 1
            self.curr_time = time_ts
            pedestrian_info = utils.setup_pedestrian_info(time_ts)
            self.eval_config.set_pedestrian_info(pedestrian_info)
            self.l1l2_trajectory_cache = dict()
            self.l3_trajectory_cache = dict()
            equilibra_dict = dict()
            equilibra_dict[time_ts] = dict()
            for sv_id,sv_det in m_data.items():
                if self.eval_config.update_only and time_ts in self.eval_config.eq_acts_in_db:
                    if sv_id in self.eval_config.eq_acts_in_db[time_ts]:
                        log.info("equilibrium already in db....continuing "+str(ct)+"/"+str(N))
                        continue
                if len(sv_det[sv_id]) == 0:
                    log.info("no info on sv....continuing "+str(ct)+"/"+str(N))
                    continue
                equilibra_dict[time_ts][sv_id] = dict()
                l1l2_utility_dict,sv_actions,belief_dict = self.build_l1l2_utility_table(sv_det,time_ts)
                
                logging.info('calculating utilities...')
                if self.eval_config.l3_eq is None:
                    u_ct = 0
                    N_payoff_table = len(l1l2_utility_dict)
                    dict_proc = DictProcessor.fromFlattenedDict(l1l2_utility_dict, self.assign_utility_to_table, {}, 100)
                    payoffdict = dict_proc.execute()
                    #payoffdict = self.assign_utility_to_table(l1l2_utility_dict,{})
                    ''' release this '''    
                    l1l2_utility_dict = None
                    logging.info('calculating utilities...DONE')
                    ''' release this '''    
                    l1l2_utility_dict = None
                    logging.info('calculating utilities...DONE')
                else:
                    u_ct = 0
                    N_payoff_table = len(l1l2_utility_dict)
                    chunk_yield = utils.dict_chunks(l1l2_utility_dict, math.ceil(N_payoff_table/20))
                    l1l2_utility_dict_chunks = [x for x in chunk_yield]
                    all_l3_eq = dict()
                    payoffdict = dict()
                    chunks_n = len(l1l2_utility_dict_chunks)
                    ct_1 = 0
                    if N_payoff_table < 4000:
                        multithread = True
                        dict_processor = DictProcessor(multithread_l1l2_utility_execution,(sv_det,time_ts,self),l1l2_utility_dict_chunks,l1l2_utility_dict)
                        all_l3_eq,payoffdict = dict_processor.execute_threads_callback(multithread_l1l2_utility_execution_callback, (all_l3_eq,payoffdict))
                    else:
                        for ch_idx,l1l2_utility_dict_chunk in enumerate(l1l2_utility_dict_chunks):
                            l1l2_processor_obj = L1L2UtilityProcessor(l1l2_utility_dict_chunk,sv_det,time_ts,self)
                            l1l2_processor_obj.run_idx = ch_idx
                            l1l2_processor_obj.process()
                            all_l3_eq.update(l1l2_processor_obj.all_l3_eq)
                            payoffdict.update(l1l2_processor_obj.l1l2_payoffdict)
                logging.info(self.eval_config.direction+" "+str(time_ts)+"-"+str(sv_id)+":"+str(ct)+'/'+str(N)+":"+str(N_payoff_table))
                all_eq = []
                l1_agent = False
                if self.eval_config.l1_eq == 'NASH':
                    eq_core = EquilibriaCore(self.eval_config.num_agents,payoffdict,N_payoff_table,sv_actions,False)
                    logging.info('calculating equilibria...')
                    eq = eq_core.calc_pure_strategy_nash_equilibrium_exhaustive()
                    logging.info('calculating equilibria...DONE '+str(eq_core.exec_time)+'ms')
                    for e,p in eq.items():
                        l1l2_eq = self.L1L2Equilibrium()
                        l1l2_eq.equilibria_actions = {(int(k[0][3:6]), int(k[0][6:9])):(k[0],k[1]) for k in zip(e,p)}
                        eq_act_tuple = [x if x[6:9]!='000' else None for x in list(e)]
                        sv_act_payoffs = []
                        for sv_act in sv_actions:
                            _act_tup = tuple([x if x is not None else sv_act for x in eq_act_tuple])
                            sv_index = _act_tup.index(sv_act)
                            sv_payoff = round(payoffdict[_act_tup][sv_index],6)
                            sv_act_payoffs.append(sv_payoff)
                        readable_eq = utils.print_readable(e)
                        l1l2_eq.all_act_payoffs = sv_act_payoffs
                        l1l2_eq.oth_ag_eq_info = self.construct_oth_ag_eq_info(e,p,time_ts,sv_id,payoffdict)
                        all_eq.append(l1l2_eq)
                        
                elif self.eval_config.l1_eq == 'BR_TRUE_BELIEF':
                    eq_strats,all_acts,act_payoffs = eq_core.calc_best_response_with_beliefs(payoffdict, belief_dict)
                    for e in eq_strats:
                        l1l2_eq = self.L1L2Equilibrium()
                        for aa_idx,a_a in enumerate(all_acts):
                            for _a_idx,_act in enumerate(a_a):
                                ag_key = (int(_act[3:6]), int(_act[6:9]))
                                if _act in e:
                                    l1l2_eq.equilibria_actions[ag_key] = (_act,act_payoffs[aa_idx][_a_idx])
                        sv_act_payoffs = []
                        for sv_act in sv_actions:
                            for aa_idx,a_a in enumerate(all_acts):
                                if len(a_a) > 0:
                                    if int(a_a[0][6:9]) == 0:
                                        for _a_idx,_act in enumerate(a_a):
                                            if _act == sv_act:
                                                sv_payoff = act_payoffs[aa_idx][_a_idx]
                                                sv_act_payoffs.append(sv_payoff)
                        l1l2_eq.all_act_payoffs = sv_act_payoffs
                        all_eq.append(l1l2_eq)
                elif self.eval_config.l1_eq == 'BR' or self.eval_config.l1_eq == 'L1BR':
                    l1_agent = True if self.eval_config.l1_eq == 'L1BR' else False
                    eq_core = EquilibriaCore(self.eval_config.num_agents,payoffdict,N_payoff_table,sv_actions,l1_agent)
                    logging.info('calculating equilibria...')
                    eq = eq_core.calc_best_response()
                    for e,p in eq.items():
                        l1l2_eq = self.L1L2Equilibrium()
                        l1l2_eq.equilibria_actions = {(int(k[0][3:6]), int(k[0][6:9])):(k[0],k[1]) for k in zip(e,p)}
                        if not l1_agent:
                            sv_act_payoffs = eq_core.sv_action_payoffs
                        else:
                            sv_act_payoffs = eq_core.sv_act_payoffs[e]
                        readable_eq = utils.print_readable(e)
                        l1l2_eq.all_act_payoffs = sv_act_payoffs
                        l1l2_eq.oth_ag_eq_info = self.construct_oth_ag_eq_info(e,p,time_ts,sv_id,payoffdict)
                        all_eq.append(l1l2_eq)
                elif self.eval_config.l1_eq == 'MAXMIN' or self.eval_config.l1_eq == 'L1MAXMIN':
                    l1_agent = True if self.eval_config.l1_eq == 'L1MAXMIN' else False
                    eq_core = EquilibriaCore(self.eval_config.num_agents,payoffdict,N_payoff_table,sv_actions,l1_agent)
                    logging.info('calculating equilibria...')
                    eq = eq_core.calc_max_min_response()
                    for e,p in eq.items():
                        l1l2_eq = self.L1L2Equilibrium()
                        l1l2_eq.equilibria_actions = {(int(k[0][3:6]), int(k[0][6:9])):(k[0],k[1]) for k in zip(e,p)}
                        if not l1_agent:
                            sv_act_payoffs = eq_core.sv_action_payoffs
                        else:
                            sv_act_payoffs = eq_core.sv_act_payoffs[e]
                        readable_eq = utils.print_readable(e)
                        l1l2_eq.all_act_payoffs = sv_act_payoffs
                        l1l2_eq.oth_ag_eq_info = self.construct_oth_ag_eq_info(e,p,time_ts,sv_id,payoffdict)
                        all_eq.append(l1l2_eq)
                else:
                    sys.exit('equilibria type not implemented')
                
                equilibra_dict[time_ts][sv_id]['eq_info'] = all_eq
                equilibra_dict[time_ts][sv_id]['all_actions'] = sv_actions
                if self.eval_config.l3_eq is not None:
                    equilibra_dict[time_ts][sv_id]['all_l3_eq'] = all_l3_eq
                if len(all_eq) > 1:
                    brk=1
            self.set_equilibria_dict(equilibra_dict)
            self.insert_to_db()
        #return equilibra_dict
                
                    
    def set_equilibria_dict(self,eq_dict):
        self.equilibria_dict = eq_dict
        
    
    def calc_equilibrium(self):
        self.calc_l1l2_equilibrium()
        
    
    
    def insert_to_db(self):
        conn = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        q_string = "SELECT max(ROWID) from EQUILIBRIUM_ACTIONS"
        c.execute(q_string)
        res = c.fetchone()
        if res[0] is not None:
            max_id = int(res[0]) + 1 
        else:
            max_id = 1
        param_str = self.eval_config.l1_eq +'|'+ self.eval_config.l3_eq +'|'+ self.eval_config.traj_type if self.eval_config.l3_eq is not None else self.eval_config.l1_eq +'|BASELINE_ONLY'
        q_string = "REPLACE INTO EQUILIBRIUM_ACTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        l3_ins_list,oth_eq_ins_list = [],[]
        for time_ts,eq_det in self.equilibria_dict.items():
            for sv_id,sv_eq in eq_det.items():
                track_id,curr_time,relev_agents,eq_act,emp_act,all_acts,all_act_payoffs,num_eq = sv_id,time_ts,[],[],self.empirical_actions[time_ts][(sv_id,0)],None,[],len(sv_eq['eq_info'])
                all_acts = sv_eq['all_actions']
                for eq_inst in sv_eq['eq_info']:
                    for k,v in eq_inst.equilibria_actions.items():
                        if k[1] != 0:
                            if k[1] not in relev_agents:
                                relev_agents.append(k[1])
                        else:
                            eq_act.append(v[0])
                    all_act_payoffs.append(eq_inst.all_act_payoffs)
                    for oth_ag_id,oth_ag_eq in eq_inst.oth_ag_eq_info.items():
                        for oth_ag_eq_det in oth_ag_eq:
                            oth_eq_ins_list.append((str(max_id), str(oth_ag_id), str(oth_ag_eq_det[0]), str(oth_ag_eq_det[1])))
                ins_tuple = (str(max_id),int(constants.CURRENT_FILE_ID),self.eval_config.direction,track_id,curr_time,None,None,None,None,None,None,None,relev_agents,None,\
                     eq_act,emp_act,param_str,all_acts,all_act_payoffs,num_eq)
                ins_tuple = tuple(str(x) if x is not None else x for x in ins_tuple)
                ins_list.append(ins_tuple)
                if self.eval_config.l3_eq is not None:
                    all_l3_eq = sv_eq['all_l3_eq']
                    for l1l2_strat,l3_eq_det in all_l3_eq.items():
                        all_l3_payoffs = l3_eq_det.all_traj_act_payoffs
                        all_l3_acts = l3_eq_det.sv_acts
                        l3_eq_acts = []
                        for eq_s,eq_p in l3_eq_det.equilibria_actions.items():
                            l3_eq_acts.append(list(eq_s))
                        l3_ins_list.append((str(max_id),constants.CURRENT_FILE_ID,self.eval_config.l3_eq,str(l3_eq_acts),str(all_l3_acts),str(all_l3_payoffs)))
                
                max_id += 1
        
        if len(ins_list) > 0:
            c.executemany(q_string,ins_list)
            conn.commit()
            if self.eval_config.l3_eq is not None:
                q_string = "REPLACE INTO L3_EQUILIBRIUM_ACTIONS VALUES (?,?,?,?,?,?)"
                c.executemany(q_string,l3_ins_list)
            q_string = "REPLACE INTO EQUILIBRIUM_ACTIONS_OTHER_AGENTS VALUES (?,?,?,?)"
            c.executemany(q_string,oth_eq_ins_list)
            conn.commit()
        conn.close()
        
    def update_db(self,colmn_list):
        attr_map = {'ALL_ACTIONS':'sub_all_actions','ALL_ACTION_PAYOFFS':'sub_all_action_payoffs_at_eq'}
        conn = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_'+constants.CURRENT_FILE_ID+'.db')
        cur = conn.cursor()
        q_string = ["UPDATE EQUILIBRIUM_ACTIONS SET "]
        for c in colmn_list[:-1]:
            q_string.append(c+'=?, ')
        q_string.append(colmn_list[-1]+'=? ')
        q_string.append(' WHERE TRACK_ID=? AND TIME=? and EQ_CONFIG_PARMS=?')
        q_string = ''.join(q_string)
        for eq_i in np.arange(len(self.sub_all_action_payoffs_at_eq)):
            _z = list(zip(self.sub_all_actions,self.sub_all_action_payoffs_at_eq[eq_i]))
            _z.sort()
            self.sub_all_action_payoffs_at_eq[eq_i] = [x[1] for x in _z]
            self.sub_all_actions = [x[0] for x in _z]
        u_tuple = [getattr(self, attr_map[c]) for c in colmn_list] + [self.track_id,self.curr_time,self.param_str]
        u_tuple = tuple([str(x) for x in u_tuple])    
        cur.execute(q_string,u_tuple)
        conn.commit()
        conn.close()



class EvalConfig:
    
    def setup_parameters(self,direction):
        self.traj_dict = dict()
        strat_traj_ids = []
        self.direction = direction
        self.traj_metadata = db_utils.get_traj_metadata(direction,self.traj_type)
        self.time_list = []

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
        
    def set_l1_eq_type(self,l1_eq):
        self.l1_eq = l1_eq
        
    def set_l3_eq_type(self,l3_eq):
        self.l3_eq = l3_eq     
        
    def set_traj_type(self,traj_type):
        self.traj_type = traj_type     
        
    def set_pedestrian_info(self, pedestrian_info):
        self.pedestrian_info = pedestrian_info
    
    def set_eq_acts_in_db(self,eq_acts_in_db):
        self.eq_acts_in_db = eq_acts_in_db
        
    def set_update_only(self,update_only):
        self.update_only = update_only
        
    
        