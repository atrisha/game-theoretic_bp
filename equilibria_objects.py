'''
Created on Apr 22, 2020

@author: Atrisha
'''
import numpy as np
import sqlite3
import db_utils
import utils
from planning_objects import *
from constants import *
import itertools
import ast
from cost_evaluation import CostEvaluation
import equilibria_core as eq_core
import sys
import logging
logging.basicConfig(format='%(levelname)-8s %(filename)s: %(message)s',level=logging.INFO)

class Equilibria:
    
    class L1L2Equilibrium:
        
        def __init__(self):
            self.equilibria_actions = dict()      
            
            
    class L3Equilibria:
        
        def __init__(self,eval_config):
            self.equibilria_type = eval_config.l3_eq
        
        def set_equilibria(self,eq_dict):
            self.eq_dict = eq_dict    
        
    
    def __init__(self,eval_config):
        self.eval_config = eval_config
    
        
                
    
    def calc_empirical_actions(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "SELECT * FROM TRAJECTORIES_0769_EXT WHERE L1_ACTION IS NOT NULL"
        c.execute(q_string)
        res = c.fetchall()
        emp_act_in_db = dict()
        for row in res:
            emp_act_in_db[(row[1],row[0])] = ast.literal_eval(row[3])
        conn_trajdb = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
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
                                        baseline_track_time_ref = available_times[[round(x) for x in available_times].index(int(emp_traj_start_ts))]
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
                
                
        if len(emp_act_in_db) == 0:
            emp_act_in_db = dict()
            for ts,v in self.empirical_actions.items():
                for k,acts in v.items():
                    if k[1] == 0:
                        emp_act_in_db[(ts,k[0])] = acts
                    else:
                        emp_act_in_db[(ts,k[1])] = acts
            u_string = "UPDATE TRAJECTORIES_0769_EXT SET L1_ACTION=? WHERE TRACK_ID=? AND TIME=?"
            u_list = []
            for k,v in emp_act_in_db.items():
                u_list.append((str(v),k[1],k[0]))
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
        unreadable_str = '769'+agent+relev_agent+l1_action+l2_action
        return unreadable_str
                
    def build_l1l2_utility_table(self,sv_det,time_ts):
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
        self.trajectory_cache = utils.load_trajs_for_traj_info_id(all_baseline_ids)
        if 'BELIEF' in self.eval_config.l1_eq:
            return utility_dict,sv_actions,belief_dict
        else:
            return utility_dict,sv_actions,None
    
    def calc_l1l2_equilibrium(self):
        equilibra_dict = dict()
        ct,N = 0,len(self.eval_config.traj_metadata)
        for time_ts,m_data in self.eval_config.traj_metadata.items():
            ct += 1
            self.curr_time = time_ts
            pedestrian_info = utils.setup_pedestrian_info(time_ts)
            self.eval_config.set_pedestrian_info(pedestrian_info)
            equilibra_dict[time_ts] = dict()
            for sv_id,sv_det in m_data.items():
                
                equilibra_dict[time_ts][sv_id] = dict()
                l1l2_utility_dict,sv_actions,belief_dict = self.build_l1l2_utility_table(sv_det,time_ts)
                payoffdict = dict()
                for l1l2_strat,l1l2_utility in l1l2_utility_dict.items():
                    cost_ev = CostEvaluation(self)
                    l3_payoff = cost_ev.calc_l3_payoffs(self,l1l2_strat)
                    assert(len(l3_payoff)==1)
                    ''' the l1l2 payoff table is constructed with the payoffs from lower levels '''
                    payoffdict[tuple([k.split('-')[0] for k in l1l2_strat])] = next(iter(l3_payoff.values()))
                ''' release this '''    
                l1l2_utility_dict = None
                N_payoff_table = len(payoffdict)
                logging.info(self.eval_config.direction+" "+str(time_ts)+"-"+str(sv_id)+":"+str(ct)+'/'+str(N)+":"+str(N_payoff_table))
                all_eq = []
                if self.eval_config.l1_eq == 'NASH':
                    eq = eq_core.calc_pure_strategy_nash_equilibrium_exhaustive(payoffdict,True)
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
                                                
                                                
                               
                else:
                    sys.exit('equilibria type not implemented')
                equilibra_dict[time_ts][sv_id]['eq_info'] = all_eq
                equilibra_dict[time_ts][sv_id]['all_actions'] = sv_actions
                if len(all_eq) > 1:
                    brk=1 
        return equilibra_dict
                
                    
    def set_equilibria_dict(self,eq_dict):
        self.equilibria_dict = eq_dict
        
    
    def calc_equilibrium(self):
        equilibra_dict = self.calc_l1l2_equilibrium()
        return equilibra_dict
    
    
    def insert_to_db(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        param_str = self.eval_config.l1_eq +'|'+ self.eval_config.l3_eq if self.eval_config.l3_eq is not None else self.eval_config.l1_eq +'|BASELINE_ONLY'
        q_string = "REPLACE INTO EQUILIBRIUM_ACTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        ins_list = []
        for time_ts,eq_det in self.equilibria_dict.items():
            for sv_id,sv_eq in eq_det.items():
                track_id,curr_time,relev_agents,eq_act,emp_act,all_acts,all_act_payoffs,num_eq = sv_id,time_ts,[],[],self.empirical_actions[time_ts][(sv_id,0)],None,[],len(sv_eq['eq_info'])
                all_acts = sv_eq['all_actions']
                for eq_inst in sv_eq['eq_info']:
                    for k,v in eq_inst.equilibria_actions.items():
                        if k[1] != 0:
                            relev_agents.append(k[1])
                        else:
                            eq_act.append(v[0])
                    all_act_payoffs.append(eq_inst.all_act_payoffs)
                ins_tuple = (int(constants.CURRENT_FILE_ID),self.eval_config.direction,track_id,curr_time,None,None,None,None,None,None,None,relev_agents,None,\
                     eq_act,emp_act,param_str,all_acts,all_act_payoffs,num_eq)
                ins_tuple = tuple(str(x) if x is not None else x for x in ins_tuple)
                ins_list.append(ins_tuple)
        c.executemany(q_string,ins_list)
        conn.commit()
        conn.close()
        
    def update_db(self,colmn_list):
        attr_map = {'ALL_ACTIONS':'sub_all_actions','ALL_ACTION_PAYOFFS':'sub_all_action_payoffs_at_eq'}
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
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
        self.traj_metadata = db_utils.get_traj_metadata(direction)
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
        
    def set_pedestrian_info(self, pedestrian_info):
        self.pedestrian_info = pedestrian_info
    
        