'''
Created on Apr 22, 2020

@author: Atrisha
'''
import numpy as np
import sqlite3

class Equilibria:
    
    def __init__(self,eq_list,sub_all_actions,sub_all_payoffs,emp_act,curr_time,traj_det,task,param_str):
        self.param_str = param_str
        self.eq_list = eq_list
        self.sub_all_actions = sub_all_actions
        eq_act_list = set()
        _eq_dict = dict()
        
        for i,eq in enumerate(eq_list):
            for sv_act in sub_all_actions:
                if sv_act in eq:
                    if sv_act in _eq_dict:
                        _eq_dict[sv_act].append(sub_all_payoffs[i])
                    else:
                        _eq_dict[sv_act] = [sub_all_payoffs[i]]
        
        for k,v in _eq_dict.items():
            _payoffs = np.vstack(v)
            _mean_payoffs = np.mean(_payoffs,axis=0)
            _eq_dict[k] = _mean_payoffs.tolist()
        self.sub_empirical_act = emp_act
        self.sub_eq_acts = list(_eq_dict.keys())
        self.sub_all_action_payoffs_at_eq = list(_eq_dict.values())
        self.num_eqs = len(eq_list)
        self.num_agents = len(eq_list[0])
        self.curr_time = curr_time
        self.task = task
        self.relev_agents = None
        for k,v in traj_det.items():
            if not isinstance(k, str):
                self.track_id = k
            elif k == 'relev_agents':
                self.relev_agents = list(v.keys())
        
    
    def insert_to_db(self):
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        q_string = "REPLACE INTO EQUILIBRIUM_ACTIONS VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        for eq_i in np.arange(len(self.sub_all_action_payoffs_at_eq)):
            _z = list(zip(self.sub_all_actions,self.sub_all_action_payoffs_at_eq[eq_i]))
            _z.sort()
            self.sub_all_action_payoffs_at_eq[eq_i] = [x[1] for x in _z]
            self.sub_all_actions = [x[0] for x in _z]
        
        ins_tuple = (769,self.task,self.track_id,self.curr_time,None,None,None,None,None,None,None,self.relev_agents,\
                     self.sub_eq_acts,self.sub_empirical_act,self.param_str,self.sub_all_actions,self.sub_all_action_payoffs_at_eq,self.num_eqs)
        ins_tuple = tuple(str(x) for x in ins_tuple)
        c.execute(q_string,ins_tuple)
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
            
        