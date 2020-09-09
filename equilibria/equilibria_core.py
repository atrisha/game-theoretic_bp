'''
Created on Feb 11, 2020

@author: Atrisha
'''
import numpy as np
import itertools
from all_utils import utils
import operator
import datetime
import os
import constants
import logging
logging.basicConfig(format='%(levelname)-8s %(filename)s: %(message)s',level=logging.INFO)

class EquilibriaCore:
    
    def __init__(self,num_players,pay_off_dict,N,sv_actions,is_l1agent):
        self.num_players = num_players
        self.pay_off_dict = pay_off_dict
        self.N = N
        self.sv_actions = sv_actions
        self.isl1agent = is_l1agent
        #self.bidirec_payoff_dict = bidict(pay_off_dict).inverse

    def calc_pure_strategy_nash_equilibrium_exhaustive(self):
        num_players = self.num_players
        pay_off_dict = self.pay_off_dict
        strat_sets,eq_strats = [],[]
        for i in np.arange(num_players):
            strat_set = set([x[i] for x in list(pay_off_dict.keys())])
            strat_sets.append(strat_set)
        start_time = datetime.datetime.now()
        for i in np.arange(num_players):
            ag_eq_strats = []
            all_other_strats = list(itertools.product(*[strat_sets[j] for j in np.arange(num_players) if j!=i]))
            for oth_strat in all_other_strats:
                curr_agent_strats = [] 
                for strat in strat_sets[i]:
                    _s = list(oth_strat)
                    _s.insert(i,strat)
                    curr_agent_strats.append(_s)
                try:
                    max_strat_payoffs = max([v[i] for k,v in pay_off_dict.items() if k in [tuple(strat) for strat in curr_agent_strats]])
                except IndexError:
                    brk=1
                max_strat = [k for k,v in pay_off_dict.items() if v[i]==max_strat_payoffs and k in [tuple(strat) for strat in curr_agent_strats] ]
                for m in max_strat:
                    if m not in ag_eq_strats:
                        ag_eq_strats.append(m)
            eq_strats.append(ag_eq_strats)
        result = list(pay_off_dict.keys())
        for i in np.arange(len(eq_strats)):
            result = list(set(result).intersection(eq_strats[i]))
            if len(result) == 0:
                break
        end_time = datetime.datetime.now()
        self.exec_time = str((end_time-start_time).microseconds/1000)
        return {k:v for k,v in pay_off_dict.items() if k in result}
    
    def calc_max_min_response_deprecated(self,all=False):
        num_players = self.num_players
        pay_off_dict = self.pay_off_dict
        strat_sets,eq_strats = [],[]
        for i in np.arange(num_players):
            strat_set = set([x[i] for x in list(pay_off_dict.keys())])
            strat_sets.append(strat_set)
        
        max_min_strats = [list()]*num_players
        for i in np.arange(num_players):
            ag_eq_strats = []
            for strat in strat_sets[i]:
                all_other_strats = list(itertools.product(*[strat_sets[j] for j in np.arange(num_players) if j!=i]))
                curr_agent_strats = []
                for oth_strat in all_other_strats:
                    _s = list(oth_strat)
                    _s.insert(i,strat)
                    curr_agent_strats.append(_s)
                min_strat_payoffs = min([v[i] for k,v in pay_off_dict.items() if k in [tuple(this_strat) for this_strat in curr_agent_strats]])
                ag_eq_strats.append((strat,min_strat_payoffs))
            ag_eq_strats.sort(key=lambda tup: tup[1], reverse = True)
            ag_eq_strats = [x[0] for x in ag_eq_strats if x[1]==ag_eq_strats[0][1]]
            max_min_strats[i] = ag_eq_strats
        eq_strats = list(itertools.product(*[v for v in max_min_strats]))
        if not all and len(eq_strats) > 1:
            cumul_payoffs_sorted = sorted([(k,sum(pay_off_dict[k])) for k in eq_strats], key=lambda tup: tup[1], reverse = True)
            result = {cumul_payoffs_sorted[0][0]:pay_off_dict[cumul_payoffs_sorted[0][0]]}
        else:
            result = {k:pay_off_dict[k] for k in eq_strats}
        return result
    
    def calc_max_min_response(self):
        payoff_dict = self.pay_off_dict
        num_players = self.num_players
        res_dict = {n:{} for n in np.arange(self.num_players)}
        for s,p in payoff_dict.items():
            for i in np.arange(num_players):  
                if s[i] not in res_dict[i]:
                    res_dict[i][s[i]] = np.inf
                if p[i] < res_dict[i][s[i]]:
                    res_dict[i][s[i]] = p[i]
        eq_strat = [None]*num_players
        self.sv_action_payoffs = []
        for k,v in res_dict.items():
            eq_strat[k] = max(v.items(), key=operator.itemgetter(1))[0]
            if next(iter(v.keys()))[6:9] == '000':
                for sv_action in self.sv_actions:
                    self.sv_action_payoffs.append(round(v[sv_action],6))
        eq_strat = tuple(eq_strat)
        eq_res = {eq_strat:payoff_dict[eq_strat]}
        if self.isl1agent:
            qbr_eq_res = dict()
            self.sv_act_payoffs = dict()
            for e,p in eq_res.items():
                eq_act_tuple = [x if x[6:9]!='000' else None for x in list(e)]
                _t_sv_act_payoffs = []
                _t_eq_pay,_t_eq_strat = [-np.inf]*self.num_players,None
                for sv_act in self.sv_actions:
                    _act_tup = tuple([x if x is not None else sv_act for x in eq_act_tuple])
                    sv_index = _act_tup.index(sv_act)
                    sv_payoff = round(payoff_dict[_act_tup][sv_index],6)
                    if sv_payoff > _t_eq_pay[sv_index]:
                        _t_eq_pay = payoff_dict[_act_tup]
                        _t_eq_strat = _act_tup
                    _t_sv_act_payoffs.append(sv_payoff)
                qbr_eq_res[_t_eq_strat] = _t_eq_pay
                self.sv_act_payoffs[_t_eq_strat] = _t_sv_act_payoffs
            return qbr_eq_res
        else:
            return eq_res
        return eq_res
    
    def calc_best_response(self):
        payoff_dict = self.pay_off_dict
        num_players = self.num_players
        res_dict = {n:{} for n in np.arange(self.num_players)}
        for s,p in payoff_dict.items():
            for i in np.arange(num_players):  
                if s[i] not in res_dict[i]:
                    res_dict[i][s[i]] = -np.inf 
                if p[i] > res_dict[i][s[i]]:
                    res_dict[i][s[i]] = p[i]
        eq_strat = [None]*num_players
        
        self.sv_action_payoffs = []
        for k,v in res_dict.items():
            eq_strat[k] = max(v.items(), key=operator.itemgetter(1))[0]
            if next(iter(v.keys()))[6:9] == '000':
                for sv_action in self.sv_actions:
                    self.sv_action_payoffs.append(round(v[sv_action],6))
        eq_strat = tuple(eq_strat)
        eq_res = {eq_strat:payoff_dict[eq_strat]}
        if self.isl1agent:
            qbr_eq_res = dict()
            self.sv_act_payoffs = dict()
            for e,p in eq_res.items():
                eq_act_tuple = [x if x[6:9]!='000' else None for x in list(e)]
                _t_sv_act_payoffs = []
                _t_eq_pay,_t_eq_strat = [-np.inf]*self.num_players,None
                for sv_act in self.sv_actions:
                    _act_tup = tuple([x if x is not None else sv_act for x in eq_act_tuple])
                    sv_index = _act_tup.index(sv_act)
                    sv_payoff = round(payoff_dict[_act_tup][sv_index],6)
                    if sv_payoff > _t_eq_pay[sv_index]:
                        _t_eq_pay = payoff_dict[_act_tup]
                        _t_eq_strat = _act_tup
                    _t_sv_act_payoffs.append(sv_payoff)
                qbr_eq_res[_t_eq_strat] = _t_eq_pay
                self.sv_act_payoffs[_t_eq_strat] = _t_sv_act_payoffs
            return qbr_eq_res
        else:
            return eq_res
    
    def calc_best_response_deprecated2(self):
        num_players = len(list(self.pay_off_dict.values())[0]) 
        br_strats = tuple([max(self.pay_off_dict.keys(), key=(lambda k: self.pay_off_dict[k][i]))[i] for i in np.arange(num_players)])
        br_payoffs = self.pay_off_dict[br_strats]
        eq_dict = dict()
        for i in np.arange(num_players):
            for k,v in self.pay_off_dict.items():
                if v[i]==br_payoffs[i]:
                    eq_dict[k] = v
        return eq_dict
        
        

def calc_pure_strategy_nash_equilibrium_exhaustive2(pay_off_dict,all_eqs=False):
    num_players = 2
    for k,v in pay_off_dict.items():
        for i in np.arange(num_players):
            p_k = i
        
                

def calc_best_response(pay_off_dict):
    num_players = len(list(pay_off_dict.values())[0]) 
    br_strats = tuple([max(pay_off_dict.keys(), key=(lambda k: pay_off_dict[k][i]))[i] for i in np.arange(num_players)])
    br_payoffs = pay_off_dict[br_strats]
    eq_dict = dict()
    for i in np.arange(num_players):
        for k,v in pay_off_dict.items():
            if v[i]==br_payoffs[i]:
                eq_dict[k] = v
    return eq_dict

def calc_best_response_with_beliefs(pay_off_dict,belief_dict):
    num_players = len(list(pay_off_dict.values())[0])
    player_actions = dict()
    for i in np.arange(num_players):
        pl_acts = set([x[i] for x in list(pay_off_dict.keys())])
        player_actions[i] = dict()
        for act in pl_acts:
            player_actions[i][act] = {k:v for k,v in pay_off_dict.items() if k[i]==act } 
    all_acts,act_payoffs = [],[]
    for i in np.arange(num_players):
        for pl_act,act_dict in player_actions[i].items():
            exp_payoffs = 0
            for k,v in act_dict.items():
                bel_vect = belief_dict[k]
                strat_prob = np.prod([x for idx,x in enumerate(bel_vect) if idx!=i])
                _p = strat_prob*v[i]
                exp_payoffs += _p
            player_actions[i][pl_act] = exp_payoffs
    eq_acts = []
    for i in np.arange(num_players):
        all_acts.append(list(player_actions[i].keys()))
        act_payoffs.append(list(player_actions[i].values()))
        max_payoffs = max(list(player_actions[i].values()))
        eq_acts.append([x for idx,x in enumerate(list(player_actions[i].keys())) if list(player_actions[i].values())[idx]==max_payoffs])
    eq_strats = list(itertools.product(*[v for v in eq_acts]))
    return eq_strats,all_acts,act_payoffs
    
            

def calc_pure_strategy_nash_equilibrium_exhaustive(pay_off_dict,all_eqs=False):
    num_players = len(list(pay_off_dict.values())[0])
    start_time = datetime.datetime.now()
    eq = list(pay_off_dict.keys())
    N = len(pay_off_dict)
    ct = 0
    for i in np.arange(num_players):
        for k1,v1 in pay_off_dict.items():
            #print(ct,'/',N)
            #ct += 1
            for k2,v2 in pay_off_dict.items():
                if k2==(124,27387,29316,31028) and k1 == (64,27387,28595,30446) and i==1:
                    brk=1
                    
                ''' agent i's strategy changes'''
                _v1 = v1[i]
                _v2 = v2[i]
                if v2[i] > v1[i]:
                    ''' all other agent's strategy remains same '''
                    oth_strategy_same = True
                    for j in np.arange(num_players):
                        if j!= i:
                            if k2[j] == k1[j]:
                                oth_strategy_same = oth_strategy_same and True
                            else:
                                oth_strategy_same = False
                                break
                    ''' not an equilibrium '''
                    if k1 in eq and oth_strategy_same:
                        eq.remove(k1)
    eq_dict =  {k:pay_off_dict[k] for k in eq}
    end_time = datetime.datetime.now()
    print((end_time-start_time).microseconds)
    if not all_eqs:
        ''' if multiple nash equilibria found, then select the one that has highest cumulative payoff '''
        if len(eq_dict) > 1:
            max_sum = max([sum(v) for v in eq_dict.values()])
            for k,v in eq_dict.items():
                if sum(v) == max_sum:
                    return {k:v}
        else:
            return eq_dict
    else:
        return eq_dict


pay_off_dict = {(64, 27387, 28595, 30446): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28595, 30460): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28595, 30593): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28595, 30607): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28595, 31028): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28916, 30446): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28916, 30460): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28916, 30593): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28916, 30607): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 28916, 31028): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29165, 30446): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29165, 30460): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29165, 30593): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29165, 30607): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29165, 31028): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29209, 30446): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29209, 30460): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29209, 30593): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29209, 30607): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29209, 31028): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29316, 30446): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29316, 30460): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29316, 30593): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29316, 30607): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (64, 27387, 29316, 31028): np.asarray([0.66578904, 1.63304171, 0.66578904, 2.        ]),
 (68, 27387, 28595, 30446): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28595, 30460): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28595, 30593): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28595, 30607): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28595, 31028): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28916, 30446): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28916, 30460): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28916, 30593): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28916, 30607): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 28916, 31028): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29165, 30446): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29165, 30460): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29165, 30593): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29165, 30607): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29165, 31028): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29209, 30446): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29209, 30460): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29209, 30593): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29209, 30607): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29209, 31028): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29316, 30446): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29316, 30460): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29316, 30593): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29316, 30607): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (68, 27387, 29316, 31028): np.asarray([0.66578904, 1.67768301, 0.66578904, 2.        ]),
 (75, 27387, 28595, 30446): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28595, 30460): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28595, 30593): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28595, 30607): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28595, 31028): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28916, 30446): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28916, 30460): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28916, 30593): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28916, 30607): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 28916, 31028): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29165, 30446): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29165, 30460): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29165, 30593): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29165, 30607): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29165, 31028): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29209, 30446): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29209, 30460): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29209, 30593): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29209, 30607): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29209, 31028): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29316, 30446): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29316, 30460): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29316, 30593): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29316, 30607): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (75, 27387, 29316, 31028): np.asarray([0.66578904, 1.71812483, 0.66578904, 2.        ]),
 (89, 27387, 28595, 30446): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28595, 30460): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28595, 30593): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28595, 30607): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28595, 31028): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28916, 30446): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28916, 30460): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28916, 30593): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28916, 30607): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 28916, 31028): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29165, 30446): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29165, 30460): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29165, 30593): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29165, 30607): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29165, 31028): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29209, 30446): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29209, 30460): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29209, 30593): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29209, 30607): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29209, 31028): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29316, 30446): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29316, 30460): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29316, 30593): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29316, 30607): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (89, 27387, 29316, 31028): np.asarray([0.66578904, 1.61923238, 0.66578904, 2.        ]),
 (124, 27387, 28595, 30446): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28595, 30460): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28595, 30593): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28595, 30607): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28595, 31028): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28916, 30446): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28916, 30460): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28916, 30593): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28916, 30607): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 28916, 31028): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29165, 30446): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29165, 30460): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29165, 30593): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29165, 30607): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29165, 31028): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29209, 30446): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29209, 30460): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29209, 30593): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29209, 30607): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29209, 31028): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29316, 30446): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29316, 30460): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29316, 30593): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29316, 30607): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ]),
 (124, 27387, 29316, 31028): np.asarray([0.66578904, 1.72805004, 0.66578904, 2.        ])}

prisoners_dilemma = {('p1s1','p2s1'):[-1,-1],
                     ('p1s1','p2s2'):[-3,0],
                     ('p1s2','p2s1'):[0,-3],
                     ('p1s2','p2s2'):[-2,-2]}

game_of_chicken2 = {('go','go'):[0,0],
                     ('go','wait'):[7,2],
                     ('wait','go'):[2,5],
                     ('wait','wait'):[4,4]}

game_of_chicken = {('go','go'):[1,-1],
                     ('go','wait'):[-1,1],
                     ('wait','go'):[-1,1],
                     ('wait','wait'):[1,-1]}


toy_merge = {(('wait', 'cancel'), ('slow', 'cont. speed')): [10, 20], 
             (('wait', 'cancel'), ('speed', 'cont. speed')): [10, 20], 
             (('wait', 'cancel'), ('slow', 'slow')): [10, 20],
              (('wait', 'cancel'), ('speed', 'slow')): [10, 20],
               (('wait', 'cont. merge'), ('slow', 'cont. speed')): [10, 20],
                (('wait', 'cont. merge'), ('speed', 'cont. speed')): [10, 20],
                 (('wait', 'cont. merge'), ('slow', 'slow')): [10, 20],
                  (('wait', 'cont. merge'), ('speed', 'slow')): [10, 20],
                   (('merge', 'cancel'), ('slow', 'cont. speed')): [19,19],
                    (('merge', 'cancel'), ('speed', 'cont. speed')): [10,10],
                     (('merge', 'cancel'), ('slow', 'slow')): [19,19],
                      (('merge', 'cancel'), ('speed', 'slow')): [10,10], 
                      (('merge', 'cont. merge'), ('slow', 'cont. speed')): [19,19],
                       (('merge', 'cont. merge'), ('speed', 'cont. speed')): [-100,-100], 
                       (('merge', 'cont. merge'), ('slow', 'slow')): [19,19],
                        (('merge', 'cont. merge'), ('speed', 'slow')): [10,14]}

'''
br_b = calc_best_response_with_beliefs(toy_merge,dict())
br = calc_best_response(toy_merge)
'''
eq = EquilibriaCore(2,game_of_chicken,len(game_of_chicken),None,None)
ne_all = eq.calc_pure_strategy_nash_equilibrium_exhaustive()
#ne = calc_pure_strategy_nash_equilibrium_exhaustive(game_of_chicken)
#print(br)
print()
print('all toy1')
for k,v in ne_all.items():
    print(k,v)
'''
ne_all = calc_pure_strategy_nash_equilibrium_exhaustive(pay_off_dict,True)
#ne = calc_pure_strategy_nash_equilibrium_exhaustive(game_of_chicken)
#print(br)
print()
print('all toy')
for k,v in ne_all.items():
    print(k,v)
#print(ne)
'''

'''
from timeit import default_timer as timer

start_t = timer()
eq = calc_pure_strategy_nash_equilibrium_exhaustive(pay_off_dict)
print(len(eq))
end_t = timer()
print('time taken',end_t - start_t)
'''
'''
start_t = timer()
eq = calc_pure_strategy_nash_equilibrium_exhaustive1(pay_off_dict)
print(len(eq))
end_t = timer()
print('time taken',end_t - start_t)
'''
'''
l3_payoff = all_utils.pickle_load(os.path.join(constants.ROOT_DIR,constants.TEMP_TRAJ_CACHE,'l3_payoff'))

eq_core = EquilibriaCore(4,l3_payoff,len(l3_payoff))
start_time = datetime.datetime.now()
eq = eq_core.calc_max_min_response_eff()
end_time = datetime.datetime.now()
exec_time = str((end_time-start_time).microseconds/1000)
print(eq)
print(exec_time+'ms')

eq_core = EquilibriaCore(4,l3_payoff,len(l3_payoff))
start_time = datetime.datetime.now()
eq = eq_core.calc_max_min_response()
end_time = datetime.datetime.now()
exec_time = str((end_time-start_time).microseconds/1000)
print(eq)
print(exec_time+'ms')


f=1
'''
