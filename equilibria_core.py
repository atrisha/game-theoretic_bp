'''
Created on Feb 11, 2020

@author: Atrisha
'''
import numpy as np
import itertools
import operator

def calc_pure_strategy_nash_equilibrium_exhaustive1(pay_off_dict):
    num_players = len(list(pay_off_dict.values())[0])
    strat_sets,eq_strats = [],[]
    for i in np.arange(num_players):
        strat_set = set([x[i] for x in list(pay_off_dict.keys())])
        strat_sets.append(strat_set)
    for i in np.arange(num_players):
        ag_eq_strats = []
        all_other_strats = list(itertools.product(*[strat_sets[j] for j in np.arange(num_players) if j!=i]))
        for oth_strat in all_other_strats:
            curr_agent_strats = [] 
            for strat in strat_sets[i]:
                _s = list(oth_strat)
                _s.insert(i,strat)
                curr_agent_strats.append(_s)
            max_strat_payoffs = max([v[i] for k,v in pay_off_dict.items() if k in [tuple(strat) for strat in curr_agent_strats]])
            max_strat = [k for k,v in pay_off_dict.items() if v[i]==max_strat_payoffs and k in [tuple(strat) for strat in curr_agent_strats] ]
            for m in max_strat:
                if m not in ag_eq_strats:
                    ag_eq_strats.append(m)
        eq_strats.append(ag_eq_strats)
    result = list(set(eq_strats[0]).intersection(*eq_strats[:1]))    
    return result
                

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

game_of_chicken = {('go','go'):[0,0],
                     ('go','wait'):[7,2],
                     ('wait','go'):[2,5],
                     ('wait','wait'):[4,4]}

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
ne_all = calc_pure_strategy_nash_equilibrium_exhaustive(toy_merge,True)
#ne = calc_pure_strategy_nash_equilibrium_exhaustive(game_of_chicken)
print(br)
print()
print('all')
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