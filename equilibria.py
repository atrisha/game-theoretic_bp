'''
Created on Feb 11, 2020

@author: Atrisha
'''
import numpy as np
import itertools

def calc_pure_strategy_nash_equilibrium_exhaustive1(pay_off_dict):
    num_players = len(list(pay_off_dict.values())[0])
    strat_sets = []
    for i in np.arange(num_players):
        strat_set = set([x[i] for x in list(pay_off_dict.keys())])
        strat_sets.append(strat_set)
    for i in np.arange(num_players):
        all_other_strats = list(itertools.product(*[strat_sets[j] for j in np.arange(num_players) if j!=i]))
        for oth_strat in all_other_strats:
            curr_agent_strats = [] 
            for strat in strat_sets[i]:
                _s = list(oth_strat)
                _s.insert(i,strat)
                curr_agent_strats.append(_s)
            max_strat_payoffs = max([v[i] for k,v in pay_off_dict.items() if k in [tuple(strat) for strat in curr_agent_strats]])
            max_strat = [k for k,v in pay_off_dict.items() if v[i]==max_strat_payoffs and k in [tuple(strat) for strat in curr_agent_strats] ]
            k=1
            
                
        
    

def calc_pure_strategy_nash_equilibrium_exhaustive(pay_off_dict):
    num_players = len(list(pay_off_dict.values())[0])
    eq = list(pay_off_dict.keys())
    N = len(pay_off_dict)
    ct = 0
    for i in np.arange(num_players):
        for k1,v1 in pay_off_dict.items():
            #print(ct,'/',N)
            #ct += 1
            for k2,v2 in pay_off_dict.items():
                ''' agent i's strategy changes'''
                if k1[i] != k2[i] and v2[i] > v1[i]:
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
    return eq



pay_off_dict = {('p1s1','p2s1','p3s1'):[0,0,0],
               ('p1s1','p2s1','p3s2'):[0,0,0],
               ('p1s1','p2s1','p3s3'):[0,0,0],
               ('p1s1','p2s2','p3s1'):[0,0,0],
               ('p1s1','p2s2','p3s2'):[0,0,0],
               ('p1s1','p2s2','p3s3'):[0,0,0],
               ('p1s1','p2s3','p3s1'):[0,0,0],
               ('p1s1','p2s3','p3s2'):[0,0,0],
               ('p1s1','p2s3','p3s3'):[0,0,0],
               ('p1s2','p2s1','p3s1'):[1,0,0],
               ('p1s2','p2s1','p3s2'):[0,0,0],
               ('p1s2','p2s1','p3s3'):[0,0,0],
               ('p1s2','p2s2','p3s1'):[0,0,0],
               ('p1s2','p2s2','p3s2'):[1,0,0],
               ('p1s2','p2s2','p3s3'):[0,0,0],
               ('p1s2','p2s3','p3s1'):[0,0,0],
               ('p1s2','p2s3','p3s2'):[0,1,0],
               ('p1s2','p2s3','p3s3'):[0,0,0],
               ('p1s3','p2s1','p3s1'):[0,0,0],
               ('p1s3','p2s1','p3s2'):[0,1,1],
               ('p1s3','p2s1','p3s3'):[0,0,0],
               ('p1s3','p2s2','p3s1'):[10,10,10],
               ('p1s3','p2s2','p3s2'):[0,0,0],
               ('p1s3','p2s2','p3s3'):[0,0,0],
               ('p1s3','p2s3','p3s1'):[0,0,0],
               ('p1s3','p2s3','p3s2'):[0,0,0],
               ('p1s3','p2s3','p3s3'):[0,0,0]}

prisoners_dilemma = {('p1s1','p2s1'):[-1,-1],
                     ('p1s1','p2s2'):[-3,0],
                     ('p1s2','p2s1'):[0,-3],
                     ('p1s2','p2s2'):[-2,-2]}

game_of_chicken = {('p1s1','p2s1'):[0,0],
                     ('p1s1','p2s2'):[7,2],
                     ('p1s2','p2s1'):[2,7],
                     ('p1s2','p2s2'):[6,6]}

from timeit import default_timer as timer

start_t = timer()
print(calc_pure_strategy_nash_equilibrium_exhaustive1(pay_off_dict))
end_t = timer()
print('time taken',end_t - start_t)