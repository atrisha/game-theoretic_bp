'''
Created on Feb 11, 2020

@author: Atrisha
'''
import numpy as np

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
