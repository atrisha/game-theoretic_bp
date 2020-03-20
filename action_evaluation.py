'''
Created on Mar 20, 2020

@author: Atrisha
'''

import constants
import numpy as np

def eval_l2_action(time_tx,vel_profile):
    selected_indices = np.arange(0,len(vel_profile),constants.DATASET_FPS*constants.LP_FREQ,dtype=int)
    vel_profile = vel_profile[selected_indices]
    ra = [(x[1]-x[0])/constants.LP_FREQ for x in zip(vel_profile[:-1],vel_profile[1:])]
    if max(ra) < constants.MAX_LONG_ACC_NORMAL:
        l2_action = 'NORMAL'
    else:
        l2_action = 'AGGRESSIVE'
    return l2_action

