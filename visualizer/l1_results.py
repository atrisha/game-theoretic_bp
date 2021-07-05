'''
Created on Nov 27, 2020

@author: Atrisha
'''
import numpy as np
import sqlite3
from all_utils import utils, stat_utils
import ast
import sys
import constants
from scipy import stats
import matplotlib.pyplot as plt
import json
import os
import os.path
import csv
import itertools
from equilibria import equilibria_core
from pylatex import Document, Section, Subsection, Tabular, MultiRow
import pylatex.utils
from collections import OrderedDict
import pandas as pd
from collections import Counter
import matplotlib as mpl
import seaborn as sn

log = constants.common_logger

def analyse_l1_precision_spec_ratio_eq_table(l3_eq,old_models):
    precision_dict = dict()
    accuracy_dict_left,accuracy_dict_right = dict(),dict()
    for l3_eq in ['BASELINE','BOUNDARY','GAUSSIAN']:
        for file_id in constants.ALL_FILE_IDS: 
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            q_string = "select * from EQUILIBRIUM_ACTIONS"
            c.execute(q_string)
            res = c.fetchall()  
            ag_info_map = utils.load_agent_info_map(file_id)  
            
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                print(file_id,'processing',ct,N)
                model_str = row[16]
                soln_strats = ast.literal_eval(row[14])
                emp_strats = ast.literal_eval(row[15])
                if len(soln_strats) == 0:
                    continue
                ag_soln_acts = [int(x[9:11]) for x in soln_strats]
                ag_emp_acts = [int(x[9:11]) for x in emp_strats]
                
                if constants.SEGMENT_MAP[row[7]] in ['prep-right-turn','exec-right-turn','right-turn-lane','exit-lane']:
                    if model_str not in accuracy_dict_right:
                        accuracy_dict_right[model_str] = {'t':0,'f':0}
                    '''
                    if len(list(set(ag_soln_acts) & set(ag_emp_acts))) != 0:
                        accuracy_dict[model_str]['t'] += 1
                    else:
                        accuracy_dict[model_str]['f'] += 1
                    '''
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        accuracy_dict_right[model_str]['t'] += 1
                    elif len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0:
                        accuracy_dict_right[model_str]['t'] += 1
                    else:
                        accuracy_dict_right[model_str]['f'] += 1
                elif constants.SEGMENT_MAP[row[7]] in ['prep-left-turn','exec-left-turn','left-turn-lane','exit-lane']:
                    if model_str not in accuracy_dict_left:
                        accuracy_dict_left[model_str] = {'t':0,'f':0}
                    '''
                    if len(list(set(ag_soln_acts) & set(ag_emp_acts))) != 0:
                        accuracy_dict[model_str]['t'] += 1
                    else:
                        accuracy_dict[model_str]['f'] += 1
                    '''
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        accuracy_dict_left[model_str]['t'] += 1
                    elif len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0:
                        accuracy_dict_left[model_str]['t'] += 1
                    else:
                        accuracy_dict_left[model_str]['f'] += 1
                
                if constants.SEGMENT_MAP[row[7]] in ['prep-right-turn','exec-right-turn']:
                    if model_str not in precision_dict:
                        precision_dict[model_str] = dict()
                        precision_dict[model_str]['right'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                        precision_dict[model_str]['left'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['right']['tp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['right']['fp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['right']['tn'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['right']['fn'] += 1
                elif constants.SEGMENT_MAP[row[7]] in ['prep-left-turn','exec-left-turn']:
                    if model_str not in precision_dict:
                        precision_dict[model_str] = dict()
                        precision_dict[model_str]['right'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                        precision_dict[model_str]['left'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['left']['tp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['left']['fp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['left']['tn'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['left']['fn'] += 1
                else:
                    continue
                
                
                
                    
                
                
                    
                
                '''
                if 2 not in ag_soln_acts:
                    continue
                ag_emp_acts = list(set([int(x[ag_idx][9:11]) for x in emp_strats]))
                if model_str not in precision_dict:
                    precision_dict[model_str] = {'tp':0,'fp':0}
                if 2 in ag_emp_acts:
                    precision_dict[model_str]['tp'] += 1
                else:
                    precision_dict[model_str]['fp'] += 1
                
                if 1 not in ag_emp_acts:
                    continue
                if model_str not in precision_dict:
                    precision_dict[model_str] = {'tp':0,'fn':0}
                if 1 in ag_soln_acts:
                    precision_dict[model_str]['tp'] += 1
                else:
                    precision_dict[model_str]['fn'] += 1
                '''
    #precision_dict = {k:round(v['tp']/(v['tp']+v['fp']),2) for k,v in precision_dict.items()}
    left_metric,right_metric,lm_tp,lm_fp,lm_tn,lm_fn,rm_tp,rm_fp,rm_tn,rm_fn = [],[],[],[],[],[],[],[],[],[]
    for k,v in precision_dict.items():
        
        lm = round(np.mean([v['left']['tp']/(v['left']['tp']+v['left']['fp']) if v['left']['tp']+v['left']['fp'] !=0 else 0, v['left']['tn']/(v['left']['tn']+v['left']['fp'])]),2)
        rm = round(np.mean([v['right']['tp']/(v['right']['tp']+v['right']['fp'])  if v['right']['tp']+v['right']['fp'] !=0 else 0, v['right']['tn']/(v['right']['tn']+v['right']['fp'])]),2)
        left_metric.append(lm)
        right_metric.append(rm)
        rm = round(v['right']['tp']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['tp']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_tp.append(lm)
        rm_tp.append(rm)
        rm = round(v['right']['tn']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['tn']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_tn.append(lm)
        rm_tn.append(rm)
        rm = round(v['right']['fp']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['fp']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_fp.append(lm)
        rm_fp.append(rm)
        rm = round(v['right']['fn']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['fn']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_fn.append(lm)
        rm_fn.append(rm)
    accuracy_dict_merged = {k:round((accuracy_dict_right[k]['t']+accuracy_dict_left[k]['t'])/(accuracy_dict_right[k]['t']+accuracy_dict_right[k]['f']+accuracy_dict_left[k]['t']+accuracy_dict_left[k]['f']),2) for k in accuracy_dict_right.keys()}
    accuracy_dict_left = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_left.items()}
    accuracy_dict_right = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_right.items()}
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.25)
    
    plt.plot(list(np.arange(len(left_metric))),left_metric, label="left turn")
    plt.plot(list(np.arange(len(right_metric))),right_metric, label="right turn")
    plt.ylabel('precision-specificity ratio')
    plt.legend(loc="upper left")
    plt.title(l3_eq)
    plt.xticks(list(np.arange(len(left_metric))), list(precision_dict.keys()), rotation=45)
    
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.plot(list(np.arange(len(accuracy_dict_left))),list(accuracy_dict_left.values()), label="left turn")
    plt.plot(list(np.arange(len(accuracy_dict_right))),list(accuracy_dict_right.values()), label="right turn")
    plt.plot(list(np.arange(len(accuracy_dict_merged))),list(accuracy_dict_merged.values()), label="all")
    plt.ylabel('accuracy')
    plt.title(l3_eq)
    assert all([x1==x2 for x1,x2 in zip(accuracy_dict_left.keys(),accuracy_dict_right.keys())])
    plt.xticks(list(np.arange(len(accuracy_dict_left))), list(accuracy_dict_left.keys()), rotation=45)
    plt.legend(loc="upper left")
    plt.show()
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.25)
    
    plt.plot(list(np.arange(len(lm_tp))),lm_tp, label="left true pos")
    plt.plot(list(np.arange(len(lm_tn))),lm_tn, label="left true neg")
    plt.plot(list(np.arange(len(lm_fp))),lm_fp, label="left false pos")
    plt.plot(list(np.arange(len(lm_fn))),lm_fn, label="left false neg")
    plt.plot(list(np.arange(len(rm_tp))),rm_tp, label="right true pos")
    plt.plot(list(np.arange(len(rm_tn))),rm_tn, label="right true neg")
    plt.plot(list(np.arange(len(rm_fp))),rm_fp, label="right false pos")
    plt.plot(list(np.arange(len(rm_fn))),rm_fn, label="right false neg")
    plt.legend(loc="upper left")
    plt.title(l3_eq)
    plt.xticks(list(np.arange(len(rm_fn))), list(precision_dict.keys()), rotation=45)
    plt.show()
    f=1
 

def analyse_l1_precision_spec_ratio(l3_eq,old_models):
    precision_dict = dict()
    accuracy_dict_left,accuracy_dict_right = dict(),dict()
    for l3_eq in ['BASELINE','BOUNDARY','GAUSSIAN']:
        for file_id in constants.ALL_FILE_IDS: 
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            q_string = "select * from L1_SOLUTIONS"
            c.execute(q_string)
            res = c.fetchall()  
            ag_info_map = utils.load_agent_info_map(file_id)  
            
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                print(file_id,'processing',ct,N)
                model_type = row[3]
                model_params = row[4].split(',')
                model_params = {x.split('=')[0]:x.split('=')[1] for x in model_params}
                model_str = model_type
                if model_params['l3_sampling'] != l3_eq:
                    continue
                if ('old_model' not in model_params and old_models) or ( 'old_model' in model_params and model_params['old_model'] != 'True' and old_models):
                    continue
                if old_models:
                    if model_type in ['QlkR','brtb']:
                        continue
                    if model_params['baseline_weights'] != 'True':
                        continue
                
                if model_type == 'Ql1':
                    model_str = model_str + '_' + model_params['lzero_behavior']
                model_str = model_str + '_' + l3_eq + '_' + model_params['l3_soln']
                ag_vect = ag_info_map[(row[0],row[1],row[2])] if (row[0],row[1],row[2]) in ag_info_map else None
                if ag_vect is None:
                    continue
                soln_strats = ast.literal_eval(row[7])
                rule_strats = ast.literal_eval(row[6])
                emp_strats = ast.literal_eval(row[5])
                ag_idx = None
                ag_id = row[1]
                if len(soln_strats) != 0:
                    _s = soln_strats[0]
                    for idx,act in enumerate(list(_s)):
                        act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                        if ag_id == act_ag_id:
                            ag_idx = idx
                            break
                else:
                    if len(emp_strats) != 0:
                        _s = emp_strats[0]
                        for idx,act in enumerate(list(_s)):
                            act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                            if ag_id == act_ag_id:
                                ag_idx = idx
                                break
                if ag_id is None:
                    sys.exit('something wrong')
                if len(soln_strats) == 0:
                    continue
                if ag_idx != 0:
                    continue
                ag_soln_acts = list(set([int(x[ag_idx][9:11]) for x in soln_strats]))
                ag_emp_acts = list(set([int(x[ag_idx][9:11]) for x in emp_strats]))
                
                if constants.SEGMENT_MAP[ag_vect[3]] in ['prep-right-turn','exec-right-turn','right-turn-lane','exit-lane']:
                    if model_str not in accuracy_dict_right:
                        accuracy_dict_right[model_str] = {'t':0,'f':0}
                    '''
                    if len(list(set(ag_soln_acts) & set(ag_emp_acts))) != 0:
                        accuracy_dict[model_str]['t'] += 1
                    else:
                        accuracy_dict[model_str]['f'] += 1
                    '''
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        accuracy_dict_right[model_str]['t'] += 1
                    elif len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0:
                        accuracy_dict_right[model_str]['t'] += 1
                    else:
                        accuracy_dict_right[model_str]['f'] += 1
                elif constants.SEGMENT_MAP[ag_vect[3]] in ['prep-left-turn','exec-left-turn','left-turn-lane','exit-lane']:
                    if model_str not in accuracy_dict_left:
                        accuracy_dict_left[model_str] = {'t':0,'f':0}
                    '''
                    if len(list(set(ag_soln_acts) & set(ag_emp_acts))) != 0:
                        accuracy_dict[model_str]['t'] += 1
                    else:
                        accuracy_dict[model_str]['f'] += 1
                    '''
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        accuracy_dict_left[model_str]['t'] += 1
                    elif len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.WAIT_ACTIONS]))) > 0:
                        accuracy_dict_left[model_str]['t'] += 1
                    else:
                        accuracy_dict_left[model_str]['f'] += 1
                
                if constants.SEGMENT_MAP[ag_vect[3]] in ['prep-right-turn','exec-right-turn']:
                    if model_str not in precision_dict:
                        precision_dict[model_str] = dict()
                        precision_dict[model_str]['right'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                        precision_dict[model_str]['left'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['right']['tp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['right']['fp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['right']['tn'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['right']['fn'] += 1
                elif constants.SEGMENT_MAP[ag_vect[3]] in ['prep-left-turn','exec-left-turn']:
                    if model_str not in precision_dict:
                        precision_dict[model_str] = dict()
                        precision_dict[model_str]['right'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                        precision_dict[model_str]['left'] = {'tp':0,'fp':0,'tn':0,'fn':0}
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['left']['tp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['left']['fp'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0:
                        precision_dict[model_str]['left']['tn'] += 1
                    if len(list(set(ag_soln_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) == 0 and len(list(set(ag_emp_acts) & set([constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]))) > 0:
                        precision_dict[model_str]['left']['fn'] += 1
                else:
                    continue
                
                
                
                    
                
                
                    
                
                '''
                if 2 not in ag_soln_acts:
                    continue
                ag_emp_acts = list(set([int(x[ag_idx][9:11]) for x in emp_strats]))
                if model_str not in precision_dict:
                    precision_dict[model_str] = {'tp':0,'fp':0}
                if 2 in ag_emp_acts:
                    precision_dict[model_str]['tp'] += 1
                else:
                    precision_dict[model_str]['fp'] += 1
                
                if 1 not in ag_emp_acts:
                    continue
                if model_str not in precision_dict:
                    precision_dict[model_str] = {'tp':0,'fn':0}
                if 1 in ag_soln_acts:
                    precision_dict[model_str]['tp'] += 1
                else:
                    precision_dict[model_str]['fn'] += 1
                '''
    #precision_dict = {k:round(v['tp']/(v['tp']+v['fp']),2) for k,v in precision_dict.items()}
    left_metric,right_metric,lm_tp,lm_fp,lm_tn,lm_fn,rm_tp,rm_fp,rm_tn,rm_fn = [],[],[],[],[],[],[],[],[],[]
    for k,v in precision_dict.items():
        
        lm = round(np.mean([v['left']['tp']/(v['left']['tp']+v['left']['fp']) if v['left']['tp']+v['left']['fp'] !=0 else 0, v['left']['tn']/(v['left']['tn']+v['left']['fp'])]),2)
        rm = round(np.mean([v['right']['tp']/(v['right']['tp']+v['right']['fp'])  if v['right']['tp']+v['right']['fp'] !=0 else 0, v['right']['tn']/(v['right']['tn']+v['right']['fp'])]),2)
        left_metric.append(lm)
        right_metric.append(rm)
        rm = round(v['right']['tp']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['tp']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_tp.append(lm)
        rm_tp.append(rm)
        rm = round(v['right']['tn']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['tn']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_tn.append(lm)
        rm_tn.append(rm)
        rm = round(v['right']['fp']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['fp']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_fp.append(lm)
        rm_fp.append(rm)
        rm = round(v['right']['fn']/(v['right']['tp']+v['right']['fp']+v['right']['tn']+v['right']['fn']),2)
        lm = round(v['left']['fn']/(v['left']['tp']+v['left']['fp']+v['left']['tn']+v['left']['fn']),2)
        lm_fn.append(lm)
        rm_fn.append(rm)
    accuracy_dict_merged = {k:round((accuracy_dict_right[k]['t']+accuracy_dict_left[k]['t'])/(accuracy_dict_right[k]['t']+accuracy_dict_right[k]['f']+accuracy_dict_left[k]['t']+accuracy_dict_left[k]['f']),2) for k in accuracy_dict_right.keys()}
    accuracy_dict_left = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_left.items()}
    accuracy_dict_right = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_right.items()}
    plt.plot(list(np.arange(len(left_metric))),left_metric, label="left turn")
    plt.plot(list(np.arange(len(right_metric))),right_metric, label="right turn")
    plt.ylabel('precision-specificity ratio')
    plt.legend(loc="upper left")
    plt.title(l3_eq)
    plt.xticks(list(np.arange(len(left_metric))), list(precision_dict.keys()), rotation=45)
    
    plt.figure()
    plt.plot(list(np.arange(len(accuracy_dict_left))),list(accuracy_dict_left.values()), label="left turn")
    plt.plot(list(np.arange(len(accuracy_dict_right))),list(accuracy_dict_right.values()), label="right turn")
    plt.plot(list(np.arange(len(accuracy_dict_merged))),list(accuracy_dict_merged.values()), label="all")
    plt.ylabel('accuracy')
    plt.title(l3_eq)
    assert all([x1==x2 for x1,x2 in zip(accuracy_dict_left.keys(),accuracy_dict_right.keys())])
    plt.xticks(list(np.arange(len(accuracy_dict_left))), list(accuracy_dict_left.keys()), rotation=45)
    plt.legend(loc="upper left")
    plt.show()
    plt.figure()
    plt.plot(list(np.arange(len(lm_tp))),lm_tp, label="left true pos")
    plt.plot(list(np.arange(len(lm_tn))),lm_tn, label="left true neg")
    plt.plot(list(np.arange(len(lm_fp))),lm_fp, label="left false pos")
    plt.plot(list(np.arange(len(lm_fn))),lm_fn, label="left false neg")
    plt.plot(list(np.arange(len(rm_tp))),rm_tp, label="right true pos")
    plt.plot(list(np.arange(len(rm_tn))),rm_tn, label="right true neg")
    plt.plot(list(np.arange(len(rm_fp))),rm_fp, label="right false pos")
    plt.plot(list(np.arange(len(rm_fn))),rm_fn, label="right false neg")
    plt.legend(loc="upper left")
    plt.title(l3_eq)
    plt.xticks(list(np.arange(len(rm_fn))), list(precision_dict.keys()), rotation=45)
    plt.show()
    f=1
    
def analyse_turning_utilities(l3_eq,weighted,diff_plots,old_models):
    true_turn_utils = {'left':[],'right':[]}
    model_turn_utils = dict()
    for l3_eq in ['BASELINE','BOUNDARY','GAUSSIAN']:
        for file_id in constants.ALL_FILE_IDS: 
            #if int(file_id) > 769:
            #    continue
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            q_string = "select * from L1_SOLUTIONS"
            c.execute(q_string)
            res = c.fetchall()  
            ag_info_map = utils.load_agent_info_map(file_id)  
            
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                print(file_id,'processing',ct,N)
                model_type = row[3]
                model_params = row[4].split(',')
                model_params = {x.split('=')[0]:x.split('=')[1] for x in model_params}
                model_str = model_type
                if model_params['l3_sampling'] != l3_eq:
                    continue
                if ('old_model' not in model_params and old_models) or ( 'old_model' in model_params and model_params['old_model'] != 'True' and old_models):
                        continue
                if old_models:
                    if model_type in ['QlkR','brtb']:
                        continue
                    if model_params['baseline_weights'] != 'True':
                        continue
                
                if model_type == 'Ql1':
                    model_str = model_str + '_' + model_params['lzero_behavior']
                model_str = model_str + '_' + l3_eq + '_' + model_params['l3_soln']
                ag_vect = ag_info_map[(row[0],row[1],row[2])] if (row[0],row[1],row[2]) in ag_info_map else None
                if ag_vect is None:
                    continue
                soln_strats = ast.literal_eval(row[7])
                rule_strats = ast.literal_eval(row[6])
                emp_strats = ast.literal_eval(row[5])
                if len(soln_strats) == 0:
                    continue
                ag_idx = None
                ag_id = row[1]
                if len(soln_strats) != 0:
                    _s = soln_strats[0]
                    for idx,act in enumerate(list(_s)):
                        act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                        if ag_id == act_ag_id:
                            ag_idx = idx
                            break
                else:
                    if len(emp_strats) != 0:
                        _s = emp_strats[0]
                        for idx,act in enumerate(list(_s)):
                            act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                            if ag_id == act_ag_id:
                                ag_idx = idx
                                break
                if ag_idx is None:
                    sys.exit('something wrong')
                if len(soln_strats) == 0:
                    continue
                ag_soln_acts = list(set([int(x[ag_idx][9:11]) for x in soln_strats]))
                ag_emp_acts = list(set([int(x[ag_idx][9:11]) for x in emp_strats]))
                ag_soln_utils, ag_emp_utils, weights = None, None, None
                if row[8] is not None and json.loads(row[8]) is not None:
                    ag_emp_utils = [np.asarray(x)[:,ag_idx] if not old_models else np.asarray(x)[ag_idx] for x in json.loads(row[8])]
                if row[10] is not None and json.loads(row[10]) is not None:
                    ag_soln_utils = [np.asarray(x)[:,ag_idx] if not old_models else np.asarray(x)[ag_idx] for x in json.loads(row[10])]
                if row[11] is not None and json.loads(row[11]) is not None:
                    weights = json.loads(row[11])
                    low_weights = [weights[i] for i in (4,5,6)]
                    high_weights = [weights[i] for i in (0,1,2)]
                
                if constants.SEGMENT_MAP[ag_vect[3]] in ['prep-right-turn','exec-right-turn']:
                    if model_str not in model_turn_utils:
                        model_turn_utils[model_str] = {'left':[],'right':[]}
                    if not diff_plots:
                        for act_idx, act in enumerate(ag_soln_acts):
                            if act in [constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]:
                                if weighted:
                                    act_util_low = low_weights @ ag_soln_utils[act_idx]
                                    model_turn_utils[model_str]['right'].append(act_util_low)
                                    act_util_high = high_weights @ ag_soln_utils[act_idx]
                                    model_turn_utils[model_str]['right'].append(act_util_high) 
                                else:
                                    act_util = min(np.take(ag_soln_utils[act_idx], [0,2]))
                                    model_turn_utils[model_str]['right'].append(act_util)
                        for act_idx, act in enumerate(ag_emp_acts):
                            if act in [constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]:
                                if weighted:
                                    act_util_low = low_weights @ ag_emp_utils[act_idx]
                                    true_turn_utils['right'].append(act_util_low)
                                    act_util_high = high_weights @ ag_emp_utils[act_idx]
                                    true_turn_utils['right'].append(act_util_high)
                                else:
                                    act_util = min(np.take(ag_emp_utils[act_idx], [0,2]))
                                    true_turn_utils['right'].append(act_util)
                    else:
                        for sol_act_idx,sol_act in enumerate(ag_soln_acts):
                            for emp_act_idx,emp_act in enumerate(ag_emp_acts):
                                if not old_models:
                                    emp_act_util_low = low_weights @ ag_emp_utils[emp_act_idx]
                                    emp_act_util_high = high_weights @ ag_emp_utils[emp_act_idx]
                                    this_emp_act_util = np.mean([emp_act_util_low,emp_act_util_high])
                                    sol_act_util_low = low_weights @ ag_soln_utils[sol_act_idx]
                                    sol_act_util_high = high_weights @ ag_soln_utils[sol_act_idx]
                                    this_sol_act_util = np.mean([sol_act_util_low,sol_act_util_high])
                                    util_diff = this_emp_act_util-this_sol_act_util
                                else:
                                    this_emp_act_util = ag_emp_utils[emp_act_idx]
                                    this_soln_act_util = ag_soln_utils[sol_act_idx]
                                    util_diff = this_emp_act_util-this_soln_act_util if this_emp_act_util is not None and this_soln_act_util is not None else None
                                if util_diff is not None:
                                    model_turn_utils[model_str]['right'].append(util_diff)
                                
                elif constants.SEGMENT_MAP[ag_vect[3]] in ['prep-left-turn','exec-left-turn']:
                    if model_str not in model_turn_utils:
                        model_turn_utils[model_str] = {'left':[],'right':[]}
                    if not diff_plots:
                        for act_idx, act in enumerate(ag_soln_acts):
                            if act in [constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]:
                                if weighted:
                                    act_util_low = low_weights @ ag_soln_utils[act_idx]
                                    model_turn_utils[model_str]['left'].append(act_util_low)
                                    act_util_high = high_weights @ ag_soln_utils[act_idx]
                                    model_turn_utils[model_str]['left'].append(act_util_high)
                                else:
                                    act_util = min(np.take(ag_soln_utils[act_idx], [0,2]))
                                    model_turn_utils[model_str]['left'].append(act_util)
                        for act_idx, act in enumerate(ag_emp_acts):
                            if act in [constants.L1_ACTION_CODES[x] for x in constants.PROCEED_ACTIONS]:
                                if weighted:
                                    act_util_low = low_weights @ ag_emp_utils[act_idx]
                                    true_turn_utils['left'].append(act_util_low)
                                    act_util_high = high_weights @ ag_emp_utils[act_idx]
                                    true_turn_utils['left'].append(act_util_high)
                                else:
                                    act_util = min(np.take(ag_emp_utils[act_idx], [0,2]))
                                    true_turn_utils['left'].append(act_util)
                    else:
                        for sol_act_idx,sol_act in enumerate(ag_soln_acts):
                            for emp_act_idx,emp_act in enumerate(ag_emp_acts):
                                if not old_models:
                                    emp_act_util_low = low_weights @ ag_emp_utils[emp_act_idx]
                                    emp_act_util_high = high_weights @ ag_emp_utils[emp_act_idx]
                                    this_emp_act_util = np.mean([emp_act_util_low,emp_act_util_high])
                                    sol_act_util_low = low_weights @ ag_soln_utils[sol_act_idx]
                                    sol_act_util_high = high_weights @ ag_soln_utils[sol_act_idx]
                                    this_sol_act_util = np.mean([sol_act_util_low,sol_act_util_high])
                                    util_diff = this_emp_act_util-this_sol_act_util
                                    
                                else:
                                    this_emp_act_util = ag_emp_utils[emp_act_idx]
                                    this_soln_act_util = ag_soln_utils[sol_act_idx]
                                    util_diff = this_emp_act_util-this_soln_act_util if this_emp_act_util is not None and this_soln_act_util is not None else None
                                if util_diff is not None:
                                    model_turn_utils[model_str]['left'].append(util_diff)
                else:
                    continue
    if not diff_plots:
        left_turn_utils = [('true_util',[x for x in true_turn_utils['left'] if not np.isnan(x)])]
        right_turn_utils = [('true_util',[x for x in true_turn_utils['right'] if not np.isnan(x)])]
    else:
        left_turn_utils = []
        right_turn_utils = []
        all_util_diffs = []
    for k,v in model_turn_utils.items():
        left_turn_utils.append((k,v['left']))
        right_turn_utils.append((k,v['right']))
        all_util_diffs.append((k,v['right']+v['left']))
    fig = plt.figure()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.suptitle(l3_eq)
    ax1 = fig.add_subplot(211)
    data_plot = [[y for y in x[1] if not np.isnan(y)] for x in left_turn_utils]
    data_labels = [x[0] for x in left_turn_utils]
    bp = ax1.boxplot(data_plot,showfliers=False)
    ax1.set_xticklabels(data_labels)
    ax1.title.set_text('left turn')
    ax2 = fig.add_subplot(212)
    data_plot = [[y for y in x[1] if not np.isnan(y)] for x in right_turn_utils]
    data_labels = [x[0] for x in right_turn_utils]
    bp = ax2.boxplot(data_plot,showfliers=False)
    ax2.set_xticklabels(data_labels)
    ax2.title.set_text('right turn')
    if diff_plots:
        fig = plt.figure()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.suptitle(l3_eq)
        ax1 = fig.add_subplot(111)
        data_plot = [[y for y in x[1] if not np.isnan(y)] for x in all_util_diffs]
        data_labels = [x[0] for x in all_util_diffs]
        bp = ax1.boxplot(data_plot,showfliers=False)
        ax1.set_xticklabels(data_labels)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        ax1.title.set_text('all util diffs')
    plt.show()
    f=1

def analyse_weight_distribution(l3_eq):
    weights_dict_right,weights_dict_left = dict(),dict()
    for file_id in constants.ALL_FILE_IDS[0:5]: 
        #if int(file_id) > 769:
        #    continue
        model_type = l3_eq
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        q_string = "select * from UTILITY_WEIGHTS WHERE L3_MODEL_TYPE='"+model_type+"'"
        c.execute(q_string)
        res = c.fetchall()  
        
        ct,N = 0,len(res)
        for row in res:
            ct += 1
            print(file_id,'processing',ct,N)
            if row[12] != l3_eq:
                continue
            model_str = row[11]
            if model_str not in weights_dict_right:
                weights_dict_right[model_str] = {'vi_h':[],'vi_l':[],'pi_h':[],'pi_l':[],'rl_h':[],'rl_l':[],'ex_h':[],'ex_l':[]}
            if model_str not in weights_dict_left:
                weights_dict_left[model_str] = {'vi_h':[],'vi_l':[],'pi_h':[],'pi_l':[],'rl_h':[],'rl_l':[],'ex_h':[],'ex_l':[]}
            
            if constants.SEGMENT_MAP[row[6]] in ['prep-right-turn','exec-right-turn']:
                weights_dict_right[model_str]['vi_h'].append(row[13])
                weights_dict_right[model_str]['ex_h'].append(row[14])
                weights_dict_right[model_str]['pi_h'].append(row[15])
                weights_dict_right[model_str]['rl_h'].append(row[16])
                weights_dict_right[model_str]['vi_l'].append(row[17])
                weights_dict_right[model_str]['ex_l'].append(row[18])
                weights_dict_right[model_str]['pi_l'].append(row[19])
                weights_dict_right[model_str]['rl_l'].append(row[20])
            elif constants.SEGMENT_MAP[row[6]] in ['prep-left-turn','exec-left-turn']:
                weights_dict_left[model_str]['vi_h'].append(row[13])
                weights_dict_left[model_str]['ex_h'].append(row[14])
                weights_dict_left[model_str]['pi_h'].append(row[15])
                weights_dict_left[model_str]['rl_h'].append(row[16])
                weights_dict_left[model_str]['vi_l'].append(row[17])
                weights_dict_left[model_str]['ex_l'].append(row[18])
                weights_dict_left[model_str]['pi_l'].append(row[19])
                weights_dict_left[model_str]['rl_l'].append(row[20])
            else:
                continue
    for k in list(weights_dict_left.keys()):
        for k1 in list(weights_dict_left[k].keys()):
            v1 = [x for x in weights_dict_left[k][k1] if x is not None]
            v1 = np.mean(v1)
            weights_dict_left[k][k1] = v1
    for k in list(weights_dict_right.keys()):
        for k1 in list(weights_dict_right[k].keys()):
            v1 = [x for x in weights_dict_right[k][k1] if x is not None]
            v1 = np.mean(v1)
            weights_dict_right[k][k1] = v1
    
    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
    plt.suptitle(l3_eq+'_'+'left')
    for i, ax in enumerate(axes.flatten()):
        if i >= len(weights_dict_left):
            continue
        model_str = list(weights_dict_left.keys())[i]
        ax.title.set_text(model_str)
        ax.bar(['vi','pi','ex','rl'],[weights_dict_left[model_str]['vi_h']-weights_dict_left[model_str]['vi_l'],weights_dict_left[model_str]['pi_h']-weights_dict_left[model_str]['pi_l'],
                                      weights_dict_left[model_str]['ex_h']-weights_dict_left[model_str]['ex_l'],0.1],
               bottom = [weights_dict_left[model_str]['vi_l'],weights_dict_left[model_str]['pi_l'],weights_dict_left[model_str]['ex_l'],weights_dict_left[model_str]['rl_l']])
    
    
    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
    plt.suptitle(l3_eq+'_'+'right')
    for i, ax in enumerate(axes.flatten()):
        if i >= len(weights_dict_right):
            continue
        
        model_str = list(weights_dict_right.keys())[i]
        if model_str == 'CorrEq':
            continue
        ax.title.set_text(model_str)
        ax.bar(['vi','pi','ex','rl'],[weights_dict_right[model_str]['vi_h']-weights_dict_right[model_str]['vi_l'],weights_dict_right[model_str]['pi_h']-weights_dict_right[model_str]['pi_l'],
                                      weights_dict_right[model_str]['ex_h']-weights_dict_right[model_str]['ex_l'],0.1],
               bottom = [weights_dict_right[model_str]['vi_l'],weights_dict_right[model_str]['pi_l'],weights_dict_right[model_str]['ex_l'],weights_dict_right[model_str]['rl_l']])
    
    
    plt.show()
    f=1
    
def old_filename(file_id,model_str):
    chunks = model_str.split('_')
    if chunks[-1] == 'NA':
        chunks[-2] = 'BASELINE_ONLY'
        chunks = chunks[:-1]
    model = []
    l1_map = {'NASH':'NASH',
              'Ql1_maxmin':'L1MAXMIN',
              'Ql1_maxmax':'L1BR',
              'maxmax':'BR',
              'maxmin':'MAXMIN'}
    if chunks[0] == 'Ql1':
        l1_chunk = l1_map[chunks[0]+'_'+chunks[1]]
        if 'BASELINE_ONLY' in chunks[2:]:
            l2_chunk = ''.join(chunks[2:])
        else:
            l2_chunk = ','.join(chunks[2:][::-1])
    else:
        l1_chunk = l1_map[chunks[0]]
        if 'BASELINE_ONLY' in chunks[1:]:
            l2_chunk = ''.join(chunks[1:])
        else:
            l2_chunk = ','.join(chunks[1:][::-1])
    filename_str = file_id + '_' + ','.join([l1_chunk,l2_chunk]) + '_u_deltas.csv'
    return filename_str
    

def generate_results_file(l3_eq,weighted,diff_plots,old_models):
    given_file_id = sys.argv[1]
    old_results_dir = "G:\\AAAI_submission_764_code\\results"
    new_results_dir = "F:\\Spring2017\\workspaces\\game_theoretic_planner\\results_all"
    true_turn_utils = {'left':[],'right':[]}
    model_turn_utils = dict()
    direction_map = utils.load_direction_map()
    all_models = []
    output_dict = dict()
    field_names = ['EQ_TYPE','SEGMENT','TASK','NEXT_CHANGE','SPEED','PEDESTRIAN','RELEV_VEHICLE','LEAD_VEHICLE','AGGRESSIVE','FILE_ID','TRACK_ID','TIME','ON_EQ']
    for l3_eq in ['BASELINE','BOUNDARY','GAUSSIAN']:
        if l3_eq == 'BASELINE':
            is_transformed = False
        else:
            is_transformed = True
        for file_id in constants.ALL_FILE_IDS: 
            if file_id != given_file_id:
                continue
            output_dict = dict()
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            q_string = "select * from L1_SOLUTIONS"
            c.execute(q_string)
            res = c.fetchall()  
            ag_info_map = utils.load_agent_info_map(file_id)  
            
            
            '''
            ag_info_map = dict()
            file_keys = os.listdir(old_results_dir)
            for f in file_keys:
                if f.split('_')[0] == file_id:
                    with open(os.path.join(old_results_dir,f), newline='\n') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if (row['TRACK_ID'],row['TIME']) not in ag_info_map:
                                ag_info_map[(row['TRACK_ID'],row['TIME'])] = [(row['SEGMENT'],row['TASK'],row['NEXT_CHANGE'],row['SPEED'],row['PEDESTRIAN'],row['RELEV_VEHICLE'],row['ACTIONS'],row['AGGRESSIVE'],row['FILE_ID'],row['TRACK_ID'],row['TIME'])]
                            else:
                                ag_info_map[(row['TRACK_ID'],row['TIME'])].append((row['SEGMENT'],row['TASK'],row['NEXT_CHANGE'],row['SPEED'],row['PEDESTRIAN'],row['RELEV_VEHICLE'],row['ACTIONS'],row['AGGRESSIVE'],row['FILE_ID'],row['TRACK_ID'],row['TIME']))
            '''    
                     
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                print(file_id,'processing',ct,N,l3_eq)
                model_type = row[3]
                model_params = row[4].split(',')
                model_params = {x.split('=')[0]:x.split('=')[1] for x in model_params}
                model_str = model_type
                time_ts = row[2]
                
                if model_params['l3_sampling'] != l3_eq:
                    continue
                if ('old_model' not in model_params and old_models) or ( 'old_model' in model_params and model_params['old_model'] != 'True' and old_models):
                        continue
                if old_models:
                    if model_type in ['QlkR','brtb']:
                        continue
                    if model_params['baseline_weights'] != 'True':
                        continue
                
                if model_type == 'Ql1':
                    model_str = model_str + '_' + model_params['lzero_behavior']
                model_str = model_str + '_' + l3_eq + '_' + model_params['l3_soln']
                old_name = old_filename(file_id, model_str)
                l3_cache_str = constants.CACHE_DIR+"l3_trees_"+l3_eq+"_"+model_params['l3_soln']+"\\"+file_id
                file_str = str(row[1])+'_'+str(row[2]).replace('.', ',')
                if model_type == 'maxmin':
                    f=1
                payoff_dict = None
                if not os.path.exists(os.path.join(l3_cache_str,file_str)):
                    continue
                    '''
                    file_keys = os.listdir(l3_cache_str)
                    for f in file_keys:
                        if f.split(',')[0] == file_str.split(',')[0]:
                            payoff_dict = utils.pickle_load(os.path.join(l3_cache_str,f))
                            break
                    '''
                else:
                    payoff_dict = utils.pickle_load(os.path.join(l3_cache_str,file_str))
                if payoff_dict is None:
                    continue
                
                if not os.path.exists(os.path.join(old_results_dir,old_name)):
                    sys.exit()
                
                if not is_transformed:
                    payoff_dict = {k:np.asarray([0.25,0.25,0.5])@v for k,v in payoff_dict.items()}
                
                ag_vect = ag_info_map[(row[0],row[1],row[2])] if (row[0],row[1],row[2]) in ag_info_map else None
                if ag_vect is None:
                    continue
                soln_strats = ast.literal_eval(row[7])
                all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
                num_players = len(all_agents)
                player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]      
                ag_id,ag_idx = None,None
                for idx,ag in enumerate(all_agents):
                    if ag[1] == 0:
                        ag_id = ag[0]
                        ag_idx = idx
                        break
                         
                print('calculating soln',len(payoff_dict))
                if model_type == 'Ql1':
                    eq = equilibria_core.EquilibriaCore(num_players,payoff_dict,len(payoff_dict),player_actions[ag_idx],True)
                    soln = eq.calc_best_response() if model_params['lzero_behavior'] == 'maxmax' else eq.calc_max_min_response()
                elif model_type == 'NASH':
                    eq = equilibria_core.EquilibriaCore(num_players,payoff_dict,len(payoff_dict),player_actions[ag_idx],False)
                    soln = eq.calc_pure_strategy_nash_equilibrium_exhaustive()
                else:
                    eq = equilibria_core.EquilibriaCore(num_players,payoff_dict,len(payoff_dict),player_actions[ag_idx],False)
                    if model_type == 'maxmin':
                        soln = eq.calc_max_min_response()
                    else:
                        soln = eq.calc_best_response()
                soln_strats = list(soln.keys())
                rule_strats = ast.literal_eval(row[6])
                emp_strats = ast.literal_eval(row[5])
                if len(soln_strats) == 0:
                    continue
                '''
                ag_idx = None
                ag_id = row[1]
                all_agents_from_row = [(int(x[3:6]), int(x[6:9])) for x in soln_strats[0]]
                if (ag_id,0) not in all_agents_from_row:
                    continue
                if len(soln_strats) != 0:
                    _s = soln_strats[0]
                    for idx,act in enumerate(list(_s)):
                        act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                        if ag_id == act_ag_id:
                            ag_idx = idx
                            break
                else:
                    if len(emp_strats) != 0:
                        _s = emp_strats[0]
                        for idx,act in enumerate(list(_s)):
                            act_ag_id = int(act[3:6]) if int(act[6:9]) == 0 else int(act[6:9])
                            if ag_id == act_ag_id:
                                ag_idx = idx
                                break
                '''
                if ag_idx is None:
                    sys.exit('something wrong')
                payoff_diff_list = []
                
                if ag_idx >= len(emp_strats):
                    print(ag_idx,len(emp_strats),'continuing')
                    continue
                if model_type == 'NASH':
                    for e_s in emp_strats:
                        for s_s in soln_strats:
                            _s = list(s_s)
                            _s[ag_idx] = e_s[ag_idx]
                            if tuple(_s) not in payoff_dict:
                                continue
                            emp_agent_payoff = payoff_dict[tuple(_s)][ag_idx]
                            soln_payoff = payoff_dict[s_s][ag_idx]
                            payoff_diff_list.append(soln_payoff-emp_agent_payoff)
                            if soln_payoff<emp_agent_payoff:
                                f=1 
                            
                elif model_type == 'Ql1':
                    #q_string = "SELECT * FROM L1_SOLUTIONS WHERE MODEL='"+model_params['lzero_behavior']+"' and L1_SOLUTIONS.MODEL_PARMS LIKE '%old_model=True%' and TRACK_ID="+str(ag_id)+" and TIME="+str(time_ts)
                    #c.execute(q_string)
                    #lzero_res =  c.fetchone() 
                    #lzero_solns = ast.literal_eval(lzero_res[7])
                    all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
                    num_players = len(all_agents)
                    player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]           
                    eq = equilibria_core.EquilibriaCore(num_players,payoff_dict,len(payoff_dict),player_actions[ag_idx],False)
                    if model_params['lzero_behavior'] == 'maxmin':
                        soln = eq.calc_max_min_response()
                    else:
                        soln = eq.calc_best_response()
                    lzero_solns = list(soln.keys())
                    ag_solns,ag_emp_acts = list(set([x[ag_idx] for x in soln_strats])),list(set([x[ag_idx] for x in emp_strats]))
                    ql1_emp_strats,ql1_soln_strats = [],[]
                    for lzs in lzero_solns:
                        for ags in ag_emp_acts:
                            _s = list(lzs)
                            _s[ag_idx] = ags
                            ql1_emp_strats.append(tuple(_s))
                        for ags in ag_solns:
                            _s = list(lzs)
                            _s[ag_idx] = ags
                            ql1_soln_strats.append(tuple(_s))
                    for e_s in ql1_emp_strats:
                        for s_s in ql1_soln_strats:
                            _s = list(s_s)
                            _s[ag_idx] = e_s[ag_idx]
                            if tuple(_s) not in payoff_dict:
                                log.warning('strategy '+str(_s)+ ' not in table. continuing.....')
                                continue
                            emp_agent_payoff = payoff_dict[tuple(_s)][ag_idx]
                            soln_payoff = payoff_dict[s_s][ag_idx]
                            payoff_diff_list.append(soln_payoff-emp_agent_payoff)    
                            if soln_payoff<emp_agent_payoff:
                                f=1
                else:
                    all_agents = [(int(x[3:6]), int(x[6:9])) for x in list(payoff_dict.keys())[0]]
                    num_players = len(all_agents)
                    player_actions = [list(set([k[i] for k in payoff_dict.keys()])) for i in np.arange(num_players)]           
                    ag_solns,ag_emp_acts = list(set([x[ag_idx] for x in soln_strats])),list(set([x[ag_idx] for x in emp_strats]))
                    
                    for ags in ag_solns:
                        for ages in ag_emp_acts:
                            if ags not in player_actions[ag_idx] or ages not in player_actions[ag_idx]:
                                log.warning('action '+str(ags)+ ' not in table. continuing.....')
                                continue
                            replaced_p_a = list(player_actions) 
                            replaced_p_a[ag_idx] = [None]
                            ql0_emp_strats = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                            for idx_2,s in enumerate(ql0_emp_strats):
                                _s = list(s)
                                _s[ag_idx] = ages
                                ql0_emp_strats[idx_2] = tuple(_s)
                            ql0_soln_strats = list(set(itertools.product(*[v for idx_2,v in enumerate(replaced_p_a)])))
                            for idx_2,s in enumerate(ql0_soln_strats):
                                _s = list(s)
                                _s[ag_idx] = ags
                                ql0_soln_strats[idx_2] = tuple(_s)
                            if model_type == 'maxmax':
                                emp_agent_payoff = max([payoff_dict[x][ag_idx] for x in ql0_emp_strats])
                                soln_payoff = max([payoff_dict[x][ag_idx] for x in ql0_soln_strats])
                            else:
                                emp_agent_payoff = min([payoff_dict[x][ag_idx] for x in ql0_emp_strats])
                                soln_payoff = min([payoff_dict[x][ag_idx] for x in ql0_soln_strats])
                            payoff_diff_list.append(soln_payoff-emp_agent_payoff)
                            if soln_payoff<emp_agent_payoff:
                                f=1
                if len(payoff_diff_list) == 0:
                    continue
                util_diff = min(payoff_diff_list)
                if constants.SEGMENT_MAP[ag_vect[3]] in ['prep-right-turn','exec-right-turn','right-turn-lane','exit-lane']:
                    if model_str not in model_turn_utils:
                        model_turn_utils[model_str] = {'left':[],'right':[]}
                    if util_diff is not None:
                        model_turn_utils[model_str]['right'].append(util_diff)
                                
                elif constants.SEGMENT_MAP[ag_vect[3]] in ['prep-left-turn','exec-left-turn','left-turn-lane','exit-lane']:
                    if model_str not in model_turn_utils:
                        model_turn_utils[model_str] = {'left':[],'right':[]}
                    if util_diff is not None:
                        model_turn_utils[model_str]['left'].append(util_diff)
                else:
                    continue
                print(ag_vect,util_diff)
                if old_name not in output_dict:
                    output_dict[old_name] = {k:[] for k in field_names}
                if 'BASELINE' in old_name.split('_')[1]:
                    entry_model_name = old_name.split('_')[1]+'_ONLY'
                else:
                    entry_model_name = old_name.split('_')[1]
                entry_model_name = entry_model_name.replace(',','|')
                entry = [entry_model_name,constants.SEGMENT_MAP[ag_vect[3]],direction_map[file_id][ag_id],ag_vect[2]+'-'+ag_vect[1] if ag_vect[1] is not None and ag_vect[2] is not None else None,ag_vect[4],ag_vect[7],ag_vect[6],ag_vect[5],'deprecated',file_id,ag_id,time_ts,util_diff]
                for i,e in enumerate(entry):
                    output_dict[old_name][field_names[i]].append(e)
                
            f=1
            
            for k,v in output_dict.items():
                with open(os.path.join(new_results_dir,k), 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                    for idx in np.arange(len(v[field_names[0]])):
                        writer.writerow({e_k:v[e_k][idx] for e_k in field_names})
            
            
        
    all_util_diffs = []
    for k,v in model_turn_utils.items():
        all_util_diffs.append((k,v['right']+v['left']))
    fig = plt.figure()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.suptitle(l3_eq)
    ax1 = fig.add_subplot(111)
    data_plot = [[y for y in x[1] if not np.isnan(y)] for x in all_util_diffs]
    data_labels = [x[0] for x in all_util_diffs]
    bp = ax1.boxplot(data_plot,showfliers=False)
    ax1.set_xticklabels(data_labels)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    ax1.title.set_text('all util diffs')
    plt.show()
    f=1   
    
def get_newname(old_name):
    newname = []
    if old_name[0:2] == 'L1':
        newname.append('Ql1')
    elif old_name[0:4] == 'NASH':
        newname.append('PNE_QE')
    else:
        newname.append('Ql0')
    old_name = old_name.split('|')
    if newname[0] == 'Ql1':
        if old_name[0][2:] == 'BR':
            newname.append('MX')
        else:
            newname.append('MM')
    elif newname[0] == 'Ql0':
        if old_name[0][0:2] == 'BR':
            newname.append('MX')
        else:
            newname.append('MM')
    if old_name[1] != 'BASELINE_ONLY':
        if old_name[1][0:2] == 'BR':
            newname.append('MX')
        else:
            newname.append('MM')
        if old_name[-1] == 'GAUSSIAN':
            newname.append('S(1+G)')
        else:
            newname.append('S(1+B)')
               
    else:
        newname.append('S(1)')
    return ' '.join(newname)
        
        

def plot_util_hist():
    #mpl.style.use('ggplot')
    old_results_dir = "G:\\AAAI_submission_764_code\\results"
    #new_results_dir = "F:\\Spring2017\\workspaces\\game_theoretic_planner\\results_all"
    true_turn_utils = {'left':[],'right':[]}
    model_turn_utils = dict()
    direction_map = utils.load_direction_map()
    all_models,sit_distr = OrderedDict(), dict()
    files = [f for f in os.listdir(old_results_dir) if 'csv' in f]
    headers = ['SEGMENT','TASK','NEXT_CHANGE','SPEED','PEDESTRIAN','RELEV_VEHICLE','AGGRESSIVE']
    for f in files:
        print(f)
        with open(os.path.join(old_results_dir,f), newline='\n') as csvfile:
            datareader = csv.DictReader(csvfile)
            for row in datareader:
                if get_newname(row['EQ_TYPE']) not in all_models:
                    all_models[get_newname(row['EQ_TYPE'])] = []
                all_models[get_newname(row['EQ_TYPE'])].append(float(row['ON_EQ']))
                for h in headers:
                    if h not in sit_distr:
                        sit_distr[h] = dict()
                    if row[h] not in sit_distr[h]: 
                        sit_distr[h][row[h]] = dict()
                    if get_newname(row['EQ_TYPE']) not in sit_distr[h][row[h]]:
                        sit_distr[h][row[h]][get_newname(row['EQ_TYPE'])] = []
                    sit_distr[h][row[h]][get_newname(row['EQ_TYPE'])].append(float(row['ON_EQ']))
                
    all_models = OrderedDict(sorted(all_models.items(), key=lambda x:x[0], reverse=True))            
    print('*****MLE Estimates*****')
            
            
    mle_estimates = dict()
    for k,v in all_models.items():
        mle_estimates[k] = len(v)/sum(v)
        #mle_estimates[k] = stat_utils.fit_exponential(np.asarray(v))
    for k,v in mle_estimates.items():
        print(k,v)       
    
    fig = plt.figure()
    plt.gcf().subplots_adjust(bottom=0.15)
    '''
    ax1 = fig.add_subplot(211)
    '''
    data_plot = [[y for y in x if not np.isnan(y)] for x in all_models.values()]
    data_labels = [x for x in list(all_models.keys())]
    print(','.join(data_labels))
    doc = Document("multirow")
    section = Section('Multirow Test')
    test2 = Subsection('MultiRow')
    col_head = '|'.join(['c']+['p{.76cm}']*20+['p{1cm}']*5)
    table2 = Tabular(col_head,booktabs=True)
    table2.add_row(tuple(['']+ [x for x in data_labels]))
    for s in sit_distr.keys():
        if s == 'TASK':
            continue
        
        table2.add_hline()
        est_vect = []
        for m in data_labels:
            if sum(sit_distr[s][list(sit_distr[s].keys())[0]][m]) == 0 and len(sit_distr[s][list(sit_distr[s].keys())[0]][m]) > 0:
                est_vect.append(np.inf)
            elif sum(sit_distr[s][list(sit_distr[s].keys())[0]][m]) != 0:
                est_vect.append(round(len(sit_distr[s][list(sit_distr[s].keys())[0]][m])/sum(sit_distr[s][list(sit_distr[s].keys())[0]][m]),1))
            else:
                est_vect.append(np.nan)
        table2.add_row(tuple([s]+ ['']*25))
        for idx,sval in enumerate(list(sit_distr[s].keys())):
            if sval == 'HIGH  SPEED':
                continue
            if idx >= 0:
                est_vect = []
                for m in data_labels:
                    if sum(sit_distr[s][sval][m]) == 0 and len(sit_distr[s][sval][m]) > 0:
                        est_vect.append(np.inf)
                    elif sum(sit_distr[s][sval][m]) != 0:
                        est_vect.append(round(len(sit_distr[s][sval][m])/sum(sit_distr[s][sval][m]),1))
                    else:
                        est_vect.append(np.nan)
                max_idx = est_vect.index(max(est_vect))
                table2.add_row(tuple([sval]+[str(x) if i!= max_idx else pylatex.utils.bold(str(x)+'*') for i,x in enumerate(est_vect)]))
    test2.append(table2)
    section.append(test2)
    doc.append(section)
    doc.generate_pdf(clean_tex=False,compiler='pdfLaTeX')   
            
    '''      
    for idx,m in enumerate(list(all_models.keys())):
        plt.text(idx+1, 0.00165, str(round(mle_estimates[m],1)), rotation=90, va='center')
    bp = ax1.boxplot(data_plot,showfliers=False,patch_artist=True)
    colors = ['cyan']*10+ ['tan']*10+ ['pink']*5
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.setp(bp['medians'], color='black')
    
    ax1.set_xticklabels([])
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    '''
    ax2 = fig.add_subplot(111)
    for idx,m in enumerate(list(all_models.keys())):
        plt.text(idx+.75, 0.00165, str(round(mle_estimates[m],1)), rotation=90, va='center')
    bp = ax2.boxplot(data_plot,showfliers=False,patch_artist=True)
    colors = ['#377eb8']*10+ ['#ff7f00']*10+ ['#4daf4a']*5
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.setp(bp['medians'], color='black')
    
    ax2.set_xticklabels([x for x in data_labels],ha='right')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    plt.show()

def print_rules_mismatch():
    def _rulematch(_manv,_rule_list):
        manv,rule_list = utils.get_l1_action_string(_manv),[utils.get_l1_action_string(x) for x in _rule_list]
        match = False
        for r in rule_list:
            if r != manv:
                if r in constants.WAIT_ACTIONS and manv in constants.WAIT_ACTIONS:
                    match = True
                    break
            else:
                match = True
                break
        return match
                    
    file_id = '769'
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
    c = conn.cursor()
    q_string = "select * from L1_SOLUTIONS WHERE MODEL_PARMS LIKE '%l3_sampling=BASELINEUTILS%'"
    c.execute(q_string)
    res = c.fetchall()  
    rule_mismatch_list = []
    N = len(res)
    okct = 0 
    for runidx,row in enumerate(res):
        print('processing',file_id,runidx,'/',N)
        rule_strat = ast.literal_eval(row[6])
        emp_strat = ast.literal_eval(row[5])
        if len(emp_strat) >0 and len(rule_strat) >0:
            num_agents = len(emp_strat[0])
            for agidx in np.arange(num_agents):
                rule_acts = []
                for strat in rule_strat:
                    _act = strat[agidx]
                    ag_id = int(_act[3:6]) if int(int(_act[6:9])) == 0 else int(_act[6:9])
                    if int(_act[9:11]) not in rule_acts:
                        rule_acts.append(int(_act[9:11]))
                for strat in emp_strat:
                    _act = strat[agidx]
                    ag_id = int(_act[3:6]) if int(int(_act[6:9])) == 0 else int(_act[6:9])
                    if not _rulematch(int(_act[9:11]),rule_acts):
                        if (file_id,ag_id,row[2]) not in rule_mismatch_list:
                            rule_mismatch_list.append((file_id,row[1],row[2],ag_id,utils.get_l1_action_string(int(_act[9:11])),[utils.get_l1_action_string(x) for x in rule_acts]))
                    else:
                        okct += 1
    for _e in rule_mismatch_list:
        print(_e)
    print(len(rule_mismatch_list),okct)
    
    
def print_conf_matrix_results(sampling_type):
    def _rulematch(_manv,_rule_list):
        manv,rule_list = utils.get_l1_action_string(_manv),[utils.get_l1_action_string(x) for x in _rule_list]
        match = False
        for r in rule_list:
            if r != manv:
                if r in constants.WAIT_ACTIONS and manv in constants.WAIT_ACTIONS:
                    match = True
                    break
            else:
                match = True
                break
        return match
    
    #file_id = '776'
    print('***************',sampling_type,'********************')  
    for model_type in [('NASH',None),('QlkR',None),('brtb',None),('maxmin',None),('maxmax',None),('Ql1','maxmin'),('Ql1','maxmax')]:
        confusion_dict,confusion_dict_N = dict(), dict()
        for file_id in [x for x in constants.ALL_FILE_IDS if int(x) > 775]:
            conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
            c = conn.cursor()
            
            if model_type[1] is None:
                q_string = "select * from L1_SOLUTIONS WHERE MODEL_PARMS LIKE '%l3_sampling="+sampling_type+",%' AND MODEL='"+model_type[0]+"'"
            else:
                q_string = "select * from L1_SOLUTIONS WHERE MODEL_PARMS LIKE '%l3_sampling="+sampling_type+",%lzero_behavior="+model_type[1]+"%' AND MODEL='"+model_type[0]+"'"
            
            #q_string = "select * from L1_SOLUTIONS WHERE MODEL_PARMS LIKE '%l3_sampling="+sampling_type+",%'"
            c.execute(q_string)
            res = c.fetchall()  
            rule_mismatch_list = []
            N = len(res)
            okct = 0 
            for runidx,row in enumerate(res):
                #print('processing',file_id,runidx,'/',N)
                soln_strat = ast.literal_eval(row[7])
                emp_strat = ast.literal_eval(row[5])
                row_agid = int(row[1])
                if len(emp_strat) >0 and len(soln_strat) >0:
                    num_agents = len(emp_strat[0])
                    if num_agents <2:
                        continue
                    agidx = None
                    for _idx,_es in enumerate(emp_strat[0]):
                        _thisagid = int(_es[3:6]) if int(_es[6:9]) == 0 else int(_es[6:9])
                        if _thisagid == row_agid:
                            agidx = _idx
                            break
                    soln_acts = list(set([int(x[agidx][9:11]) for x in soln_strat]))
                    emp_acts = list(set([int(x[agidx][9:11]) for x in emp_strat]))
                    for e_a in emp_acts:
                        if e_a not in confusion_dict:
                            confusion_dict[e_a] = []
                        if e_a not in confusion_dict_N:
                            confusion_dict_N[e_a] = 0
                        if e_a in soln_acts:
                            confusion_dict[e_a].append(e_a)
                            confusion_dict_N[e_a] += 1
                        else:
                            for s_a in soln_acts:
                                confusion_dict[e_a].append(s_a)
                            #since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization
                            confusion_dict[e_a] += [e_a]*(len(soln_acts)-1)
                            confusion_dict_N[e_a] += 1
        confusion_dict = {utils.get_l1_action_string(int(k)):[utils.get_l1_action_string(int(x)) for x in v] for k,v in confusion_dict.items()}
        confusion_dict_N = {utils.get_l1_action_string(int(k)):v for k,v in confusion_dict_N.items()}
        confusion_key_list = list(confusion_dict.keys())
        confusion_matrix = []
        for k in confusion_key_list:
            confusion_arr = []
            ctr = Counter(confusion_dict[k])
            _sum = sum(ctr.values())
            for p_k in confusion_key_list:
                th_prob = ctr[p_k]/_sum
                confusion_arr.append(th_prob)
            confusion_matrix.append(confusion_arr)
            
        df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(confusion_dict_N[x])+')' for x in confusion_key_list],
                      columns = confusion_key_list)
        plt.figure()
        chart = sn.heatmap(df_cm, annot=True)
        chart.figure.tight_layout()
        sn.set(font_scale=.50)
        plt.xticks(rotation=75)
        plt.yticks(rotation=0)
        plt.xlabel('Equilibrium action', labelpad=15)
        plt.ylabel('Empirical action')
        b, t = plt.ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        plt.ylim(b, t)
        model_str = sampling_type + '_' + model_type[0] if model_type[1] is None else sampling_type + '_' + model_type[0]+'-'+model_type[1]
        plt.savefig(model_str+'.png', bbox_inches='tight')
        #plt.show()
        
        print('-----------',model_type if model_type[1] is not None else model_type[0],'---------------')
        _sum = []
        for idx,e in enumerate(confusion_matrix):
            print(confusion_key_list[idx],e[idx])
            _sum.append(e[idx])
        print('mean:',np.mean(_sum))
        f=1
    print('***************',sampling_type,'********************')                   
                    
def print_rulemismatch_results():
    def _rulematch(_manv,_rule_list):
        if _manv == 5 and 2 in _rule_list:
            f=1
        manv,rule_list = utils.get_l1_action_string(_manv),[utils.get_l1_action_string(x) for x in _rule_list]
        match = False
        for r in rule_list:
            if r != manv:
                if r in constants.WAIT_ACTIONS and manv in constants.WAIT_ACTIONS:
                    match = True
                    break
            else:
                match = True
                break
        return match
    
    #file_id = '776'
    sampling_type = 'SAMPLING_EQ'
    
    confusion_dict,confusion_dict_N = dict(), dict()
    for file_id in [x for x in constants.ALL_FILE_IDS if int(x) > 775]:
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        
        q_string = "select * from L1_SOLUTIONS WHERE MODEL_PARMS LIKE '%l3_sampling="+sampling_type+",%'"
        c.execute(q_string)
        res = c.fetchall()  
        rule_mismatch_list = []
        N = len(res)
        okct = 0 
        for runidx,row in enumerate(res):
            print('processing',file_id,runidx,'/',N)
            rule_strat = ast.literal_eval(row[6])
            emp_strat = ast.literal_eval(row[5])
            row_agid = int(row[1])
            if len(emp_strat) >0 and len(rule_strat) >0:
                num_agents = len(emp_strat[0])
                if num_agents <2:
                    continue
                agidx = None
                for _idx,_es in enumerate(emp_strat[0]):
                    _thisagid = int(_es[3:6]) if int(_es[6:9]) == 0 else int(_es[6:9])
                    if _thisagid == row_agid:
                        agidx = _idx
                        break
                rule_acts = list(set([int(x[agidx][9:11]) for x in rule_strat]))
                emp_acts = list(set([int(x[agidx][9:11]) for x in emp_strat]))
                for e_a in emp_acts:
                    if e_a not in confusion_dict:
                        confusion_dict[e_a] = []
                    if e_a not in confusion_dict_N:
                        confusion_dict_N[e_a] = 0
                    if _rulematch(e_a, rule_acts):
                        confusion_dict[e_a].append(e_a)
                        confusion_dict_N[e_a] += 1
                    else:
                        for s_a in rule_acts:
                            confusion_dict[e_a].append(s_a)
                        #since the mis-prediction is multi-valued we need to add N-1 true values to the count for proper normalization
                        confusion_dict[e_a] += [e_a]*(len(rule_acts)-1)
                        confusion_dict_N[e_a] += 1
    confusion_dict = {utils.get_l1_action_string(int(k)):[utils.get_l1_action_string(int(x)) for x in v] for k,v in confusion_dict.items()}
    confusion_dict_N = {utils.get_l1_action_string(int(k)):v for k,v in confusion_dict_N.items()}
    confusion_key_list = list(confusion_dict.keys())
    confusion_matrix = []
    for k in confusion_key_list:
        confusion_arr = []
        ctr = Counter(confusion_dict[k])
        _sum = sum(ctr.values())
        for p_k in confusion_key_list:
            th_prob = ctr[p_k]/_sum
            confusion_arr.append(th_prob)
        confusion_matrix.append(confusion_arr)
    _sum = []
    for idx,e in enumerate(confusion_matrix):
        print(confusion_key_list[idx],e[idx])
        _sum.append(e[idx])
    print('mean:',np.mean(_sum))
    df_cm = pd.DataFrame(confusion_matrix, index = [x+'('+str(confusion_dict_N[x])+')' for x in confusion_key_list],
                  columns = confusion_key_list)
    plt.figure()
    chart = sn.heatmap(df_cm, annot=True)
    chart.figure.tight_layout()
    sn.set(font_scale=.50)
    plt.xticks(rotation=75)
    plt.yticks(rotation=0)
    plt.xlabel('Equilibrium action', labelpad=15)
    plt.ylabel('Empirical action')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)
    plt.show()
    
   
    f=1                   

if __name__ == '__main__':     
    #generate_results_file(None,True, True, True)
    #analyse_l1_precision_spec_ratio(None,True)
    #analyse_l1_precision_spec_ratio_eq_table(None,True)
    #analyse_turning_utilities(None,True, True, True)
    #analyse_weight_distribution('BASELINEUTILS')
    #analyse_weight_distribution('SAMPLING_EQ')
    #plot_util_hist()
    #print_conf_matrix_results('BASELINE')
    print_rulemismatch_results()
        
    
