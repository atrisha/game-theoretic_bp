'''
Created on Nov 27, 2020

@author: Atrisha
'''
import numpy as np
import sqlite3
from all_utils import utils
import ast
import sys
import constants
from scipy import stats
import matplotlib.pyplot as plt
import json


def analyse_l1_precision_spec_ratio(l3_eq):
    precision_dict = dict()
    accuracy_dict_left,accuracy_dict_right = dict(),dict()
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
            if model_type == 'Ql1':
                model_str = model_str + '_' + model_params['lzero_behavior']
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
        lm = round(np.mean([v['left']['tp']/(v['left']['tp']+v['left']['fp']), v['left']['tn']/(v['left']['tn']+v['left']['fp'])]),2)
        rm = round(np.mean([v['right']['tp']/(v['right']['tp']+v['right']['fp']), v['right']['tn']/(v['right']['tn']+v['right']['fp'])]),2)
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
    accuracy_dict_left = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_left.items()}
    accuracy_dict_right = {k:round(v['t']/(v['t']+v['f']),2) for k,v in accuracy_dict_right.items()}
    plt.plot(list(np.arange(len(left_metric))),left_metric, label="left turn")
    plt.plot(list(np.arange(len(right_metric))),right_metric, label="right turn")
    plt.ylabel('precision-specificity ratio')
    plt.legend(loc="upper left")
    plt.title(l3_eq)
    plt.xticks(list(np.arange(len(left_metric))), list(precision_dict.keys()))
    plt.figure()
    plt.plot(list(np.arange(len(accuracy_dict_left))),list(accuracy_dict_left.values()), label="left turn")
    plt.plot(list(np.arange(len(accuracy_dict_right))),list(accuracy_dict_right.values()), label="right turn")
    plt.ylabel('accuracy')
    plt.title(l3_eq)
    assert all([x1==x2 for x1,x2 in zip(accuracy_dict_left.keys(),accuracy_dict_right.keys())])
    plt.xticks(list(np.arange(len(accuracy_dict_left))), list(accuracy_dict_left.keys()))
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
    plt.xticks(list(np.arange(len(rm_fn))), list(precision_dict.keys()))
    plt.show()
    f=1
    
def analyse_turning_utilities(l3_eq,weighted,diff_plots):
    true_turn_utils = {'left':[],'right':[]}
    model_turn_utils = dict()
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
            if model_type == 'Ql1':
                model_str = model_str + '_' + model_params['lzero_behavior']
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
            if row[8] is not None:
                ag_emp_utils = [np.asarray(x)[:,ag_idx] for x in json.loads(row[8])]
            if row[10] is not None:
                ag_soln_utils = [np.asarray(x)[:,ag_idx] for x in json.loads(row[10])]
            if row[11] is not None:
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
                            emp_act_util_low = low_weights @ ag_emp_utils[emp_act_idx]
                            emp_act_util_high = high_weights @ ag_emp_utils[emp_act_idx]
                            this_emp_act_util = np.mean([emp_act_util_low,emp_act_util_high])
                            sol_act_util_low = low_weights @ ag_soln_utils[sol_act_idx]
                            sol_act_util_high = high_weights @ ag_soln_utils[sol_act_idx]
                            this_sol_act_util = np.mean([sol_act_util_low,sol_act_util_high])
                            util_diff = this_emp_act_util-this_sol_act_util
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
                            emp_act_util_low = low_weights @ ag_emp_utils[emp_act_idx]
                            emp_act_util_high = high_weights @ ag_emp_utils[emp_act_idx]
                            this_emp_act_util = np.mean([emp_act_util_low,emp_act_util_high])
                            sol_act_util_low = low_weights @ ag_soln_utils[sol_act_idx]
                            sol_act_util_high = high_weights @ ag_soln_utils[sol_act_idx]
                            this_sol_act_util = np.mean([sol_act_util_low,sol_act_util_high])
                            util_diff = this_emp_act_util-this_sol_act_util
                            model_turn_utils[model_str]['left'].append(util_diff)
            else:
                continue
    if not diff_plots:
        left_turn_utils = [('true_util',[x for x in true_turn_utils['left'] if not np.isnan(x)])]
        right_turn_utils = [('true_util',[x for x in true_turn_utils['right'] if not np.isnan(x)])]
    else:
        left_turn_utils = []
        right_turn_utils = []
    for k,v in model_turn_utils.items():
        left_turn_utils.append((k,v['left']))
        right_turn_utils.append((k,v['right']))
    fig = plt.figure()
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
    plt.show()
    f=1

def analyse_weight_distribution(l3_eq):
    weights_dict_right,weights_dict_left = dict(),dict()
    for file_id in constants.ALL_FILE_IDS[0:5]: 
        #if int(file_id) > 769:
        #    continue
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        q_string = "select * from UTILITY_WEIGHTS"
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
        model_str = list(weights_dict_left.keys())[i]
        ax.title.set_text(model_str)
        ax.bar(['vi','pi','ex','rl'],[weights_dict_left[model_str]['vi_h']-weights_dict_left[model_str]['vi_l'],weights_dict_left[model_str]['pi_h']-weights_dict_left[model_str]['pi_l'],
                                      weights_dict_left[model_str]['ex_h']-weights_dict_left[model_str]['ex_l'],0.1],
               bottom = [weights_dict_left[model_str]['vi_l'],weights_dict_left[model_str]['pi_l'],weights_dict_left[model_str]['ex_l'],weights_dict_left[model_str]['rl_l']])
    
    
    fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
    plt.suptitle(l3_eq+'_'+'right')
    for i, ax in enumerate(axes.flatten()):
        model_str = list(weights_dict_right.keys())[i]
        if model_str == 'CorrEq':
            continue
        ax.title.set_text(model_str)
        ax.bar(['vi','pi','ex','rl'],[weights_dict_right[model_str]['vi_h']-weights_dict_right[model_str]['vi_l'],weights_dict_right[model_str]['pi_h']-weights_dict_right[model_str]['pi_l'],
                                      weights_dict_right[model_str]['ex_h']-weights_dict_right[model_str]['ex_l'],0.1],
               bottom = [weights_dict_right[model_str]['vi_l'],weights_dict_right[model_str]['pi_l'],weights_dict_right[model_str]['ex_l'],weights_dict_right[model_str]['rl_l']])
    
    
    #plt.show()
    f=1
     
   
    
#analyse_turning_utilities('SAMPLING_EQ',False, False)
analyse_l1_precision_spec_ratio('BASELINE')

#analyse_weight_distribution('BASELINE')
#analyse_weight_distribution('SAMPLING_EQ')
#plt.show()
