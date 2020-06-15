'''
Created on Apr 20, 2020

@author: Atrisha
'''

from equilibria import equilibria_core, cost_evaluation 
from all_utils import utils,db_utils,trajectory_utils
import itertools
from constants import *
import numpy as np
import matplotlib.pyplot as plt
from planning_objects import VehicleState
from equilibria.equilibria_objects import *
import math
import sqlite3
import ast
from scipy.optimize import curve_fit
from sklearn import tree
import csv
from sklearn.preprocessing import OneHotEncoder

log = constants.eq_logger

def _exp(u,l):
    return np.exp(u*l)

def _exp_w_scaling(u,l,a):
    return a*np.exp(u*l)

def calc_equilibria(curr_time,traj_det,payoff_type,status_str,param_str,emp_action):
    calc_beliefs = False if param_str == 'l2_Nash|l3_baseline_only' else True
    traffic_signal = utils.get_traffic_signal(curr_time, 'ALL','769')
    sub_veh_actions = []
    agent_ids = []
    ag_ct = 0
    if calc_beliefs:
        emp_action_dict = {(int(k[0][0:3]),int(k[0][4:7])):[x[8:] for x in k] for k in emp_action}
    sub_agent = int(list(traj_det['raw_data'].keys())[0].split('-')[0])
    all_actions = []
    beliefs = []
    for k,v in traj_det.items():
        if k == 'raw_data':
            continue
        if k == 'relev_agents':
            for ra,rv in traj_det[k].items():
                ac_l,b_l = [],[]
                for l1,l2 in rv.items():
                    for l2_a in l2:
                        act_str = cost_evaluation.unreadable(str(sub_agent)+'|'+str(ra)+'|'+l1+'|'+l2_a)
                        ac_l.append(act_str)
                        if calc_beliefs:
                            if l1+'_'+l2_a in emp_action_dict[(sub_agent,ra)]:
                                b_l.append(1/len(emp_action_dict[(sub_agent,ra)]))
                            else:
                                b_l.append(0)
                if calc_beliefs:
                    beliefs.append(b_l)
                all_actions.append(ac_l)
        else:
            ac_l,b_l = [],[]
            for l1,l2 in v.items():
                for l2_a in l2:
                    act_str = cost_evaluation.unreadable(str(k)+'|000|'+l1+'|'+l2_a)
                    ac_l.append(act_str)
                    if calc_beliefs:
                        if l1+'_'+l2_a in emp_action_dict[(sub_agent,0)]:
                            b_l.append(1/len(emp_action_dict[(sub_agent,0)]))
                        else:
                            b_l.append(0)
            if calc_beliefs:
                beliefs.append(b_l)
            all_actions.append(ac_l)
    for act_l in all_actions:
        if act_l[0][6:9] == '000':
            sub_veh_actions = list(act_l)
            break
    all_action_combinations = list(itertools.product(*[v for v in all_actions]))      
    if calc_beliefs:
        all_belief_combinations = list(itertools.product(*[v for v in beliefs]))
        belief_dict = {k:v for k,v in zip(all_action_combinations,all_belief_combinations)}
    #fig, ax = plt.subplots()
    traj_ct = 0
    N_traj_ct = len(all_action_combinations)
    if N_traj_ct > 512:
        brk=1
    traj_dict = dict()
    pay_off_dict = dict()
    num_agents = len(agent_ids)
    payoff_trajectories_indices_dict = dict()
    equilibria_actions,sv_all_resp_payoffs = [],[]
    eval_config = EvalConfig('baseline_only')
    eval_config.set_traffic_signal(traffic_signal)
    traj_info_to_traj_id = dict()
    if BASELINE_TRAJECTORIES_ONLY:
        all_traj_info_ids = []
        for a_c in all_action_combinations:
            for a in a_c:
                l1_action = [k for k,v in L1_ACTION_CODES.items() if v == int(a[9:11])][0]
                l2_action = [k for k,v in L2_ACTION_CODES.items() if v == int(a[11:13])][0]
                _k= str(int(a[3:6]))+'-'+str(int(a[6:9]))
                traj_info_id = [x for x in traj_det['raw_data'][_k] if x[4]==l1_action and x[5]==l2_action][0][1]
                if traj_info_id not in all_traj_info_ids:
                    all_traj_info_ids.append(traj_info_id)
        traj_dict_for_traj_info = utils.load_trajs_for_traj_info_id(all_traj_info_ids, BASELINE_TRAJECTORIES_ONLY)
        for k,v in traj_dict_for_traj_info.items():
            eval_config.traj_dict[k] = [x[1:] for x in v]
            traj_info_to_traj_id[v[0][0]] = k
    traj_ct = 0
    for i,a_c in enumerate(all_action_combinations):
        traj_id_list = []
        eval_config.set_curr_strat_tuple(a_c)
        eval_config.set_curr_strat_traj_ids([])
        eval_config.set_num_agents(len(a_c))
        if calc_beliefs:
            eval_config.set_curr_beliefs(all_belief_combinations[i])
        for a in a_c:
            l1_action = [k for k,v in L1_ACTION_CODES.items() if v == int(a[9:11])][0]
            l2_action = [k for k,v in L2_ACTION_CODES.items() if v == int(a[11:13])][0]
            _k= str(int(a[3:6]))+'-'+str(int(a[6:9]))
            traj_info_id = [x for x in traj_det['raw_data'][_k] if x[4]==l1_action and x[5]==l2_action][0][1]
            if not BASELINE_TRAJECTORIES_ONLY:
                traj_ids = utils.load_traj_ids_for_traj_info_id(traj_info_id, BASELINE_TRAJECTORIES_ONLY)
                traj_id_list.append(traj_ids)
            else:
                eval_config.strat_traj_ids.append([traj_info_to_traj_id[traj_info_id]])
                
        #print('calculating payoffs')
        if not BASELINE_TRAJECTORIES_ONLY:
            payoffs,traj_indices,eq = cost_evaluation.calc_l3_equilibrium_payoffs(True,True,traj_id_list,a_c,traffic_signal)
            if a_c not in pay_off_dict:
                pay_off_dict[a_c] = payoffs
                payoff_trajectories_indices_dict[a_c] = traj_indices
            traj_ct += 1
            print(status_str,traj_ct,'/',N_traj_ct)
        else:
            payoffs = cost_evaluation.calc_baseline_traj_payoffs(eval_config)
            if a_c not in pay_off_dict:
                pay_off_dict[a_c] = next(iter(payoffs.values())).tolist()
                payoff_trajectories_indices_dict[a_c] = traj_id_list
            traj_ct += 1
            print(status_str,traj_ct,'/',N_traj_ct)
        #print('calculating payoffs....DONE')
    if not BASELINE_TRAJECTORIES_ONLY:    
        seq = ['max','min','mean']
        
        for i in np.arange(len(seq)):
            if payoff_type is not None and payoff_type != seq[i]:
                continue
            _t_p_dict = dict()
            for k,v in pay_off_dict.items():
                _t_p_dict[k] = v[i,:]
            eq = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(_t_p_dict)
            
            print(seq[i],'equilibria_core are')
            for e in eq:
                print(utils.print_readable(e), [round(float(p),4) for p in _t_p_dict[e]])
                traj_indices = payoff_trajectories_indices_dict[e]
                equilibria_actions.append((e,traj_indices[i,:],_t_p_dict[e]))
                traj_xs,traj_ys = [],[]
                for j in np.arange(num_agents):
                    traj_xs.append(traj_dict[e[j]][int(traj_indices[0,j])]['x'])
                    traj_ys.append(traj_dict[e[j]][int(traj_indices[0,j])]['y'])
                #visualizer.show_animation(traj_xs, traj_ys)
    else:
        if param_str == 'l2_Nash|l3_baseline_only':
            eq = equilibria_core.calc_pure_strategy_nash_equilibrium_exhaustive(pay_off_dict,True)
            for e,p in eq.items():
                readable_eq = utils.print_readable(e)
                eq_act_tuple = [x if x[6:9]!='000' else None for x in list(e)]
                sv_act_payoffs = []
                for sv_act in sub_veh_actions:
                    _act_tup = tuple([x if x is not None else sv_act for x in eq_act_tuple])
                    sv_index = _act_tup.index(sv_act)
                    sv_payoff = round(pay_off_dict[_act_tup][sv_index],6)
                    sv_act_payoffs.append(sv_payoff)
                print(readable_eq)
                equilibria_actions.append(e)
                sv_all_resp_payoffs.append(sv_act_payoffs)
        else:
            equilibria_actions,all_actions,all_act_payoffs = equilibria_core.calc_best_response_with_beliefs(pay_off_dict, belief_dict)
            sub_veh_idx = None
            for idx,act in enumerate(all_actions):
                if len(act)>0 and act[0][6:9] == '000':
                    sub_veh_idx = idx
                    break
            sub_veh_actions,sv_all_resp_payoffs = all_actions[sub_veh_idx], [[round(x,6) for x in all_act_payoffs[sub_veh_idx]]]*len(equilibria_actions)
            sv_all_resp_payoffs = []
            for eq_acts in equilibria_actions:
                eq_sv_payoffs = []
                for eq_act in eq_acts:
                    if eq_act[6:9] == '000':
                        if all_act_payoffs[sub_veh_idx][all_actions[sub_veh_idx].index(eq_act)] != max(all_act_payoffs[sub_veh_idx]):
                            brk=1
                        else:
                            eq_sv_payoffs = all_act_payoffs[sub_veh_idx]
                sv_all_resp_payoffs.append(eq_sv_payoffs)
    #print(len(all_action_combinations))
    return equilibria_actions,sub_veh_actions,sv_all_resp_payoffs




            


                
                
                        
def get_equil_action_in_db(param_str):
    eq_acts = dict()
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRACK_ID,TIME FROM EQUILIBRIUM_ACTIONS where EQ_CONFIG_PARMS='"+param_str+"'"
    c.execute(q_string)                 
    res = c.fetchall()
    for row in res:
        if row[1] in eq_acts:
            eq_acts[row[1]].append(row[0])
        else:
            eq_acts[row[1]] = [row[0]]
    return eq_acts
         

def add_eq_state_contexts(param_str,direction):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRACK_ID,TIME FROM EQUILIBRIUM_ACTIONS WHERE EQ_CONFIG_PARMS = '"+param_str+"' and task='"+str(direction)+"'"
    c.execute(q_string)                 
    res = c.fetchall()
    N_tot = len(res)
    for ct,row in enumerate(res):
        veh_state = utils.setup_vehicle_state(row[0],row[1])
        ins_string = "UPDATE EQUILIBRIUM_ACTIONS SET TRAFFIC_SIGNAL=?, NEXT_SIGNAL_CHANGE=?, SEGMENT=?, SPEED=?, X=?, Y=?, LEAD_VEHICLE_ID=? WHERE TRACK_ID=? AND TIME=?"
        ins_tuple = (veh_state.signal, str(veh_state.next_signal_change), veh_state.current_segment, veh_state.speed, veh_state.x, veh_state.y, veh_state.leading_vehicle.id if veh_state.leading_vehicle is not None else None,row[0],row[1])
        print(ct,'/',N_tot,ins_tuple)
        c.execute(ins_string,ins_tuple)
    conn.commit()
    conn.close()
    
def temp_column_fix():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRACK_ID,TIME,EQ_CONFIG_PARMS,EMPIRICAL_ACTION FROM EQUILIBRIUM_ACTIONS where EQ_CONFIG_PARMS='l2_BR_w_true_belief|l3_baseline_only'"
    c.execute(q_string)                 
    res = c.fetchall()
    N_tot = len(res)
    for ct,row in enumerate(res):
        emp_act = ast.literal_eval(row[3])
        sv_act = None
        for _a in emp_act:
            if _a[0][4:7] == '000':
                sv_act = _a
                break
        ins_string = "UPDATE EQUILIBRIUM_ACTIONS SET EMPIRICAL_ACTION=? WHERE TRACK_ID=? AND TIME=? and EQ_CONFIG_PARMS=?"
        ins_tuple = (str(sv_act),row[0],row[1],row[2])
        print(ct,'/',N_tot,ins_tuple)
        c.execute(ins_string,ins_tuple)
    conn.commit()
    conn.close()
    
def temp_all_act_order_fix():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT TRACK_ID,TIME,EQ_CONFIG_PARMS,ALL_ACTIONS,ALL_ACTION_PAYOFFS FROM EQUILIBRIUM_ACTIONS where EQ_CONFIG_PARMS='l2_BR_w_true_belief|l3_baseline_only'"
    c.execute(q_string)                 
    res = c.fetchall()
    N_tot = len(res)
    for ct,row in enumerate(res):
        all_act = ast.literal_eval(row[3])
        all_act_payoffs = ast.literal_eval(row[4])
        all_act_sorted = sorted(all_act,key = lambda val: val[9:])
        new_indx = dict()
        for i,v in enumerate(all_act):
            new_indx[i] = all_act_sorted.index(v)
        new_all_act_payoff = []
        for a_p in all_act_payoffs:
            new_p =[]
            for i,p in enumerate(a_p):
                new_p.append(a_p[new_indx[i]])
            new_all_act_payoff.append(new_p)
        ins_string = "UPDATE EQUILIBRIUM_ACTIONS SET ALL_ACTIONS=?, ALL_ACTION_PAYOFFS=?  WHERE TRACK_ID=? AND TIME=? and EQ_CONFIG_PARMS=?"
        ins_tuple = (str(all_act_sorted),str(new_all_act_payoff),row[0],row[1],row[2])
        print(ct,'/',N_tot,ins_tuple)
        c.execute(ins_string,ins_tuple)
    conn.commit()
    conn.close()    

def strat_in(eq_dict,strat):
    for k in list(eq_dict.keys()):
        if str(ast.literal_eval(k).sort()) == str(strat.sort()):
            return True
    return False
        
    
def split_by_distinct_actions(eq_info_rows):
    eq_dict = dict()
    for row in eq_info_rows:
        if len(ast.literal_eval(row[14])) == 0:
            ''' no equilibrium for this row'''
            continue
        all_acts = []
        for a in [utils.print_readable(x)[8:] for x in ast.literal_eval(row[17])]:
            if a not in all_acts:
                all_acts.append(a)
        all_acts_l = all_acts
        all_acts = str(all_acts_l)
        if all_acts not in eq_dict:
            eq_dict[all_acts] = dict()
            eq_dict[all_acts]['emp_act'] = [[utils.print_readable(x)[8:] for x in ast.literal_eval(row[15])]]
            eq_dict[all_acts]['eq_acts'] = [[utils.print_readable(x)[8:] for x  in ast.literal_eval(row[14])]]
            all_act_payoffs = ast.literal_eval(row[18]) #np.mean(np.vstack(ast.literal_eval(row[16])), axis=0)
            eq_dict[all_acts]['all_payoffs'] = [all_act_payoffs]
            for idx,eq_a in enumerate([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[14])]):
                if max(all_act_payoffs[idx]) != all_act_payoffs[idx][all_acts_l.index(eq_a)]:
                    brk=1
        else:
            eq_dict[all_acts]['emp_act'].append([utils.print_readable(x)[8:] for x in ast.literal_eval(row[15])])
            eq_dict[all_acts]['eq_acts'].append([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[14])])
            all_act_payoffs = ast.literal_eval(row[18]) #np.mean(np.vstack(ast.literal_eval(row[16])), axis=0)
            eq_dict[all_acts]['all_payoffs'].append(all_act_payoffs)
            for idx,eq_a in enumerate([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[14])]):
                max_p = max(all_act_payoffs[idx])
                eq_p = all_act_payoffs[idx][all_acts_l.index(eq_a)]
                if max_p != eq_p:
                    brk=1
    return eq_dict


def build_analysis_table(eq_non_eq,u_deltas):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    eval_config = EvalConfig()
    eval_config.set_l1_eq_type(L1_EQ_TYPE)
    eval_config.set_l3_eq_type(L3_EQ_TYPE)
    param_str = eval_config.l1_eq +'|'+ eval_config.l3_eq +'|'+ 'BOUNDARY' if eval_config.l3_eq is not None else eval_config.l1_eq +'|BASELINE_ONLY'
    all_segs = ['prep-turn%','exec-turn%','ln\__\__']
    all_conds = [[("EQUILIBRIUM_ACTIONS.SEGMENT like 'prep-turn%' ESCAPE '\\'","SEGMENT=prep-left-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'exec-turn%' ESCAPE '\\'","SEGMENT=exec-left-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'ln\__\__' ESCAPE '\\'","SEGMENT=OTHER LANES"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'rt_prep-turn%' ESCAPE '\\'","SEGMENT=prep-right-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'rt_exec-turn%' ESCAPE '\\'","SEGMENT=exec-right-turn")],[
                 ("TRAFFIC_SIGNAL = 'R'","TRAFFIC_SIGNAL=R"),
                 ("TRAFFIC_SIGNAL = 'Y'","TRAFFIC_SIGNAL=Y"),
                 ("TRAFFIC_SIGNAL = 'G'","TRAFFIC_SIGNAL=G")],[
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) < 10 AND (SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y' OR SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'R')","NEXT_CHANGE=<10-Y/R"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) < 10 AND SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y'","NEXT_CHANGE=<10-G"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) >= 10 AND (SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y' OR SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'R')","NEXT_CHANGE=>EQ 10-Y/R"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) >= 10 AND SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y'","NEXT_CHANGE=>EQ 10-G")
                 ],[("SPEED <= 8","SPEED=LOW SPEED"),
                 ("SPEED <= 14 AND SPEED > 8","SPEED=MEDIUM SPEED"),
                 ("SPEED > 14","SPEED=HIGH SPEED")],[
                 ("PEDESTRIAN = 'Y'","PEDESTRIAN=PEDESTRIAN"),
                 ("PEDESTRIAN = 'N'","PEDESTRIAN=NO PEDESTRIAN")],[
                 ("LEAD_VEHICLE_ID IS NULL","LEAD VEHICLE=NO LV"),
                 ("LEAD_VEHICLE_ID IS NOT NULL","LEAD VEHICLE=LV")],[
                 ("(LENGTH(RELEV_AGENT_IDS)-2) = 0","NUM OF RELEV AGENTS=RL 0"),
                 ("(LENGTH(RELEV_AGENT_IDS)-2) > 0 AND (LENGTH(RELEV_AGENT_IDS) - LENGTH(REPLACE(RELEV_AGENT_IDS, ',', '')))=1","NUM OF RELEV AGENTS = RL <EQ 2"),
                 ("(LENGTH(RELEV_AGENT_IDS)-2) > 0 AND (LENGTH(RELEV_AGENT_IDS) - LENGTH(REPLACE(RELEV_AGENT_IDS, ',', '')))>1","NUM OF RELEV AGENTS = RL > 2")]
                 ]
    all_cond_combinations = list(itertools.product(*[x for x in all_conds]))
    table_dict = dict()
    file_suffix = ''
    file_suffix = '_u_deltas' if u_deltas else file_suffix
    file_suffix = '_eq_non_eq' if eq_non_eq else file_suffix
    all_act_map = dict()
    all_act_idx = 1
    plt_X = []
    plt_Y = []
    plt_x_ct = 0
    with open(os.path.join(constants.ROOT_DIR,constants.RESULTS,param_str.replace("|",',')+file_suffix+'.csv'), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE,escapechar=' ')
        if eq_non_eq or u_deltas:
            writer.writerow(['EQ_TYPE','SEGMENT','SIGNAL','NEXT_CHANGE','SPEED','PEDESTRIAN','LEAD_VEHICLE','RELEV_VEHICLE','ACTIONS','ON_EQ'])
        for cond_idx,cond_info in enumerate(all_cond_combinations):
            cond,cond_str = ' and '.join([x[0] for x in cond_info]),[x[1].split("=")[1] for x in cond_info]
            table_dict[cond_idx] = dict()
            eq_dict_l = []
            q_string = "select * from EQUILIBRIUM_ACTIONS where EQ_CONFIG_PARMS= '"+param_str+"' and "+cond
            c.execute(q_string)                 
            res = c.fetchall()
            eq_dict = split_by_distinct_actions(res)
            eq_dict_l.append(eq_dict)
            for didx,eq_dict in enumerate(eq_dict_l):
                for act_k,act_info in eq_dict.items():
                    act_delta_dict = dict()
                    emp_act_distr = []
                    eq_act_distr = []            
                    all_acts = ast.literal_eval(act_k)
                    disp_str = []
                    if len(all_acts) > 0:
                        table_dict[cond_idx][str(all_acts)] = dict()
                        if str(all_acts) not in all_act_map:
                            all_act_map[str(all_acts)] = all_act_idx
                            all_act_idx += 1 
                        for k,v in act_info.items():
                            if k == 'emp_act':
                                for acts_idx,emp_acts in enumerate(v):
                                    for act_idx,e_a in enumerate(emp_acts):
                                        emp_act_idx = all_acts.index(e_a)
                                        emp_act_payoff = [x[all_acts.index(e_a)] for x in act_info['all_payoffs'][acts_idx]]
                                        max_payoff = [max(x) for x in act_info['all_payoffs'][acts_idx]]
                                        min_payoff = [min(x) for x in act_info['all_payoffs'][acts_idx]]
                                        payoff_diff = np.subtract(max_payoff,emp_act_payoff)
                                        if len(payoff_diff) == 0:
                                            brk=1
                                        min_payoff_diff =min(payoff_diff)
                                        for i in np.arange(len(payoff_diff)):
                                            if i == len(act_info['eq_acts'][acts_idx]):
                                                eq_act = act_info['eq_acts'][acts_idx][i-1]
                                            else:
                                                eq_act = act_info['eq_acts'][acts_idx][i]
                                            if eq_act == e_a:
                                                min_payoff_diff = 0
                                                break
                                        
                                        if min_payoff_diff > 0:
                                            ''' find the eq act with minimum diff with emp payoff '''
                                            for i in np.arange(len(payoff_diff)):
                                                if i == len(act_info['eq_acts'][acts_idx]):
                                                    eq_act = act_info['eq_acts'][acts_idx][i-1]
                                                else:
                                                    eq_act = act_info['eq_acts'][acts_idx][i]
                                                if payoff_diff[i] == min(payoff_diff):
                                                    diff_tup = str(eq_act)+'-'+str(e_a)
                                                    if diff_tup in act_delta_dict:
                                                        act_delta_dict[diff_tup].append(min(payoff_diff))
                                                    else:
                                                        act_delta_dict[diff_tup] = [min(payoff_diff)]
                                                if eq_act == e_a:
                                                    brk=1
                                        emp_act_distr.append(min_payoff_diff)
                            elif k == 'eq_acts':
                                for acts_idx,eq_acts in enumerate(v):
                                    for act_idx,e_a in enumerate(eq_acts):
                                        eq_act_payoff = [x[all_acts.index(e_a)] for x in act_info['all_payoffs'][acts_idx]]
                                        max_payoff = [max(x) for x in act_info['all_payoffs'][acts_idx]]
                                        payoff_diff = np.subtract(max_payoff,eq_act_payoff)
                                        eq_act_distr.append(min(payoff_diff))
                        #table_dict[cond_idx][str(all_acts)]['text_str'] = '\n'.join([t for t in all_acts])
                        if len(emp_act_distr) > 0:
                            table_dict[cond_idx][str(all_acts)]['data'] = emp_act_distr
                            if eq_non_eq:
                                for emp_pay_diff in emp_act_distr:
                                    if emp_pay_diff == 0:
                                        writer.writerow([param_str]+cond_str+['acts-'+str(all_act_map[str(all_acts)]),0])
                                        print(','.join([x for x in cond_str]),'acts-'+str(all_act_map[str(all_acts)]),0)
                                        plt_x_ct += 1
                                        plt_X.append(plt_x_ct)
                                        plt_Y.append(0)
                                    else:
                                        writer.writerow([param_str]+cond_str+['acts-'+str(all_act_map[str(all_acts)]),1])
                                        print(','.join([x for x in cond_str]),'acts-'+str(all_act_map[str(all_acts)]),1)
                                        plt_x_ct += 1
                                        plt_X.append(plt_x_ct)
                                        plt_Y.append(1)
                            else:
                                if u_deltas:
                                    for emp_pay_diff in emp_act_distr:
                                        writer.writerow([param_str]+cond_str+['acts-'+str(all_act_map[str(all_acts)]),emp_pay_diff])
                                        print(','.join([x for x in cond_str]),'acts-'+str(all_act_map[str(all_acts)]),emp_pay_diff)
                                        
                                else:
                                    for emp_pay_diff in emp_act_distr:
                                        plt_x_ct += 1
                                        plt_X.append(plt_x_ct)
                                        plt_Y.append(emp_pay_diff)
                                    _data,_bins = np.histogram(emp_act_distr, bins=np.arange(0,1.02,.01), density=True)
                                    _data = [x*0.01 for x in _data]
                                    
                                    if _data[0] == 1:
                                        l = np.inf
                                    else:
                                        popt, pcov = curve_fit(_exp, np.arange(0,1.01,.01), _data)
                                        l = -popt[0]
                                    table_dict[cond_idx][str(all_acts)]['lambda'] = l
                                    table_dict[cond_idx][str(all_acts)]['N'] = len(emp_act_distr)
                                    print(param_str,','.join([x for x in cond_str]),'|'.join(all_acts),len(emp_act_distr),l)
                                    writer.writerow([param_str]+cond_str+['|'.join(all_acts),len(emp_act_distr),l])
    print(all_act_map)
    '''
    X = np.arange(0,1.1,0.05)
    count, bins, ignored = plt.hist(plt_Y, X, density=True)
    popt, pcov = curve_fit(_exp_w_scaling, X[:-1], [x*1 for x in count])
    X_hifreq = np.arange(0,X[-1]+.01,.01)
    plt.plot(X_hifreq,[_exp_w_scaling(x,popt[0],popt[1]) for x in X_hifreq])
    plt.show()
    '''                       
                            
def analyse_equilibrium_data():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    eval_config = EvalConfig()
    eval_config.set_l1_eq_type(L1_EQ_TYPE)
    eval_config.set_l3_eq_type(L3_EQ_TYPE)
    param_str = eval_config.l1_eq +'|'+ eval_config.l3_eq +'|'+ 'BOUNDARY' if eval_config.l3_eq is not None else eval_config.l1_eq +'|BASELINE_ONLY'
    all_segs = ['prep-turn%','exec-turn%','ln\__\__','rt_prep-turn%','rt_exec-turn%']
    plot_dict = dict()
    for seg in all_segs:
        plot_dict[seg] = dict()
        eq_dict_l = []
        q_string = "select * from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.TRAFFIC_SIGNAL in ('G','Y') and EQ_CONFIG_PARMS= '"+param_str+"' and EQUILIBRIUM_ACTIONS.SEGMENT like '"+seg+"' ESCAPE '\\' "
        c.execute(q_string)                 
        res = c.fetchall()
        eq_dict = split_by_distinct_actions(res)
        eq_dict_l.append(eq_dict)
        '''
        q_string = "select * from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.TRAFFIC_SIGNAL in ('G','Y') and EQ_CONFIG_PARMS= 'l2_BR_w_true_belief|l3_baseline_only' and EQUILIBRIUM_ACTIONS.SEGMENT like '"+seg+"' ESCAPE '\\' "
        c.execute(q_string)                 
        res = c.fetchall()
        eq_dict = split_by_distinct_actions(res)
        eq_dict_l.append(eq_dict)
        '''
        for didx,eq_dict in enumerate(eq_dict_l):
            
            for act_k,act_info in eq_dict.items():
                act_delta_dict = dict()
                emp_act_distr = []
                eq_act_distr = []            
                all_acts = ast.literal_eval(act_k)
                if len(all_acts) > 0:
                    plot_dict[seg][str(all_acts)] = dict()
                    for k,v in act_info.items():
                        if k == 'emp_act':
                            for acts_idx,emp_acts in enumerate(v):
                                for act_idx,e_a in enumerate(emp_acts):
                                    emp_act_idx = all_acts.index(e_a)
                                    emp_act_payoff = [x[all_acts.index(e_a)] for x in act_info['all_payoffs'][acts_idx]]
                                    max_payoff = [max(x) for x in act_info['all_payoffs'][acts_idx]]
                                    min_payoff = [min(x) for x in act_info['all_payoffs'][acts_idx]]
                                    payoff_diff = np.subtract(max_payoff,emp_act_payoff)
                                    if len(payoff_diff) == 0:
                                        brk=1
                                    min_payoff_diff =min(payoff_diff)
                                    for i in np.arange(len(payoff_diff)):
                                        if i == len(act_info['eq_acts'][acts_idx]):
                                            eq_act = act_info['eq_acts'][acts_idx][i-1]
                                        else:
                                            eq_act = act_info['eq_acts'][acts_idx][i]
                                        if eq_act == e_a:
                                            min_payoff_diff = 0
                                            break
                                    
                                    if min_payoff_diff > 0:
                                        ''' find the eq act with minimum diff with emp payoff '''
                                        for i in np.arange(len(payoff_diff)):
                                            if i == len(act_info['eq_acts'][acts_idx]):
                                                eq_act = act_info['eq_acts'][acts_idx][i-1]
                                            else:
                                                eq_act = act_info['eq_acts'][acts_idx][i]
                                            if payoff_diff[i] == min(payoff_diff):
                                                diff_tup = str(eq_act)+'-'+str(e_a)
                                                if diff_tup in act_delta_dict:
                                                    act_delta_dict[diff_tup].append(min(payoff_diff))
                                                else:
                                                    act_delta_dict[diff_tup] = [min(payoff_diff)]
                                            if eq_act == e_a:
                                                brk=1
                                    emp_act_distr.append(min_payoff_diff)
                        elif k == 'eq_acts':
                            for acts_idx,eq_acts in enumerate(v):
                                for act_idx,e_a in enumerate(eq_acts):
                                    eq_act_payoff = [x[all_acts.index(e_a)] for x in act_info['all_payoffs'][acts_idx]]
                                    max_payoff = [max(x) for x in act_info['all_payoffs'][acts_idx]]
                                    payoff_diff = np.subtract(max_payoff,eq_act_payoff)
                                    eq_act_distr.append(min(payoff_diff))
                    plot_dict[seg][str(all_acts)]['text_str'] = '\n'.join([t for t in all_acts])
                    plot_dict[seg][str(all_acts)]['data'] = emp_act_distr
                    plot_dict[seg][str(all_acts)]['title_str'] = seg+' '+param_str
                    
                    #_ = ax.hist(emp_act_distr, bins='auto')
                    #_ = ax[1].hist(eq_act_distr, bins=np.arange(0,1.1,.1))
                    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    #ax.text(0.5, 0.95, text_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
                    #ax[1].text(0.5, 0.95, text_str, transform=ax[1].transAxes, fontsize=8,verticalalignment='top', bbox=props)
                    #title_str = param_str
                    #plt.title(seg+' '+title_str)
                    '''
                    fig, ax = plt.subplots()
                    x = np.arange(len(act_delta_dict))
                    plt.bar(x, height=list([np.mean(y) for y in act_delta_dict.values()]))
                    plt.xticks(x, list(act_delta_dict.keys()), rotation='vertical')
                    #plt.tight_layout()
                    plt.subplots_adjust(bottom=.5)
                    title_str = 'Nash eq' if didx == 0 else 'Nash eq with prior beliefs'
                    plt.title('delta_dict'+' '+title_str)
                    '''
    n_rows,max_n_cols = len(plot_dict),max([len(v) for k,v in plot_dict.items()])
    fig, ax = plt.subplots(n_rows,max_n_cols)
    r_ct = 0
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for k,v in plot_dict.items():
        c_ct = 0
        for a_act,a_act_v in v.items():
            if len(a_act_v['data']) == 0:
                continue
            _data,_bins,_ = ax[r_ct,c_ct].hist(a_act_v['data'], bins=np.arange(0,1.02,.01), density=True)
            print(sum(_data))
            _data = [x*0.01 for x in _data]
            if _data[0] == 1:
                l = '$\infty$'
            else:
                popt, pcov = curve_fit(_exp, np.arange(0,1.01,.01), _data)
                l = -popt[0]
            ax[r_ct,c_ct].text(0.5, 0.95, a_act_v['text_str']+'\n $\lambda$='+str(l), transform=ax[r_ct,c_ct].transAxes, fontsize=8,verticalalignment='top', bbox=props)
            ax[r_ct,c_ct].set_title(a_act_v['title_str'])
            c_ct += 1
        r_ct += 1
    plt.show()
            
            
def build_results_tree():
    categories = [[("EQUILIBRIUM_ACTIONS.SEGMENT like 'prep-turn%' ESCAPE '\\'","SEGMENT=prep-left-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'exec-turn%' ESCAPE '\\'","SEGMENT=exec-left-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'ln\__\__' ESCAPE '\\'","SEGMENT=OTHER LANES"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'rt_prep-turn%' ESCAPE '\\'","SEGMENT=prep-right-turn"),
                 ("EQUILIBRIUM_ACTIONS.SEGMENT like 'rt_exec-turn%' ESCAPE '\\'","SEGMENT=exec-right-turn")],[
                 ("TRAFFIC_SIGNAL = 'R'","TRAFFIC_SIGNAL=R"),
                 ("TRAFFIC_SIGNAL = 'Y'","TRAFFIC_SIGNAL=Y"),
                 ("TRAFFIC_SIGNAL = 'G'","TRAFFIC_SIGNAL=G")],[
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) < 10 AND (SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y' OR SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'R')","NEXT_CHANGE=LESS THAN 10 TO Y/R"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) < 10 AND SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y'","NEXT_CHANGE=LESS THAN 10 TO G"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) >= 10 AND (SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y' OR SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'R')","NEXT_CHANGE=MORE THAN 10 TO Y/R"),
                 ("CAST(SUBSTR(NEXT_SIGNAL_CHANGE,2,INSTR(NEXT_SIGNAL_CHANGE,',')-2) AS DECIMAL) >= 10 AND SUBSTR(NEXT_SIGNAL_CHANGE,INSTR(NEXT_SIGNAL_CHANGE,',')+3,1) = 'Y'","NEXT_CHANGE=MORE THAN 10 TO G")
                 ],[("SPEED <= 8","SPEED=LOW"),
                 ("SPEED <= 14 AND SPEED > 8","SPEED=MEDIUM"),
                 ("SPEED > 14","SPEED=HIGH")],[
                 ("PEDESTRIAN = 'Y'","PEDESTRIAN=Y"),
                 ("PEDESTRIAN = 'N'","PEDESTRIAN=N")],[
                 ("LEAD_VEHICLE_ID IS NULL","LEAD VEHICLE=N"),
                 ("LEAD_VEHICLE_ID IS NOT NULL","LEAD VEHICLE=Y")],[
                 ("(LENGTH(RELEV_AGENT_IDS)-2) = 0","NUM OF RELEV AGENTS=0"),
                 ("(LENGTH(RELEV_AGENT_IDS)-2) > 0 AND (LENGTH(RELEV_AGENT_IDS) - LENGTH(REPLACE(RELEV_AGENT_IDS, ',', '')))=1","NUM OF RELEV AGENTS = LESS THAN EQ 2"),
                 ("(LENGTH(RELEV_AGENT_IDS)-2) > 0 AND (LENGTH(RELEV_AGENT_IDS) - LENGTH(REPLACE(RELEV_AGENT_IDS, ',', '')))>1","NUM OF RELEV AGENTS = MORE THAN 2")],
                [(None,'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|proceed-turn_AGGRESSIVE|proceed-turn_NORMAL'),
                   (None,'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL'),
                   (None, 'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|proceed-turn_AGGRESSIVE|proceed-turn_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL'),
                   (None, 'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL'),
                   (None, 'proceed-turn_AGGRESSIVE|proceed-turn_NORMAL'),
                   (None, 'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|proceed-turn_AGGRESSIVE|proceed-turn_NORMAL|wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL'),
                   (None, 'wait-for-oncoming_AGGRESSIVE|wait-for-oncoming_NORMAL|proceed-turn_AGGRESSIVE|proceed-turn_NORMAL|wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL'),
                   (None, 'proceed-turn_AGGRESSIVE|proceed-turn_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL'),
                   (None, 'proceed-turn_AGGRESSIVE|proceed-turn_NORMAL|decelerate-to-stop_AGGRESSIVE|decelerate-to-stop_NORMAL'),
                   (None, 'wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|follow_lead_into_intersection_AGGRESSIVE|follow_lead_into_intersection_NORMAL'),
                   (None, 'wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|follow_lead_into_intersection_AGGRESSIVE|follow_lead_into_intersection_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL'),
                   (None, 'decelerate-to-stop_AGGRESSIVE|decelerate-to-stop_NORMAL|wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|follow_lead_into_intersection_AGGRESSIVE|follow_lead_into_intersection_NORMAL'),
                   (None, 'wait_for_lead_to_cross_AGGRESSIVE|wait_for_lead_to_cross_NORMAL|wait-for-pedestrian_AGGRESSIVE|wait-for-pedestrian_NORMAL')]
                 ]
    enc_list = []
    for indx,x in enumerate(categories):
        for y in x:
            if indx!=7:
                enc_list.extend([(indx,y[1].split('=')[1])])
            else:
                enc_list.extend([(indx,y[1])])
    eval_config = EvalConfig()
    eval_config.set_l1_eq_type(L1_EQ_TYPE)
    eval_config.set_l3_eq_type(L3_EQ_TYPE)
    param_str = eval_config.l1_eq +'|'+ eval_config.l3_eq +'|'+ 'BOUNDARY' if eval_config.l3_eq is not None else eval_config.l1_eq +'|BASELINE_ONLY'
    X,Y = [],[]
    data_dict = dict()
    act_combs = []
    with open(os.path.join(constants.ROOT_DIR,constants.RESULTS,param_str.replace("|",',')+'.csv'), newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONE,escapechar=' ')
        for row in csv_reader:
            if row[-1] != 'inf':
                print(row)
            '''
            if row[0] not in data_dict:
                data_dict[row[0]] = dict()
                data_dict[row[0]][row[8]] = dict()
                data_dict[row[0]][row[8]]['N'] = [int(row[9])]
                if row[10] != 'inf':
                    data_dict[row[0]][row[8]]['Y'] = [float(row[10])]
                else:
                    data_dict[row[0]][row[8]]['Y'] = [float(100000)]
                
                data_dict[row[0]][row[8]]['X'] = [row[1:8]]
            else:
                if row[8] not in data_dict[row[0]]:
                    data_dict[row[0]][row[8]] = dict()
                    data_dict[row[0]][row[8]]['N'] = [int(row[9])]
                    if row[10] != 'inf':
                        data_dict[row[0]][row[8]]['Y'] = [float(row[10])]
                    else:
                        data_dict[row[0]][row[8]]['Y'] = [float(100000)]
                    data_dict[row[0]][row[8]]['X'] = [row[1:8]]
                else:
                    data_dict[row[0]][row[8]]['N'].append(int(row[9]))
                    if row[10] != 'inf':
                        data_dict[row[0]][row[8]]['Y'].append(float(row[10]))
                    else:
                        data_dict[row[0]][row[8]]['Y'].append(float(100000))
                    data_dict[row[0]][row[8]]['X'].append(row[1:8])
            '''
            if row[0] not in data_dict:
                data_dict[row[0]] = dict()
                data_dict[row[0]]['N'] = [int(row[9])]
                if row[10] != 'inf':
                    data_dict[row[0]]['Y'] = [float(row[10])]
                else:
                    data_dict[row[0]]['Y'] = [float(100000)]
                
                data_dict[row[0]]['X'] = [row[1:9]]
            else:
                data_dict[row[0]]['N'].append(int(row[9]))
                if row[10] != 'inf':
                    data_dict[row[0]]['Y'].append(float(row[10]))
                else:
                    data_dict[row[0]]['Y'].append(float(100000))
                data_dict[row[0]]['X'].append(row[1:9])
    for k,v in data_dict.items():
        clf = tree.DecisionTreeRegressor(max_depth=5)
        X = []
        for r in v['X']:
            R = [0]*len(enc_list)
            for indx,f in enumerate(r):
                R[enc_list.index((indx,f))] = 1
            X.append(R)
        X = np.asarray(X)
        clf = clf.fit(X, v['Y'])
        plt.figure(figsize=(25,10))
        t = tree.plot_tree(clf,feature_names = [str(x) for x in enc_list],fontsize=12)
        f=1
        plt.savefig(os.path.join(constants.ROOT_DIR,constants.RESULTS,'tree_fig.pdf'), bbox_inches='tight')
    
                
''' this method calculates the equilibria_core based on the trajectories that were generated in the hopping plans.
Assumes that the trajectories are already present in the cache.'''
def calc_eqs_for_hopping_trajectories():
    show_plots = False
    update_only = False
    eval_config = EvalConfig()
    eval_config.set_l1_eq_type(L1_EQ_TYPE)
    eval_config.set_l3_eq_type(L3_EQ_TYPE)
    param_str = eval_config.l1_eq +'|'+ eval_config.l3_eq +'|'+ 'BOUNDARY' if eval_config.l3_eq is not None else eval_config.l1_eq +'|BASELINE_ONLY'
    if update_only:
        eval_config.set_update_only(True)
        eq_acts_in_db = get_equil_action_in_db(param_str)
        eval_config.set_eq_acts_in_db(eq_acts_in_db)
    else:
        eval_config.set_update_only(False)
        eval_config.set_eq_acts_in_db(None)
    d = ['S_W','N_E']
    task_list = ['W_S','S_W','N_E','W_N','S_E','E_S']
    for direction in task_list:
        
        logging.info('setting parameters')
        eval_config.setup_parameters(direction)
        eq = Equilibria(eval_config)
        logging.info('setting empirical actions')
        eq.calc_empirical_actions()
        eq.calc_equilibrium()
        add_eq_state_contexts(param_str,direction)
    traj_util_obj = trajectory_utils.TrajectoryUtils()
    traj_util_obj.update_l1_action_in_eq_data(param_str)
    traj_util_obj.update_pedestrian_info_in_eq_data(param_str)
    build_analysis_table(False,True)
    build_analysis_table(True,False)
    
    
if __name__ == '__main__':
    calc_eqs_for_hopping_trajectories()
    '''
    eval_config = EvalConfig()
    eval_config.set_l1_eq_type(L1_EQ_TYPE)
    eval_config.set_l3_eq_type(L3_EQ_TYPE)
    traj_util_obj = trajectory_utils.TrajectoryUtils()
    param_str = eval_config.l1_eq +'|'+ eval_config.l3_eq +'|'+ 'BOUNDARY' if eval_config.l3_eq is not None else eval_config.l1_eq +'|BASELINE_ONLY'
    traj_util_obj.update_l1_action_in_eq_data(param_str)
    traj_util_obj.update_pedestrian_info_in_eq_data(param_str)
    build_analysis_table(False,True)
    build_analysis_table(True,False)
    '''