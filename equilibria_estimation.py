'''
Created on Apr 20, 2020

@author: Atrisha
'''

import equilibria_core
import utils
import cost_evaluation
import itertools
from constants import *
import numpy as np
import db_utils
import matplotlib.pyplot as plt
from planning_objects import VehicleState
from equilibria_objects import *
import math
import sqlite3
import ast
from dask.dataframe.tests.test_rolling import idx
import logging
logging.basicConfig(format='%(levelname)-8s %(filename)s: %(message)s',level=logging.INFO)





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
        all_acts = []
        for a in [utils.print_readable(x)[8:] for x in ast.literal_eval(row[16])]:
            if a not in all_acts:
                all_acts.append(a)
        all_acts_l = all_acts
        all_acts = str(all_acts_l)
        if all_acts not in eq_dict:
            eq_dict[all_acts] = dict()
            eq_dict[all_acts]['emp_act'] = [[utils.print_readable(x)[8:] for x in ast.literal_eval(row[14])]]
            eq_dict[all_acts]['eq_acts'] = [[utils.print_readable(x)[8:] for x  in ast.literal_eval(row[13])]]
            all_act_payoffs = ast.literal_eval(row[17]) #np.mean(np.vstack(ast.literal_eval(row[16])), axis=0)
            eq_dict[all_acts]['all_payoffs'] = [all_act_payoffs]
            for idx,eq_a in enumerate([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[13])]):
                if max(all_act_payoffs[idx]) != all_act_payoffs[idx][all_acts_l.index(eq_a)]:
                    brk=1
        else:
            eq_dict[all_acts]['emp_act'].append([utils.print_readable(x)[8:] for x in ast.literal_eval(row[14])])
            eq_dict[all_acts]['eq_acts'].append([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[13])])
            all_act_payoffs = ast.literal_eval(row[17]) #np.mean(np.vstack(ast.literal_eval(row[16])), axis=0)
            eq_dict[all_acts]['all_payoffs'].append(all_act_payoffs)
            for idx,eq_a in enumerate([utils.print_readable(x)[8:] for x  in ast.literal_eval(row[13])]):
                max_p = max(all_act_payoffs[idx])
                eq_p = all_act_payoffs[idx][all_acts_l.index(eq_a)]
                if max_p != eq_p:
                    brk=1
    return eq_dict
    
def analyse_equilibrium_data():
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    all_segs = ['prep-turn%','exec-turn%','ln\__\__']
    for seg in ['prep-turn%','exec-turn%']:
        eq_dict_l = []
        q_string = "select * from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.TRAFFIC_SIGNAL in ('G','Y') and EQ_CONFIG_PARMS= 'NASH|BASELINE_ONLY' and EQUILIBRIUM_ACTIONS.SEGMENT like '"+seg+"' ESCAPE '\\' "
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
                    for k,v in act_info.items():
                        if k == 'emp_act':
                            for acts_idx,emp_acts in enumerate(v):
                                for act_idx,e_a in enumerate(emp_acts):
                                    emp_act_idx = all_acts.index(e_a)
                                    emp_act_payoff = [x[all_acts.index(e_a)] for x in act_info['all_payoffs'][acts_idx]]
                                    max_payoff = [max(x) for x in act_info['all_payoffs'][acts_idx]]
                                    payoff_diff = np.subtract(max_payoff,emp_act_payoff)
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
                    text_str = '\n'.join([t for t in all_acts])
                    fig, ax = plt.subplots()
                    _ = ax.hist(emp_act_distr, bins='auto')
                    #_ = ax[1].hist(eq_act_distr, bins=np.arange(0,1.1,.1))
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.5, 0.95, text_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)
                    #ax[1].text(0.5, 0.95, text_str, transform=ax[1].transAxes, fontsize=8,verticalalignment='top', bbox=props)
                    title_str = 'Nash eq' if didx == 0 else 'Nash eq with prior beliefs'
                    plt.title(seg+' '+title_str)
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
    plt.show()
            
                
    
''' this method calculates the equilibria_core based on the trajectories that were generated in the hopping plans.
Assumes that the trajectories are already present in the cache.'''
def calc_eqs_for_hopping_trajectories():
    show_plots = False
    update_only = False
    param_str = 'l2_BR_w_true_belief|l3_baseline_only'
    if update_only:
        eq_acts_in_db = get_equil_action_in_db(param_str)
    eval_config = EvalConfig()
    d = ['S_W','N_E']
    task_list = ['W_N','E_S']
    for direction in task_list:
        logging.info('setting parameters')
        eval_config.setup_parameters(direction)
        eval_config.set_l1_eq_type(L1_EQ_TYPE)
        eval_config.set_l3_eq_type(L3_EQ_TYPE)
        eq = Equilibria(eval_config)
        logging.info('setting empirical actions')
        eq.calc_empirical_actions()
        equilibria_dict = eq.calc_equilibrium()
        eq.set_equilibria_dict(equilibria_dict)
        eq.insert_to_db()
        param_str = eq.eval_config.l1_eq +'|'+ eq.eval_config.l3_eq if eq.eval_config.l3_eq is not None else eq.eval_config.l1_eq +'|BASELINE_ONLY'
        add_eq_state_contexts(param_str,direction)
        f=1
        '''
        step_size_secs = PLAN_FREQ
        ct = 0
        N_tot = len(traj_metadata)
        for k,v in traj_metadata.items():
            ct += 1
            status_str = '('+str(ct)+'/'+str(N_tot)+')'
            print('------time:',k)
            if k > 78:
                brk=1
            split_traj_metadata = split_traj_metadata_by_agents(v)
            if len(split_traj_metadata) > 1:
                brk=1
            for ag_id,ag_dat in split_traj_metadata.items():
                if update_only:
                    if k in eq_acts_in_db:
                        if ag_id in eq_acts_in_db[k]:
                            continue
                if len(ag_dat[ag_id]) == 0:
                     the subject agent has no registered action (might be due to only partial track in data), so continue 
                    continue
                if param_str == 'l2_BR_w_true_belief|l3_baseline_only':
                    sv_only = False
                else:
                    sv_only = True
                emp_action = calc_empirical_actions(k,ag_dat,sv_only)
                eq_list,sv_actions,eq_all_sv_response = calc_equilibria(k,ag_dat,'mean',status_str,param_str,emp_action)
                eq_obj = Equilibria(eq_list,sv_actions,eq_all_sv_response,emp_action,k,ag_dat,task,param_str)
                #eq_obj.update_db(['ALL_ACTIONS','ALL_ACTION_PAYOFFS'])
                #eq_obj.insert_to_db()
                if show_plots:
                    for eq_idx, eq_info in enumerate(eq_list):
                        eqs, traj_idx, eq_payoffs = eq_info[0], eq_info[1], eq_info[2]
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                        for _ag_idx,act in enumerate(eqs):
                            agent_id = int(act[6:9]) if int(act[6:9])!=0 else int(act[3:6]) 
                            file_key = L3_ACTION_CACHE+act
                            traj = utils.pickle_load(file_key)
                            #visualizer.plot_all_trajectories(traj,ax1,ax2,ax3)
                            plan_horizon_slice = step_size_secs * int(PLAN_FREQ / LP_FREQ)
                             we can cut the slice from 0 since the trajectory begins at the given timestamp (i) and not from the beginning of the scene 
                            traj_slice = traj[int(traj_idx[_ag_idx])][0][:,:plan_horizon_slice]
                            traj_vels = traj_slice[4,:] 
                            get the timestamp in ms and convert it to seconds
                            horizon_start = int(act.split('_')[1])/1000
                            horizon_end = step_size_secs + (int(act.split('_')[1])+step_size_secs)/1000
                            traj_vels_times = [horizon_start + x for x in traj_slice[0,:]]
                            #visualizer.plot_velocity(list(zip(traj_vels_times,traj_vels)),agent_id,(horizon_start,horizon_end),_ag_idx,ax4)
                        plt.show()
                        plt.clf()

        '''        

#calc_eqs_for_hopping_trajectories()