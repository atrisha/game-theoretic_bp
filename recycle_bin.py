'''
Created on Mar 19, 2020

@author: Atrisha
'''
''' this method genrate plans for a vehicle from the start of its real trajectory but evolving according
to a Nash-Q equilibrium plan. Since we are calculating the Nash equilibrium exhaustively, we do not need
 to start from t=T and do backward induction. Instead we can calculate the Nash eq at every time step and
 follow an equilibrium path.'''
def generate_equilibrium_trajectories():
    ''' the use of subject vehicle in this trajectory generation is a misnomer.
    We are using the term 'subject vehicle' since the real trajectory plans were
    generated keeping in mind a single subject vehicle and getting other relevant
    agents with respect to the vehicle. '''
    subject_agent_id_str = '011'
    subject_agent_id = int(subject_agent_id_str)
    start_ts = 0
    end_ts = 12
    curr_eq_ts = start_ts
    cache_dir = '769'+subject_agent_id_str+'_'+str(start_ts+1)
    eq_dict = dict()
    init_veh_state = setup_init_scene(subject_agent_id)
    veh_state_ts = init_veh_state
    eq_tree_key = ''
    traj_pattern = '769'+str(subject_agent_id_str)+'......._'+str(start_ts)
    ''' we will work with only the mean payoff equilibria for now'''
    eq_list = cost_evaluation.calc_equilibria(constants.L3_ACTION_CACHE,traj_pattern,'mean',start_ts)
    eq_tree_key = eq_tree_key + '_' + str(start_ts)
    if eq_tree_key not in eq_dict:
        eq_dict[eq_tree_key] = eq_list
    
    for eq_idx, eq_info in enumerate(eq_dict[eq_tree_key]):
        eqs, eq_payoffs = eq_info[0], eq_info[1]
        strategies = eqs[0]
        traj_idx = eqs[1]
        while curr_eq_ts <= end_ts:
            ts = curr_eq_ts
            sv_info,list_of_rv_info = dict(),[]
            ts = ts+1
            for i,strtg_key in enumerate(strategies):
                if curr_eq_ts == 0:
                    file_key = constants.L3_ACTION_CACHE+strtg_key
                else:
                    strtg_key = strtg_key[:-2]+'_'+str(curr_eq_ts*1000)
                    file_key = os.path.join(constants.L3_ACTION_CACHE, cache_dir, strtg_key)
                traj = utils.pickle_load(file_key)
                
                ''' we will work with only the mean payoff equilibria for now'''
                #traj_type = traj[int(traj_idx[i])][1]
                plan_horizon_slice = int(constants.PLAN_FREQ / constants.LP_FREQ)
                #utils.plot_velocity(traj[int(traj_idx[i])][0][4,:plan_horizon_slice],11,(0,30))
                traj = traj[int(traj_idx[i])][0][:,plan_horizon_slice]
                ''' get the subject and relevant vehicle info into a dict
                to be used for constructing the vehicle states for planning '''
                if strtg_key[6:9] == '000':
                    ''' this is the subject vehicle '''
                    agent_id = int(strtg_key[3:6])
                    sv_info['id'] = agent_id
                    sv_info['current_time'] = ts
                    sv_info['trajectories'] = traj
                else:
                    rv_info = dict()
                    relev_agent_id = int(strtg_key[6:9])
                    rv_info['id'] = relev_agent_id
                    rv_info['current_time'] = ts
                    rv_info['trajectories'] = traj
                    list_of_rv_info.append(rv_info)
            '''
            #cache_dir = '769'+subject_agent_id_str+'_'+str(ts)
            veh_state_ts_plus_one = generate_simulation_action_plans(veh_state_ts,sv_info, list_of_rv_info,cache_dir)
            veh_state_ts = veh_state_ts_plus_one
            traj_pattern = '769'+str(subject_agent_id_str)+'......._'+str(ts)
            equilibria_t_plus_one = cost_evaluation.calc_equilibria(os.path.join(constants.L3_ACTION_CACHE, cache_dir),traj_pattern,'mean')
            dict_key = str(curr_eq_ts)+'('+str(eq_idx)+')'+'_'+str(ts)
            strategies = equilibria_t_plus_one[0]
            traj_idx = equilibria_t_plus_one[1]
            eq_dict[dict_key] = equilibria_t_plus_one
            #utils.clear_cache(os.path.join(constants.L3_ACTION_CACHE, cache_dir))
            curr_eq_ts = curr_eq_ts + 1
            '''




def generate_simulation_action_plans(veh_state_prev,sv_info,list_of_rv_info,cache_dir):
    ''' expect a veh_state from previous state, and dict for subject vehicle and for rv with the trajectory ndarray as an entry in the info dict'''
    veh_state = VehicleState()
    veh_state.set_id(veh_state_prev.id)
    agent_id = veh_state.id
    veh_state.track_info_set = False
    veh_state.track_id = veh_state.id
    ''' this has already been incremented '''
    curr_time = sv_info['current_time']
    veh_state.set_current_time(veh_state_prev.current_time+constants.PLAN_FREQ)
    veh_state.x = sv_info['trajectories'][1]
    veh_state.y = sv_info['trajectories'][2]
    veh_state.prev_pos = (veh_state_prev.x,veh_state_prev.y)
    veh_state.speed = sv_info['trajectories'][4]
    veh_state.long_acc = sv_info['trajectories'][5]
    ''' fix this '''
    veh_state.tan_acc = 0
    veh_state.time = sv_info['current_time']
    veh_state.yaw = sv_info['trajectories'][3]
    veh_state.action_plans = dict(veh_state_prev.action_plans)
    relev_agents_dict = dict()
    ''' keep the same relevant agents as the scene initialization. This is because it's not possible to predict the arrival of an agent
    in an hypothetical simulation world that started from the initial scene but did not happen. Keeping the same agents as the scene
    seems most reasonable at this point.'''
    for rv_info in list_of_rv_info:
        r_a_state = VehicleState()
        r_a_state.set_id(rv_info['id'])
        r_a_state.track_info_set = False
        r_a_state.track_id = rv_info['id']
        r_a_state.x = rv_info['trajectories'][1]
        r_a_state.y = rv_info['trajectories'][2]
        r_a_state.speed = rv_info['trajectories'][4]
        r_a_state.long_acc = rv_info['trajectories'][5]
        ''' fix this '''
        r_a_state.tan_acc = 0
        r_a_state.time = rv_info['current_time']
        r_a_state.set_current_time(r_a_state.time)
        r_a_state.yaw = rv_info['trajectories'][3]
        r_a_state.action_plans = dict()
        relev_agents_dict[r_a_state.track_id] = r_a_state
    track_segment_seq = utils.get_track_segment_seq(veh_state.id)
    veh_state.set_segment_seq(track_segment_seq)
    path,gates,direction = utils.get_path_gates_direction(None,veh_state.id)
    task = constants.TASK_MAP[direction]
    veh_state.set_gates(gates)
    gate_crossing_times = utils.gate_crossing_times(veh_state)
    veh_state.set_gate_crossing_times(gate_crossing_times)
    ''' this is the segment from previous time stamp at this point'''
    veh_state.current_segment = veh_state_prev.current_segment
    ''' segment updated '''
    current_segment = utils.assign_curent_segment(None, veh_state, True)
    veh_state.set_current_segment(current_segment)
    veh_direction = 'L_'+track_segment_seq[0][3].upper()+'_'+track_segment_seq[-1][3].upper()
    if current_segment[0:2] == 'ln':
        veh_current_lane = current_segment
    elif 'int-entry' in current_segment:
        dir_key = str(track_segment_seq[0][3:])+'-'+track_segment_seq[-1][3:]
        veh_current_lane = constants.LANE_MAP[dir_key]
    else:
        veh_current_lane = veh_direction
    veh_state.set_current_lane(veh_current_lane)
        
    if curr_time not in veh_state.action_plans:
        veh_state.action_plans[curr_time] = dict()
    actions = utils.get_actions(veh_state)
    actions_l1 = list(actions.keys())
    for l1 in actions_l1:
        actions_l2 = actions[l1]
        for l2 in actions_l2:
            if l1 not in veh_state.action_plans[curr_time]:
                veh_state.action_plans[curr_time][l1] = dict()
            veh_state.action_plans[curr_time][l1][l2] = None
            trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,task)
            trajectory_plan.set_lead_vehicle(None)
            ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
            cache_dir_key = cache_dir
            file_key = os.path.join(constants.L3_ACTION_CACHE, cache_dir_key, get_l3_action_file(None, agent_id, 0, curr_time, l1, l2))
            print('time',curr_time,'agent',agent_id,l1,l2)
            if not os.path.isfile(file_key):
                l3_actions = trajectory_plan.generate_trajectory(veh_state)
                if len(l3_actions) > 0:
                    utils.pickle_dump(file_key, l3_actions)
            else:
                l3_actions = utils.pickle_load(file_key)
            veh_state.action_plans[curr_time][l1][l2] = np.copy(l3_actions)
            
    
    for r_a_id, r_a_state in relev_agents_dict.items():
        if r_a_state.id == agent_id:
            continue
        
        r_a_track_segment_seq = utils.get_track_segment_seq(r_a_state.id)
        r_a_state.set_segment_seq(r_a_track_segment_seq)
        r_a_path,r_a_gates,r_a_direction = utils.get_path_gates_direction(None,r_a_state.id)
        r_a_task = constants.TASK_MAP[r_a_direction]
        r_a_state.set_gates(r_a_gates)
        r_a_gate_crossing_times = utils.gate_crossing_times(r_a_state)
        r_a_state.set_gate_crossing_times(r_a_gate_crossing_times)
        
        ''' get relevant state attributes from previous step '''
        lead_vehicle = None
        for _r in veh_state.action_plans[curr_time-1]['relev_agents']:
            if _r.id == r_a_state.id:
                ''' this is the segment from previous time stamp at this point'''
                r_a_state.current_segment = _r.current_segment
                lead_vehicle = _r.leading_vehicle
                break
            r_a_state.current_segment = None
        if r_a_state.current_segment is None:
            sys.exit('previous state segment is not set')
        ''' segment update '''
        r_a_current_segment = utils.assign_curent_segment(None, r_a_state, True)
        r_a_state.set_current_segment(r_a_current_segment)
        r_a_direction = 'L_'+r_a_track_segment_seq[0][3].upper()+'_'+r_a_track_segment_seq[-1][3].upper()
        if r_a_current_segment[0:2] == 'ln':
            r_a_current_lane = r_a_current_segment
        elif 'int-entry' in r_a_current_segment:
            dir_key = str(r_a_track_segment_seq[0][3:])+'-'+r_a_track_segment_seq[-1][3:]
            r_a_current_lane = constants.LANE_MAP[dir_key]
        else:
            r_a_current_lane = r_a_direction
        r_a_state.set_current_lane(r_a_current_lane)
        ''' if the lead vehicle is in the list of relevant agents, it's trajectory is already available, so load from that.
        Otherise, use a constant acceleration model to project the trajectory. '''
        if lead_vehicle is not None:
            if lead_vehicle.id in relev_agents_dict.keys():
                lead_vehicle = relev_agents_dict[lead_vehicle.id]
            else:
                lead_vehicle = setup_lead_vehicle(lead_vehicle,False)
        r_a_state.set_leading_vehicle(lead_vehicle)
        if curr_time not in r_a_state.action_plans:
            r_a_state.action_plans[curr_time] = dict()
        r_a_actions = utils.get_actions(r_a_state)
        r_a_actions_l1 = list(r_a_actions.keys())
        for l1 in actions_l1:
            if (lead_vehicle is None and l1=='follow_lead') or (lead_vehicle is not None and l1=='track_speed'):
                continue
            actions_l2 = r_a_actions[l1]
            for l2 in actions_l2:
                ''' cache directory for simulation file_id(3),agent_id(3)_currtime'''
                
                ''' l3_action_trajectory file id: file_id(3),agent_id(3),relev_agent_id(3),l1_action(2),l2_action(2)'''
                cache_dir_key = cache_dir
                file_key = os.path.join(constants.L3_ACTION_CACHE, cache_dir_key, get_l3_action_file(None, agent_id, r_a_state.id, curr_time, l1, l2))
                if l1 not in r_a_state.action_plans[curr_time]:
                    r_a_state.action_plans[curr_time][l1] = dict()
                    r_a_state.action_plans[curr_time][l1][l2] = None
                trajectory_plan = motion_planner.TrajectoryPlan(l1,l2,r_a_task)
                trajectory_plan.set_lead_vehicle(lead_vehicle)
                if lead_vehicle is not None:
                    print('time',curr_time,'agent',agent_id,'relev agent',r_a_state.id,l1,'(lv:'+str(lead_vehicle.id)+')',l2)
                else:
                    print('time',curr_time,'agent',agent_id,'relev agent',r_a_state.id,l1,l2)
                if not os.path.isfile(file_key):
                    l3_actions = trajectory_plan.generate_trajectory(r_a_state)
                    if len(l3_actions) > 0:
                        utils.pickle_dump(file_key, l3_actions)
                else:
                    l3_actions = utils.pickle_load(file_key)
                l3_action_size = l3_actions.shape[0] if l3_actions is not None else 0
                r_a_state.action_plans[curr_time][l1][l2] = np.copy(l3_actions)
                    
                
        if 'relev_agents' not in veh_state.action_plans[curr_time]:
            veh_state.action_plans[curr_time]['relev_agents'] = [r_a_state]
        else:
            veh_state.action_plans[curr_time]['relev_agents'].append(r_a_state)
    #print(relev_agents)
    return veh_state

    



''' this set's up the initial scene from an agent's perspective from the real data and returns
the veh_state object for that vehicle.'''
def setup_init_scene(agent_id):
    ''' veh_state object maintains details about an agent'''
    veh_state = motion_planner.VehicleState()
    veh_state.set_id(agent_id)
    veh_state.set_current_time(0)
    
    ''' find the sequence of segments of the agent. This defines its path. '''
    track_region_seq = utils.get_track_segment_seq(veh_state.id)
    veh_state.set_segment_seq(track_region_seq)
    
    ''' get the agent's trajectory'''
    agent_track = utils.get_track(veh_state,None)
    veh_state.set_full_track(agent_track)
    veh_state.action_plans = dict()
    veh_state.origin = (float(agent_track[0,1]),float(agent_track[0,2]))
    track_info = agent_track[0]
    curr_time = float(track_info[6,])
    init_veh_state = generate_action_plans(veh_state,0)
    return init_veh_state



def get_simulation_cache_dir(file_id,agent_id,curr_time):
    curr_time = round(float(curr_time)*1000)
    file_id = '769'
    agent_id = str(agent_id).zfill(3)
    dir_key = file_id+agent_id+'_'+str(curr_time)
    return dir_key