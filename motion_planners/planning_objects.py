'''
Created on Feb 13, 2020

@author: Atrisha
'''

import constants

class SceneState:
    
    def __init__(self,pedestrians_info,vehicles_info):
        self.pedestrians_info = pedestrians_info
        self.vehicles_info = vehicles_info
        

class VehicleState:
    def set_track_info(self,vehicle_track_info):
        self.track_info_set = True
        self.track_id = vehicle_track_info[0,]
        self.id = int(self.track_id)
        if vehicle_track_info[1,] is None:
            print("no track for",self.track_id)
        self.x = float(vehicle_track_info[1,])
        self.y = float(vehicle_track_info[2,])
        self.speed = kph_to_mps(float(vehicle_track_info[3,]))
        self.tan_acc = float(vehicle_track_info[4,])
        self.long_acc = float(vehicle_track_info[5,])
        self.time = float(vehicle_track_info[6,])
        self.current_time = self.time
        self.yaw = float(vehicle_track_info[7,])
        self.traffic_region = vehicle_track_info[8,]
        
    def __init__(self):
        self.track_info_set = True
    
    def set_current_segment(self,segment):
        self.current_segment = segment
    
    def set_current_lane(self,current_lane):
        self.current_lane = current_lane
    
    def set_segment_seq(self,segment_seq):
        self.segment_seq = segment_seq
        
    def set_current_time(self,time):
        if time is not None:
            self.current_time = float(time)
        else:
            self.current_time = None
        
    def set_gates(self,gates):
        self.gates = gates
        
    def set_traffic_light(self,signal):
        self.signal = signal
    
    def set_entry_exit_time(self,time_tuple):
        self.entry_exit_time = time_tuple
        
    def set_id(self,id):
        self.id = id
        
    def set_gate_crossing_times(self,times):
        self.gate_crossing_times = times
        
    def set_vect_to_segment_exit(self,dist):
        self.vect_to_segment_exit= dist
        
    def set_out_of_view(self,oov):
        self.out_of_view = oov
        
    def set_full_track(self,track):
        self.track = track
    
    def set_leading_vehicle(self,leading_vehicle):
        self.leading_vehicle = leading_vehicle
    
    def set_merging_vehicle(self,merging_vehicle):
        self.merging_vehicle = merging_vehicle
    
    def set_path_origin(self,point):
        self.path_origin = point
        
    def set_current_l1_action(self,l1_action):
        self.l1_action = l1_action
        
    def set_current_l2_action(self,l2_action):
        self.l2_action = l2_action
        
    def set_direction(self,direction):
        self.direction = direction
        
    def set_task(self,task):
        self.task = task
        
    def set_time_to_next_signal(self, next_signal_change):
        self.next_signal_change = next_signal_change
        
    def set_relev_crosswalks(self,crosswalks):
        self.relev_crosswalks = crosswalks
        
    def set_relev_pedestrians(self,pedestrians):
        self.relev_pedestrians = pedestrians
        
    def set_scene_state(self,scene_state):
        self.scene_state = scene_state
    
    def set_dist_to_sub_agent(self,dist):
        self.dist_to_sv = dist

def kph_to_mps(kph):
    return kph/3.6

class TrajectoryDef:
    
    def __init__(self,string_rep):
        self.file_id = string_rep[0:2]
        self.agent_id = string_rep[3:6]
        self.relev_agent_id = string_rep[6:9]
        self.l1_action = string_rep[9:11]
        self.l2_action = string_rep[11:13]
        self.acc = False if int(self.l1_action) in [1,5,6,8] else True
        if self.acc:
            self.max_acc_long = constants.MAX_LONG_ACC_NORMAL if self.l2_action == 2 else constants.MAX_LONG_ACC_AGGR
            self.max_jerk = constants.MAX_ACC_JERK_AGGR
        else:
            self.max_acc_long = constants.MAX_LONG_DEC_NORMAL if self.l2_action == 2 else constants.MAX_LONG_DEC_AGGR
            self.max_jerk = constants.MAX_DEC_JERK_AGGR
        

class PedestrianState:
    
    def __init__(self,p_id,curr_time):
        self.curr_time = curr_time
        self.p_id = p_id
        
    def set_scene_entry_exit_times(self,time_tup):
        self.entry_time = time_tup[0]
        self.exit_time = time_tup[1]
        
    def set_gates_passed(self,gates):
        self.gates_passed = gates
        
    def set_speed(self,speed):
        self.speed = speed
        
    def set_yaw(self,yaw):
        self.yaw = yaw
        
    def set_x(self,x):
        self.x = x
    
    def set_y(self,y):
        self.y = y    
        
    def set_dist_to_gates(self,dist_dict):
        self.dist_to_gates = dist_dict
        
    def set_gate_passing_times(self,passing_times):
        self.passing_times = passing_times
        
    def set_location_info(self,location_info):
        self.location_inf = location_info
        
    def set_signal(self,signal):
        self.signal = signal
        
    def set_crosswalks(self,crosswalks):
        self.crosswalks = crosswalks
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

