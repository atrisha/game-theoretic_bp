'''
Created on Feb 13, 2020

@author: Atrisha
'''

import constants


class VehicleState:
    def set_track_info(self,vehicle_track_info):
        self.track_info_set = True
        self.track_id = vehicle_track_info[0,]
        self.x = float(vehicle_track_info[1,])
        self.y = float(vehicle_track_info[2,])
        self.speed = kph_to_mps(float(vehicle_track_info[3,]))
        self.tan_acc = float(vehicle_track_info[4,])
        self.long_acc = float(vehicle_track_info[5,])
        self.time = float(vehicle_track_info[6,])
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
        
    def set_dist_to_segment_exit(self,dist):
        self.dist_to_segment_exit= dist
        
    def set_out_of_view(self,oov):
        self.out_of_view = oov
        
    def set_full_track(self,track):
        self.track = track
    
    def set_leading_vehicle(self,leading_vehicle):
        self.leading_vehicle = leading_vehicle
    
    def set_path_origin(self,point):
        self.path_origin = point

def kph_to_mps(kph):
    return kph/3.6

class TrajectoryDef:
    
    def __init__(self,string_rep):
        self.file_id = string_rep[0:2]
        self.agent_id = string_rep[3:6]
        self.relev_agent_id = string_rep[6:9]
        self.l1_action = string_rep[9:11]
        self.l1_action_readable = constants.L1_ACTION_CODES_2_NAME[int(string_rep[9:11])]
        self.l2_action = string_rep[11:13]
        self.l2_action_readable = constants.L2_ACTION_CODES_2_NAME[int(string_rep[11:13])]
        acc = False if self.l1_action_readable == 'wait' else True
        if acc:
            self.max_acc_long = constants.MAX_LONG_ACC_NORMAL if self.l2_action_readable is 'NORMAL' else constants.MAX_LONG_ACC_AGGR
            self.max_jerk = constants.MAX_ACC_JERK_AGGR
        else:
            self.max_acc_long = constants.MAX_LONG_DEC_NORMAL if self.l2_action_readable is 'NORMAL' else constants.MAX_LONG_DEC_AGGR
            self.max_jerk = constants.MAX_DEC_JERK_AGGR
        

