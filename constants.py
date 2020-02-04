

'''
Values derived from https://www.mdpi.com/2079-9292/8/9/943
'''
MAX_LONG_ACC_NORMAL = 1.47
MAX_LONG_ACC_AGGR = 3.5
MAX_LONG_ACC_EMG = 5

MAX_LAT_ACC_NORMAL = 4
MAX_LAT_ACC_AGGR = 5.6
MAX_LAT_ACC_EMG = 7.6

MAX_LONG_DEC_NORMAL = -2
MAX_LONG_DEC_AGGR = -3.5
MAX_LONG_DEC_EMG = -5

MAX_LAT_DEC_NORMAL = -4
MAX_LAT_DEC_AGGR = -5.6
MAX_LAT_DEC_EMG = -7.6

MAX_ACC_JERK_NORMAL = 0.9
MAX_ACC_JERK_AGGR = 2

MAX_DEC_JERK_NORMAL = -0.9
MAX_DEC_JERK_AGGR = -2

STOP_LOC_STD_DEV = 1.5
STOP_VEL_TOLERANCE = 0.5

PROCEED_VEL_MEAN = 8
PROCEED_VEL_SD = 1

MAINTAIN_VEL_SD = 2
TARGET_VEL = 14

PROCEED_ACC_SD = 0.5
PROCEED_ACC_MEAN = 1

STOP_DEC_STD_DEV = 0.01

STOP_YAW_TOLERANCE = 0.08
PROCEED_YAW_TOLERANCE = 0.15

CAR_LENGTH = 5
CAR_WIDTH = 2

N_STOP_POS_SAMPLES = {'prep-turn_s':5}
N_STOP_VEL_SAMPLES = {'prep-turn_s':5}

CAR_FOLLOWING_SAFETY_DISTANCE = 2

N_PROCEED_POS_SAMPLES = {'prep-turn_s':5,'int-entry_n':10}
N_PROCEED_VEL_SAMPLES = {'prep-turn_s':5,'int-entry_n':100}

EXIT_SEGMENTS = {'prep-turn_s':'exec-turn_s','int-entry_n':'l_n_s_l'}

LATERAL_TOLERANCE_STOPLINE = [1,0.5,-0.5,-1]
LATERAL_TOLERANCE_EXITLINE = [1,0.5,-0.5,-1]

LANE_BOUNDARY = {'prep-turn_s|proceed':'l_s_n|left_boundary'}

L1_ACTION_MAP = {'prep_turn_':['wait','go'],'exec_turn_':['wait','go'],'int_entry_':['track_speed','follow_lead'],
              'ln_n_':['track_speed','follow_lead'],
              'ln_s__':['track_speed','follow_lead'],
              'ln_n__':['track_speed','follow_lead'],
              'ln_w__':['track_speed','follow_lead'],
              'ln_w_':['track_speed','follow_lead'],
              'ln_s_':['track_speed','follow_lead'],
              'ln_e_':['track_speed','follow_lead']}
LP_FREQ = 0.1

gate_map = {'north_exit':[72,73],
                'south_exit':[18,130],
                'east_exit':[34,132],
                'west_exit':[63,131],
                'north_entry':[59,60,61,126],
                'south_entry':[17,28,29,127],
                'east_entry':[26,27,30,128],
                'west_entry':[64,65,125,129]}

#traffic_segment_seq_map = {:'d'}

RELEV_VEHS_TIME_THRESH = 1
VEH_ARRIVAL_HORIZON = 3
