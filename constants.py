

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

MAX_SAMPLES_TRACK_SPEED = 20
MAX_SAMPLES_FOLLOW_LEAD = 20

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

N_PROCEED_POS_SAMPLES = {'prep-turn_s':5,'int-entry_n':10,'exec-turn_s':5}
N_PROCEED_VEL_SAMPLES = {'prep-turn_s':5,'int-entry_n':100,'exec-turn_s':5}

EXIT_SEGMENTS = {'prep-turn_s':'exec-turn_s','int-entry_n':'l_n_s_l','exec-turn_s':'ln_w_-1'}

LATERAL_TOLERANCE_STOPLINE = [1,0.5,-0.5,-1]
LATERAL_TOLERANCE_EXITLINE = [1,0.5,-0.5,-1]

LANE_BOUNDARY = {'prep-turn_s|proceed':'l_s_n|left_boundary'}

TASK_MAP = {'L_N_S':'STRAIGHT',
            'L_N_W':'DEDICATED_RIGHT_TURN',
            'L_N_E':'LEFT_TURN',
            'L_S_N':'STRAIGHT',
            'L_S_W':'LEFT_TURN',
            'L_S_E':'RIGHT_TURN',
            'L_E_W':'STRAIGHT',
            'L_E_N':'DEDICATED_RIGHT_TURN',
            'L_E_S':'LEFT_TURN',
            'L_W_E':'STRAIGHT',
            'L_W_N':'LEFT_TURN',
            'L_W_S':'RIGHT_TURN',
            }

LANE_MAP = {'n_2-s_-1':'l_n_s_l',
            'n_3-s_-2':'l_n_s_r',
            's_3-n_-2':'l_s_n_r',
            's_2-n_-1':'l_s_n_l',
            'e_2-w_-1':'l_e_w_l',
            'e_3-w_-2':'l_e_w_e',
            'w_3-e_-2':'l_w_e_r',
            'w_2-e_-1':'l_w_e_l',
            }

L1_ACTION_MAP = {'prep_turn_':['wait','proceed'],'exec_turn_':['proceed'],'int_entry_':['track_speed','follow_lead'],
              'ln_n_':['track_speed','follow_lead'],
              'ln_s__':['track_speed','follow_lead'],
              'ln_n__':['track_speed','follow_lead'],
              'ln_w__':['track_speed','follow_lead'],
              'ln_w_':['track_speed','follow_lead'],
              'ln_s_':['track_speed','follow_lead'],
              'ln_e_':['track_speed','follow_lead']}
LP_FREQ = 0.1
PLAN_FREQ = 1
DATASET_FPS = 30

colors = ['k','g','r','c','m','b','w']

L1_ACTION_CODES = {'wait':1,
                   'proceed':2,
                   'track_speed':3,
                   'follow_lead':4}

L1_ACTION_CODES_2_NAME = {1:'wait',
                   2:'proceed',
                   3:'track_speed',
                   4:'follow_lead'}

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

L2_ACTION_MAP = {'wait':['AGGRESSIVE','NORMAL'],
                 'proceed':['AGGRESSIVE','NORMAL'],
                 'track_speed':['AGGRESSIVE','NORMAL'],
                 'follow_lead':['AGGRESSIVE','NORMAL']}

L2_ACTION_CODES = {'AGGRESSIVE':1,
                   'NORMAL':2} 

L2_ACTION_CODES_2_NAME = {1:'AGGRESSIVE',
                   2:'NORMAL'}

MAX_L3_ACTIONS = 10

L3_ACTION_CACHE = 'l3_action_trajectories/'

DIST_COST_MEAN = 3
DIST_COST_SD = 0.5

SPEED_COST_MEAN = 11
SPEED_COST_SD = 5

INHIBITORY_PAYOFF_WEIGHT = 0.9
EXCITATORY_PAYOFF_WEIGHT = 0.1
L2_ACTION_PAYOFF_ADDITIVE = 0.3

''' This is kept here so that we don't have to query the db everytime'''
VIEWPORT = ([538782.42,538861.33,538892.67,538811.68],[4814012.83,4814060.05,4814013.85,4813967.66])