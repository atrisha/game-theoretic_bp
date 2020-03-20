

'''
Values derived from https://www.mdpi.com/2079-9292/8/9/943
'''
import numpy as np
MAX_LONG_ACC_NORMAL = 1.5
MAX_LONG_ACC_AGGR = 3.6
MAX_LONG_ACC_EMG = 5

MAX_TURN_ACC_AGGR = 5
MAX_TURN_ACC_NORMAL = 3.6
MAX_TURN_JERK = 3

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

PROCEED_VEL_MEAN_EXEC_TURN = 8
PROCEED_VEL_SD_EXEC_TURN = 1
PROCEED_VEL_AGGRESSIVE_ADDITIVE = 1
PROCEED_VEL_MEAN_PREP_TURN = 7
PROCEED_VEL_SD_PREP_TURN = 1

PROCEED_VEL_MEAN_EXIT = 11
PROCEED_VEL_SD_EXIT = 2

CURVATURE_LIMIT_FOR_CV = 0.06




MAX_SAMPLES_TRACK_SPEED = 20
MAX_SAMPLES_FOLLOW_LEAD = 20
MAX_SAMPLES_DEC_TO_STOP = 20

MAINTAIN_VEL_SD = 2
TARGET_VEL = 8.5
LEFT_TURN_VEL_START_POS = 7
LEFT_TURN_VEL_START_POS_AGGR_ADDITIVE = 1
TARGET_VEL_SD = 1

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

N_PROCEED_POS_SAMPLES = {'ln_s_1':20,'prep-turn_s':20,'int-entry_n':10,'exec-turn_s':20}
N_PROCEED_VEL_SAMPLES = {'ln_s_1':20,'prep-turn_s':20,'int-entry_n':100,'exec-turn_s':20}

EXIT_SEGMENTS = {'prep-turn_s':'exec-turn_s','int-entry_n':'l_n_s_l','exec-turn_s':'ln_w_-1'}

LATERAL_TOLERANCE_STOPLINE = np.arange(-.25,6.25,.25)
LATERAL_TOLERANCE_EXITLINE = [1,0.5,-0.5,-1]
LATERAL_TOLERANCE_WAITLINE = np.arange(1,-7,-0.5)
LATERAL_TOLERANCE_DISTANCE_GAPS = np.arange(0,6.25,.25)

LANE_BOUNDARY = {'prep-turn_s|proceed-turn':'l_s_n|left_boundary'}
LANE_WIDTH = 4

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

SEGMENT_MAP = {'ln_s_1':'left-turn-lane',
               'prep-turn_s':'prep-left-turn',
               'exec-turn_s':'exec-left-turn',
               'ln_w_-1':'exit-lane',
               'ln_w_-2':'exit-lane',
               'ln_n_1':'left-turn-lane',
               'ln_n_2':'through-lane-entry',
               'ln_n_3':'through-lane-entry',
               'ln_s_-1':'exit-lane',
               'ln_s_-2':'exit-lane',
               'l_n_s_l':'through-lane',
               'l_n_s_r':'through-lane',
               'l_s_n_l':'through-lane',
               'l_s_n_r':'through-lane'}


LP_FREQ = 0.1
PLAN_FREQ = 2
DATASET_FPS = 30

colors = ['k','g','r','c','m','b','w']

L1_ACTION_CODES = {'wait-for-oncoming':1,
                   'proceed-turn':2,
                   'track_speed':3,
                   'follow_lead':4,
                   'decelerate-to-stop':5,
                   'wait_for_lead_to_cross':6,
                   'follow_lead_into_intersection':7,
                   'wait-on-red':8}

L1_ACTION_CODES_2_NAME = {1:'wait',
                   2:'proceed-turn',
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