

'''
Values derived from https://www.mdpi.com/2079-9292/8/9/943
'''
import numpy as np
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR,'logs')
DB_DIR = "D:\\intersections_dataset\\dataset\\"
CACHE_DIR = "F:\\Spring2017\\workspaces\\game_theoretic_planner_cache\\"

MAX_LONG_ACC_NORMAL = 2
MAX_LONG_ACC_AGGR = 3.6
MAX_LONG_ACC_EMG = 5

MAX_TURN_ACC_AGGR = 5
MAX_TURN_ACC_NORMAL = 3.6

MAX_TURN_JERK = 3
MAX_TURN_JERK_NORMAL = 1.5
MAX_TURN_JERK_AGGR = 2.5

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

PROCEED_VEL_MEAN_EXIT = 12
PROCEED_VEL_SD_EXIT = 2

CURVATURE_LIMIT_FOR_CV = 0.06


TRACK_SPEED_VEL_LIMIT = 20

MAX_SAMPLES_TRACK_SPEED = 20
MAX_SAMPLES_FOLLOW_LEAD = 20
MAX_SAMPLES_DEC_TO_STOP = 20

MAINTAIN_VEL_SD = 2
TARGET_VEL = 8.5
LEFT_TURN_VEL_START_POS = 7
LEFT_TURN_VEL_START_POS_AGGR_ADDITIVE = 1
TARGET_VEL_SD = 3

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

SEGMENT_MAP = {
               'prep-turn_s':'prep-left-turn',
               'prep-turn_n':'prep-left-turn',
               'prep-turn_e':'prep-left-turn',
               'exec-turn_e':'exec-left-turn',
               'exec-turn_n':'exec-left-turn',
               'exec-turn_s':'exec-left-turn',
               'prep-turn_w':'prep-left-turn',
               'exec-turn_w':'exec-left-turn',
               'rt_prep-turn_s':'prep-right-turn',
               'rt_exec-turn_s':'exec-right-turn',
               'rt_prep-turn_w':'prep-right-turn',
               'rt_exec-turn_w':'exec-right-turn',
               'ln_w_-2':'exit-lane',
               'ln_w_-1':'exit-lane',
               'ln_w_1':'left-turn-lane',
               'ln_w_2':'through-lane-entry',
               'ln_w_3':'through-lane-entry',
               'ln_w_4':'right-turn-lane',
               'ln_n_-2':'exit-lane',
               'ln_n_-1':'exit-lane',
               'ln_n_1':'left-turn-lane',
               'ln_n_2':'through-lane-entry',
               'ln_n_3':'through-lane-entry',
               'ln_s_-2':'exit-lane',
               'ln_s_-1':'exit-lane',
               'ln_s_1':'left-turn-lane',
               'ln_s_2':'through-lane-entry',
               'ln_s_3':'through-lane-entry',
               'ln_s_4':'right-turn-lane',
               'ln_e_-2':'exit-lane',
               'ln_e_-1':'exit-lane',
               'ln_e_1':'left-turn-lane',
               'ln_e_2':'through-lane-entry',
               'ln_e_3':'through-lane-entry',
               'l_n_s_l':'through-lane',
               'l_n_s_r':'through-lane',
               'l_s_n_l':'through-lane',
               'l_s_n_r':'through-lane',
               'l_e_w_l':'through-lane',
               'l_e_w_r':'through-lane',
               'l_w_e_l':'through-lane',
               'l_w_e_r':'through-lane',
               'ln_n_4':'right-turn-lane',
               'l_n_w':'exec-right-turn'
               }

STOP_SEGMENT = {'wait-for-oncoming':['exec-left-turn','exec-right-turn'],
                'decelerate-to-stop':['prep-right-turn','prep-left-turn','through-lane'],
                'wait-on-red':['prep-right-turn','prep-left-turn','through-lane'],
                'wait-for-pedestrian':['prep-right-turn','exec-left-turn','through-lane']}

ENTRY_LANES = ['left-turn-lane', 'right-turn-lane', 'through-lane-entry']

''' direction:[(relev_agent_segment, relev_agent_direction, number of relev agents to consider, maximum distance of the relev agent)] P.S: exit_lane is the exit lane in the current subject vehicle's direction'''
RELEV_REDUCTION_MAP = {'LEFT_TURN':[('through-lane','l',2,None),('through-lane','r',2,None), ('through-lane-entry','2',1,None), ('through-lane-entry','3',1,None)],
                       'RIGHT_TURN':[('through-lane','l',2,None),('through-lane','r',2,None), ('through-lane-entry','3',1,None),
                                     ('exec-left-turn',None,1,None),('prep-left-turn',None,1,None)]}

PATH_COLLISION_MAP = {'L_S_W':['ln_w_1','prep-turn_w'],
                      'L_N_E':['ln_e_1','prep-turn_e'],
                      'L_E_S':['ln_s_1','prep-turn_s'],
                      'L_W_N':['ln_n_1','prep-turn_n']}

LP_FREQ = 0.1
PLAN_FREQ = 1
DATASET_FPS = 30
OTH_AGENT_L3_ACT_HORIZON = 5
PLAN_HORIZON_SECS = 5

colors = ['k','g','r','c','m','b','w']
BEFORE_CROSSWALK = 'BEFORE_CROSSWALK' 
ON_CROSSWALK = 'ON_CROSSWALK'
AFTER_CROSSWALK = 'AFTER_CROSSWALK' 

L1_ACTION_CODES = {'wait-for-oncoming':1,
                   'proceed-turn':2,
                   'track_speed':3,
                   'follow_lead':4,
                   'decelerate-to-stop':5,
                   'wait_for_lead_to_cross':6,
                   'follow_lead_into_intersection':7,
                   'wait-on-red':8,
                   'cut-in':9,
                   'yield-to-merging':10,
                   'wait-for-pedestrian':11}

WAIT_ACTIONS = ['yield-to-merging','wait_for_lead_to_cross','wait-for-oncoming','decelerate-to-stop','wait-on-red','wait-for-pedestrian']
PROCEED_ACTIONS = ['proceed-turn','track_speed','follow_lead','follow_lead_into_intersection','cut-in']

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
LEAD_VEH_DIST_THRESH = 50
VEH_ARRIVAL_HORIZON = 3

PEDESTRIAN_CROSSWALK_DIST_THRESH = 10
PEDESTRIAN_CROSSWALK_TIME_THRESH = 3

LATERAL_LATTICE_TOLERANCE = {'DEFAULT':1.16,
                             'exec-left-turn':2.5,
                             'prep-left-turn':1.9,
                             'exit-lane':2.8}
NUM_LATERAL_LATTICE_POINTS = 10

COLLISION_CHECK_TOLERANCE = 1.5

L2_ACTION_CODES = {'AGGRESSIVE':1,
                   'NORMAL':2} 

L2_ACTION_CODES_2_NAME = {1:'AGGRESSIVE',
                   2:'NORMAL'}

MAX_L3_ACTIONS = 10

VEH_CATEGORIES = "('Car','Medium Vehicle','Heavy Vehicle','Motorcycle','Bus')"
#VEH_CATEGORIES = "('Bus')"
TURN_TIME_MAP = {'ln_w_4':[(0,7.17),(50,0)],
                 'ln_s_4':[(0,5.3),(25,0)],
                 'ln_n_1':[(0,8),(45,0)],
                 'ln_e_1':[(17,11),(60,0)],
                 'ln_w_1':[(10,10),(55,0)],
                 'ln_s_1':[(0,8),(40,0)],
                 'ln_s_2':[(0,8),(40,0)]}

CROSS_PATH_CONFLICTS = {('L_S_W','L_N_S'):['l_n_s_l','l_n_s_r'],
                        ('L_E_S','L_W_E'):['l_w_e_l','l_w_e_r'],
                        ('L_N_E','L_S_N'):['l_s_n_l','l_s_n_r'],
                        ('L_W_N','L_E_W'):['l_e_w_l','l_e_w_r']}

MERGE_PATH_CONFLICTS = {('L_E_S','L_W_S'):['ln_s_-1','ln_s_-2'],
                        ('L_N_E','L_S_E'):['ln_e_-1','ln_e_-2'],
                        ('L_W_S','L_N_S'):['ln_s_-1','ln_s_-2'],
                        ('L_S_E','L_W_E'):['ln_e_-1','ln_e_-2']}

RESULTS = None

DIST_COST_MEAN = 3
DIST_COST_SD = 0.5

SPEED_COST_MEAN = 11
SPEED_COST_SD = 5

MAX_FP = 6


INHIBITORY_PAYOFF_WEIGHT = 0.5
EXCITATORY_PAYOFF_WEIGHT = 0.5
INHIBITORY_PEDESTRIAN_PAYOFF_WEIGHT = 0.5

WEIGHTS_FLAG = None


''' This is kept here so that we don't have to query the db everytime'''
VIEWPORT = ([538782.42,538861.33,538892.67,538811.68],[4814012.83,4814060.05,4814013.85,4813967.66])
CROSSWALK_GATES = [(20,21),(36,37),(68,69),(66,67)]
CROSSWALK_SIGNAL_MAP = {'P_20_21':'L_W_E',
                        'P_36_37':'L_S_N',
                        'P_68_69':'L_E_W',
                        'P_66_67':'L_N_S'}

''' equilibrium analysis configurations '''
BASELINE_TRAJECTORIES_ONLY = False

INHIBITORY = True
EXCITATORY = True
INHIBITORY_PEDESTRIAN = True
L1_EQ_TYPE = None
L3_EQ_TYPE = None


BUILD_L3_TREE = True

TRAJECTORY_TYPE = None


CURRENT_FILE_ID = None
ALL_FILE_IDS = ['769','770','771'] + [str(x) for x in np.arange(775,786)]
TEMP_TRAJ_CACHE = None
L3_ACTION_CACHE = None
import logging
import logging.handlers
format='%(levelname)-8s %(funcName)s : %(message)s'
logging.basicConfig(format=format,level=logging.INFO)
logger = logging.getLogger("pylog")
logger.setLevel(logging.INFO)
eq_logger = logging.getLogger("pylog")
eq_logger.setLevel(logging.INFO)
common_logger = logger
show_plots = False

def setup_logger():
    logging.basicConfig(format=format,level=logging.INFO)
    logger = logging.getLogger("pylog")
    logger.setLevel(logging.INFO)
    eq_logger = logging.getLogger("pylog")
    eq_logger.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(os.path.join(LOG_DIR,CURRENT_FILE_ID+'_run.log'),maxBytes=50000000,backupCount=20)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(format))
    fheq = logging.handlers.RotatingFileHandler(os.path.join(LOG_DIR,CURRENT_FILE_ID+'_run-eq.log'),maxBytes=50000000,backupCount=20)
    fheq.setLevel(logging.INFO)
    fheq.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)
    eq_logger.addHandler(fheq)
    common_logger = logger
