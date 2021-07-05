'''
Created on Jun 8, 2021

@author: Atrisha
'''
import csv
import numpy as np
import sqlite3

import sys
import constants
from motion_planners.planning_objects import VehicleState
import ast
import matplotlib.pyplot as plt
#import all_utils
from all_utils import utils
from all_utils.utils import reduce_relev_agents
import os
from os import listdir
from collections import OrderedDict
import itertools
import io
log = constants.common_logger



def upgrade_db():
    new_db_dir = 'D:\\intersections_dataset\\database_update'
    old_db_dir = 'D:\\intersections_dataset\\dataset'
    tables_to_upgrade = {'RELEVANT_AGENTS':{'create':"CREATE TABLE IF NOT EXISTS RELEVANT_AGENTS(TRACK_ID INT, TIME NUMERIC, LEAD_VEHICLE INT, RELEV_AGENT_IDS TEXT)",
                                            'insert':"INSERT INTO RELEVANT_AGENTS VALUES (?,?,?,?)"}
                         }
    for table_name in tables_to_upgrade.keys():
        for file_id in constants.ALL_FILE_IDS:
            new_conn = sqlite3.connect(os.path.join(new_db_dir,file_id,'uni_weber_'+file_id+'.db'))
            old_conn = sqlite3.connect(os.path.join(old_db_dir,file_id,'uni_weber_'+file_id+'.db'))
            c = old_conn.cursor()
            c_newconn = new_conn.cursor()
            q_string = "SELECT name FROM sqlite_master WHERE type='table' AND name='"+table_name+"'"
            c.execute(q_string)
            res = c.fetchall()
            if len(res) == 0:
                q_string = tables_to_upgrade[table_name]['create']
                c.execute(q_string)
                old_conn.commit()
            q_string = "SELECT * FROM RELEVANT_AGENTS"
            c.execute(q_string)
            old_data = c.fetchall()
            c_newconn.execute(q_string)
            new_data = c_newconn.fetchall()
            ins_list = []
            N = len(new_data)
            for idx,new_row in enumerate(new_data):
                if new_row not in old_data:
                    print('processing',file_id,idx,'/',N)
                    ins_list.append(new_row)
            old_conn.executemany(tables_to_upgrade[table_name]['insert'],ins_list)
            old_conn.commit()
            old_conn.close()
            f=1
                    
                
            
