import unittest
import all_utils
from utils import utils
import ast
import sqlite3
import matplotlib.pyplot as plt
import numpy as np


class BaselineTrajectoryGenerationTest(unittest.TestCase):
    
    def test_dummy(self):
        self.assertEqual(1, 1, '1 is 1 passed')
    
    def test_velocity_profile(self):
        all_l1_actions = []
        conn_trajdb = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_generated_trajectories_'+constants.CURRENT_FILE_ID+'.db')
        c = conn.cursor()
        conn1 = sqlite3.connect(constants.DB_DIR+constants.CURRENT_FILE_ID+'\\intsc_data_'+constants.CURRENT_FILE_ID+'.db')
        c1 = conn1.cursor()
        ag_id = 43
        q_string = "select distinct l1_action from GENERATED_TRAJECTORY_INFO where agent_id="+str(ag_id)
        c.execute(q_string)
        res = c.fetchall()
        all_l1_actions = [row[0] for row in res]        
        emp_path = all_utils.get_path(ag_id)
        #plt.plot([x[0] for x in emp_path],[x[1] for x in emp_path],'blue')
        for l1_act in all_l1_actions:
            q_string = "SELECT DISTINCT TRAJ_ID FROM GENERATED_TRAJECTORY_INFO where l1_action='"+l1_act+"'"
            c.execute(q_string)
            res = tuple(row[0] for row in c.fetchall())
            
            q_string = "select * from GENERATED_BASELINE_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(res))+" order by trajectory_id,time"
            print(q_string)
            c.execute(q_string)
            traj_dict = dict()
            res = c.fetchall()
            ct = 0
            for row in res:
                ct += 1
                if row[0] not in traj_dict:
                    traj_dict[row[0]] = [ [ row[3] ], [ row[4] ], [ row[6] ]]
                else:
                    traj_dict[row[0]][0].append(row[3])
                    traj_dict[row[0]][1].append(row[4])
                    traj_dict[row[0]][2].append(row[6])
            for k,v in traj_dict.items():
                print(k)
                plt.plot(np.arange(len(v[2])),v[2],color='black',linewidth=0.25)
            plt.title(l1_act+'-velocity')
            plt.show()
        conn.close()
        
        self.assertEqual(1, 1, '1 is 1 passed')