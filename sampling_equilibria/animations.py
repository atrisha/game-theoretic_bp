'''
Created on Sep 22, 2020

@author: Atrisha
'''
import matplotlib
#matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import constants
from all_utils import utils
from all_utils.utils import kph_to_mps
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
log = constants.common_logger
from collections import OrderedDict
import itertools
import math
import sys
import ast
import os
import pickle
  
OBJ_CACHE_DIR = "D:\\temp_cache\\"
HORIZON = 1

def pickle_dump(file_key,obj_dump):
    file_key = os.path.join(OBJ_CACHE_DIR,file_key)
    pickle.dump( obj_dump, open( file_key, "wb" ) )

class SceneAnimation:
    
    def setup_animation(self):
        file_id = '769'
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_'+file_id+'.db')
        c = conn.cursor()
        conn_traj = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_generated_trajectories_'+file_id+'.db')
        c_traj = conn_traj.cursor()
            
        all_files = os.listdir(OBJ_CACHE_DIR)
        if file_id+'_acts_dict' in all_files :
            acts_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_acts_dict'), "rb" ) )
            self.acts_dict = acts_dict
        else:
            q_string = "select * from L1_ACTIONS"
            c.execute(q_string)
            res = c.fetchall()                
            self.acts_dict = dict()
            for row in res:
                if (round(row[0]),row[1]) not in self.acts_dict:
                    self.acts_dict[(round(row[0]),row[1])] = [utils.get_l1_action_string(int(x[9:11])) for x in ast.literal_eval(row[2])]
            pickle_dump(file_id+'_acts_dict',self.acts_dict)
        if file_id+'_eq_acts_dict' in all_files :
            eq_acts_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_eq_acts_dict'), "rb" ) )
            self.eq_acts_dict = eq_acts_dict
        else:
            q_string = "select EQUILIBRIUM_ACTIONS.TRACK_ID, EQUILIBRIUM_ACTIONS.TIME,EQ_ACTIONS from EQUILIBRIUM_ACTIONS where EQUILIBRIUM_ACTIONS.EQ_CONFIG_PARMS='NASH|MAXMIN|BOUNDARY' order by EQUILIBRIUM_ACTIONS.TRACK_ID, EQUILIBRIUM_ACTIONS.TIME"
            c.execute(q_string)
            res = c.fetchall()                
            self.eq_acts_dict = dict()
            for row in res:
                if (round(row[1]),row[0]) not in self.eq_acts_dict:
                    self.eq_acts_dict[(round(row[1]),row[0])] = list(dict.fromkeys([utils.get_l1_action_string(int(x[9:11])) for x in ast.literal_eval(row[2])]))
            pickle_dump(file_id+'_eq_acts_dict',self.eq_acts_dict)
        if file_id+'_traj_info_dict' in all_files and file_id+'_emp_traj_dict' in all_files :
            traj_info_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_traj_info_dict'), "rb" ) )
            self.traj_info_dict = traj_info_dict
            emp_traj_dict = pickle.load( open( os.path.join(OBJ_CACHE_DIR,file_id+'_emp_traj_dict'), "rb" ) )
            self.emp_traj_dict = emp_traj_dict
        else:   
            q_string = "select * from GENERATED_TRAJECTORY_INFO order by time"
            c_traj.execute(q_string)
            res = c_traj.fetchall()
            self.traj_info_dict = OrderedDict()
            ct,N = 0,len(res)
            for row in res:
                ct += 1
                #if ct == 100:
                #    break
                log.info("loading "+str(ct)+'/'+str(N))
                t = int(round(row[6]))
                if ((t,row[2]) in self.eq_acts_dict and row[3] == 0) or ((t,row[3]) in self.eq_acts_dict and row[3] != 0): 
                    eq_act = self.eq_acts_dict[(t,row[2])] if row[3] == 0 else self.eq_acts_dict[(t,row[3])]
                else:
                    if 'turn' in row[4]:
                        brk = 1
                    eq_act = self.acts_dict[(t,row[2])] if row[3] == 0 else self.acts_dict[(t,row[3])]
                if t not in self.traj_info_dict:
                    self.traj_info_dict[t] = dict()
                if len(eq_act) > 0 and row[4] in eq_act and row[5] == 'NORMAL':
                    self.traj_info_dict[t][(row[2],row[3])] = row[1]
                        
            ct,N = 0,len(self.traj_info_dict)    
            self.emp_traj_dict = dict()
            for k,v in self.traj_info_dict.items():
                ct += 1
                log.info("loading "+str(ct)+'/'+str(N))
                q_string = "select * from TRAJECTORIES_0"+file_id+" where time >="+str(k)+" and time <"+str(k+HORIZON)+" order by track_id,time"
                c.execute(q_string)
                res = c.fetchall()
                temp_trajs = dict()
                for row in res[::3]:
                    if row[0] not in temp_trajs:
                        temp_trajs[row[0]] = []
                    temp_trajs[row[0]].append((row[1], row[2], row[6]))
                self.emp_traj_dict[k] = dict()
                for t_k,t_v in self.traj_info_dict[k].items():
                    if t_k[1] == 0:
                        if t_k[0] in temp_trajs:
                            self.emp_traj_dict[k][t_k] = temp_trajs[t_k[0]]
                    else:
                        if t_k[1] in temp_trajs:
                            self.emp_traj_dict[k][t_k] = temp_trajs[t_k[1]]
                for o_a,o_t in temp_trajs.items():
                    if o_a not in [x[0] for x in self.traj_info_dict[k].keys()] and o_a not in [x[1] for x in self.traj_info_dict[k].keys()]:
                        self.emp_traj_dict[k][(o_a,-1)] = o_t
            ct,N = 0,len(self.traj_info_dict)    
            for k,v in self.traj_info_dict.items():
                ct += 1
                log.info("loading "+str(ct)+'/'+str(N))
                for ag,traj_info_id in v.items():
                    q_string = "select * from GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID="+str(traj_info_id)+" order by TRAJECTORY_ID,TIME"
                    c_traj.execute(q_string)
                    res = c_traj.fetchall()
                    if len(res) == 0:
                        self.traj_info_dict[k][ag] = []
                        continue
                    selected_traj_id = res[0][0]
                    self.traj_info_dict[k][ag] = [(row[3],row[4], row[2], row[6]) for row in res if row[0]==selected_traj_id and k <= row[2] < k+HORIZON]
                    #if ag == (8,0) and k==1:
                    #    f=1
            
            #self.traj_info_dict = self.emp_traj_dict
            #plot_traffic_regions()
            pickle_dump(file_id+'_traj_info_dict',self.traj_info_dict)
            pickle_dump(file_id+'_emp_traj_dict',self.emp_traj_dict)
        self.frame_list = []
        for k,v in self.emp_traj_dict.items():
            if len(v) == 0:
                continue
            max_len = max([len(t) for _,t in v.items()])
            this_frame_list = [(x,k)for x in np.arange(max_len)]
            self.frame_list.extend(this_frame_list)
        self.frame_list.sort(key=lambda tup: tup[1])
        conn.close()
        conn_traj.close()
        
    def __init__(self):
        self.setup_animation()
        self.fig, self.ax = plt.subplots()
        #fig.canvas.mpl_connect('button_press_event', onClick)
        self.anim_running = False
        self.title = self.ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax.transAxes, ha="center")
    
    def animate(self):
        plt.xlim(538780, 538890)
        plt.ylim(4813970, 4814055)
        img = plt.imread("D:\\behavior modeling\\background.jpg")
        plt.title("PNE-QE:MAXMIN:BOUNDARY")
        self.ax.imshow(img, extent=[538780, 538890, 4813970, 4814055])
        
        pause_ax = self.fig.add_axes((0.7, 0.025, 0.1, 0.04))
        pause_button = Button(pause_ax, 'play', hovercolor='0.975')
        pause_button.on_clicked(self._on_pause)
        self.pause_button = pause_button
        
        slider_ax = self.fig.add_axes((0.1, 0.025, 0.5, 0.04))
        self.time_slider = Slider(slider_ax, label='Time',
                                  valmin=0, valmax=len(list(set([x[1] for x in self.frame_list]))),
                                  valinit=0.0)
        
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")
        self.anim_running = False
        self.anim_started = False
        
        plt.show()
        
    
    def _on_pause(self,event):
        if self.anim_started:
            self.anim.event_source.stop()
        if not self.anim_started:
            self.anim_started = True
            freeze_frame = (0,0)
        else:
            freeze_frame = ast.literal_eval(self.title.get_text())
            freeze_frame = (freeze_frame[1],freeze_frame[0])
            
        print('pressed at',str(freeze_frame))
        if self.anim_running:
            self.pause_button.label.set_text('play')
            self.anim.event_source.stop()
            self.anim = animation.FuncAnimation(self.fig, self._update, frames=[self.frame_list[self.frame_list.index(freeze_frame)]], interval=50, blit=True, repeat=True)
            self.anim.event_source.start()
            self.anim_running = False
        else:
            self.anim = animation.FuncAnimation(self.fig, self._update, frames=self.frame_list[self.frame_list.index(freeze_frame):], interval=50, blit=True, repeat=False)
            self.pause_button.label.set_text('pause')
            self.anim.event_source.start()
            self.anim_running = True
    
    def _update(self,frame):
        return self.show_frame(frame)
            
    def show_frame(self, frame):
        i,time_ts = frame[0],frame[1]
        lines, gen_lines, emp_path_lines = dict(), dict(), dict()
        acts = dict()
        #start=mself.ax((i-5,0))
        curr_line = []
        curr_acts = []
        in_scene = []
        for ag,traj in self.emp_traj_dict[time_ts].items():
            principle_agent = ag[0] if (ag[1] == 0 or ag[1] == -1) else ag[1]
            if principle_agent not in in_scene:
                if ag not in lines:
                    if ag[1] == 0:
                        col = 'red'
                    elif ag[1] == -1:
                        col = 'black'
                    else: 
                        col = 'blue'
                    lines[ag], = self.ax.plot([], [], "o-", c=col)
                    emp_path_lines[ag], = self.ax.plot([], [], "-", c=col)
                    acts[ag] = self.ax.text(0.4,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center", fontsize="x-small")
                if ag not in gen_lines:
                    gen_lines[ag], = self.ax.plot([], [], "-", c='black')
                #start_i = mself.ax(0,i-5)
                #line1.set_data(sample_path1[start:i,0],sample_path1[start:i,1])
                #line2.set_data(sample_path2[start:i,0],sample_path2[start:i,1])
                if i+1 <= len(traj):
                    #lines[ag].set_data([x[0] for x in traj[start_i:i+1]],[x[1] for x in traj[start_i:i+1]])
                    lines[ag].set_data([x[0] for x in traj[i:]],[x[1] for x in traj[i:]])
                    emp_path_lines[ag].set_data([x[0] for x in traj[0:]],[x[1] for x in traj[0:]])
                    this_eq_act = []
                    this_act = ''
                    if ag[1] == 0:
                        this_act = str(self.acts_dict[(time_ts,ag[0])]) if (time_ts,ag[0]) in self.acts_dict else 'x'
                        if (time_ts,ag[0]) in self.eq_acts_dict:
                            for act in self.eq_acts_dict[(time_ts,ag[0])]:
                                if this_act != 'x':
                                    if act in self.acts_dict[(time_ts,ag[0])]:
                                        this_eq_act.append(act)
                                        break
                            if len(this_eq_act) == 0:
                                this_eq_act = self.eq_acts_dict[(time_ts,ag[0])]
                        this_act = this_act+'\n'+str(this_eq_act)
                    elif ag[1] != -1:
                        this_act = str(self.acts_dict[(time_ts,ag[1])]) if (time_ts,ag[1]) in self.acts_dict else 'x'
                        if (time_ts,ag[1]) in self.eq_acts_dict:
                            for act in self.eq_acts_dict[(time_ts,ag[1])]:
                                if this_act != 'x':
                                    if act in self.acts_dict[(time_ts,ag[1])]:
                                        this_eq_act.append(act)
                                        break
                            if len(this_eq_act) == 0:
                                this_eq_act = self.eq_acts_dict[(time_ts,ag[1])]
                        this_act = this_act+'\n'+str(this_eq_act) if len(this_eq_act) > 0 else this_act
                    acts[ag].set_text(this_act)
                    acts[ag].set_x(traj[i][0])
                    acts[ag].set_y(traj[i][1])
                    curr_line.append(lines[ag])
                    curr_line.append(emp_path_lines[ag])
                    curr_acts.append(acts[ag])
                in_scene.append(principle_agent)
        for ag,traj in self.traj_info_dict[time_ts].items():
            if ag not in gen_lines:
                gen_lines[ag], = self.ax.plot([], [], "-", c='black')
            start_i = max(0,i-2)
            #line1.set_data(sample_path1[start:i,0],sample_path1[start:i,1])
            #line2.set_data(sample_path2[start:i,0],sample_path2[start:i,1])
            if type(traj) is int:
                brk = 1
            if i+1 <= len(traj):
                end_idx = min(len(traj)-1,i+(HORIZON*10+1))
                #gen_lines[ag].set_data([x[0] for x in traj[start_i:i+1]],[x[1] for x in traj[start_i:i+1]])
                gen_lines[ag].set_data([x[0] for x in traj[0:end_idx]],[x[1] for x in traj[0:end_idx]])
                curr_line.append(gen_lines[ag])
            
        self.title.set_text(str((time_ts,i)))
        #self.time_slider.set_val(time_ts+(.1*i))
        return tuple(curr_line+[self.title]+curr_acts)
      
    
if __name__ == '__main__':     
    sca = SceneAnimation()
    sca.animate()