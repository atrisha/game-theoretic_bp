'''
Created on Jan 8, 2020

@author: Atrisha
'''
import csv
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import constants
from all_utils import utils
from all_utils.utils import kph_to_mps
import matplotlib.animation as animation
log = constants.common_logger
from collections import OrderedDict

def show_qre_plot_toy_example():
    x_lambdas,x_ax = [],[]
    p1_n1,p1_n2,p2_n1,p2_n2 = [],[],[],[]
    csv_file_loc = "D:\\gambit\\my games\\toy_lane_change_qre.csv"
    line_num = 0
    with open(csv_file_loc) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_lambdas.append(round(float(row[0]),5))
            x_ax.append(line_num)
            p1_n1.append(round(float(row[1]),5))
            p1_n2.append(round(float(row[3]),5))
            p2_n1.append(round(float(row[5]),5))
            p2_n2.append(round(float(row[7]),5))
            line_num += 1
    plt.title('Quantal Response Equilibrium')
    plt.plot(x_ax,p1_n1,'r')
    plt.text(x_ax[100],p1_n1[100],'merge\nwait')
    #plt.text(x_ax[1],p1_n1[1]+.1,'merge')
    
    plt.plot(x_ax,p1_n2,'r')
    plt.text(x_ax[100],p1_n2[100]-.01,'continue merge\ncancel merge')
    #plt.text(x_ax[10],p1_n2[1]+.01,'canc m')
    
    plt.plot(x_ax,p2_n1,'b')
    plt.text(x_ax[50],p2_n1[50],'speed up\nslow down ')
    #plt.text(x_ax[30],p2_n1[1]+.01,'slow down')
    
    plt.plot(x_ax,p2_n2,'b')
    plt.text(x_ax[5],p2_n2[15],'slow down\ncont. speeding')
    #plt.text(x_ax[40],p1_n1[1]+.01,'cont speed')
    
    
    plt.xticks(np.arange(0,200,50),[round(x_lambdas[i],1) for i in np.arange(0,200,50)])
    plt.ylim(0,1)
    plt.show()
    

def show_animation(data_arr_x,data_arr_y):
    time_len = max([len(x) for x in data_arr_x])
    for i in np.arange(time_len):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.grid(True)
        plt.axis("equal")
        for _aidx, arr in enumerate(data_arr_x):
            if i < arr.shape[0]:
                plt.plot([arr[:i]],[data_arr_y[_aidx][:i]],'.') 
                plt.title(str(i)+'\\'+str(time_len))
        
        '''
        plt.title("Time[s]:" + str(time[i])[0:4] +
                  " v[m/s]:" + str(rv[i])[0:4] +
                  " a[m/ss]:" + str(ra[i])[0:4] +
                  " jerk[m/sss]:" + str(rj[i])[0:4],
                  )
        '''
        plt.pause(.1)
    plt.show()

def plot_payoff_grid(payoff_arr):
    import scipy.special
    import math
    fig, (ax1,ax2) = plt.subplots(2, 1)
    c = ax1.pcolor(payoff_arr)
    fig.colorbar(c, ax=ax1)
    X = np.arange(0,20)
    Y = [scipy.special.erf((x - constants.DIST_COST_MEAN) / (constants.DIST_COST_SD * math.sqrt(2))) for x in X]
    ax2.plot(X,Y)
    fig.tight_layout()
    plt.show()
    
def plot_velocity(vel_list,agent_id,horizon,ag_idx,ax4):
    if agent_id is not None:
        q_string = "select SPEED,TIME from trajectories_0"+constants.CURRENT_FILE_ID+" where track_id="+str(agent_id)+" and (TIME BETWEEN "+str(horizon[0])+" AND "+str(horizon[1])+") order by time"
        conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
        c = conn.cursor()
        c.execute(q_string)
        res = c.fetchall()
        v_signal = []
        X = []
        for r in res:
            v_signal.append(kph_to_mps(float(r[0])))
            X.append(float(r[1]))
        #X = [X[i] for i in np.arange(horizon[0],horizon[1],int(constants.DATASET_FPS*constants.LP_FREQ))]
        #v_signal = [v_signal[i] for i in np.arange(horizon[0],horizon[1],int(constants.DATASET_FPS*constants.LP_FREQ))]
        ax4.plot(X,v_signal,color=constants.colors[ag_idx])
    if vel_list is not None:
        ax4.plot([x[0] for x in vel_list],[x[1] for x in vel_list],color=constants.colors[ag_idx],ls='--')

def plot_all_paths(veh_state):
    import ast
    q_string = "select * from traffic_regions_def where name in "+str(tuple(veh_state.segment_seq))
    #print(q_string)
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    c.execute(q_string)
    res = c.fetchall()
    pos_plotted = False
    for _l1,l2 in veh_state.action_plans[veh_state.current_time].items():
        if _l1 == 'wait-for-oncoming':
            for _l2,l3_action_list in l2.items():
                print(_l1,_l2,len(l3_action_list))
                for l in l3_action_list:
                    for row in res:
                        x_s = ast.literal_eval(row[4])
                        y_s = ast.literal_eval(row[5])
                        plt.plot(x_s,y_s)
                    plt.plot(constants.VIEWPORT[0],constants.VIEWPORT[1])
                    if l[1] == 'CP':
                        brk = 1
                    plt.plot(l[0][1],l[0][2],'-')
                    plt.plot([l[0][1][-1]],[l[0][2][-1]],'x',c='lime')
            if not pos_plotted:
                plt.plot([veh_state.x],[veh_state.y],'x',c='g')
                pos_plotted = True
            plt.axis('equal')
            plt.show()
    conn.close()
    
def plot_paths_for_traj_info_id(traj_info_id=79):
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "SELECT TRAJ_ID FROM GENERATED_TRAJECTORY_INFO WHERE TIME = 86.419667"
    c.execute(q_string)
    res = c.fetchall()
    all_traj_info_ids = [row[0] for row in res]
    q_string = "select * from GENERATED_TRAJECTORY where GENERATED_TRAJECTORY.TRAJECTORY_INFO_ID in "+str(tuple(all_traj_info_ids))
    c.execute(q_string)
    print(q_string)
    traj_dict = dict()
    res = c.fetchall()
    ct = 0
    for row in res:
        print(ct)
        ct += 1
        if row[0] not in traj_dict:
            traj_dict[row[0]] = [ [ row[3] ], [ row[4] ] ]
        else:
            traj_dict[row[0]][0].append(row[3])
            traj_dict[row[0]][1].append(row[4])
    for k,v in traj_dict.items():
        plt.plot(v[0],v[1])
    
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.axis('equaL')
    plt.show()
    
    
def plot_baselines():
    import ast
    
    all_l1_actions = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    conn1 = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c1 = conn1.cursor()
    ag_id = 44
    q_string = "select distinct l1_action from GENERATED_TRAJECTORY_INFO where agent_id="+str(ag_id)
    c.execute(q_string)
    res = c.fetchall()
    all_l1_actions = [row[0] for row in res]        
    emp_path = utils.get_path(ag_id)
    plt.plot([x[0] for x in emp_path],[x[1] for x in emp_path],'blue')
    for l1_act in all_l1_actions:
        q_string = "SELECT DISTINCT TRAJ_ID FROM GENERATED_TRAJECTORY_INFO where l1_action='"+l1_act+"' and agent_id="+str(ag_id)
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
                traj_dict[row[0]] = [ [ row[3] ], [ row[4] ] ]
            else:
                traj_dict[row[0]][0].append(row[3])
                traj_dict[row[0]][1].append(row[4])
        for k,v in traj_dict.items():
            print(k)
            plt.plot(v[0],v[1],'--',color='black')
            
        ax = plt.gca()
        #ax.set_facecolor('black')
        plt.title(l1_act)
        plt.axis('equaL')
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property <> 'left_boundary'"
        c1.execute(q_string)
        q_res = c1.fetchall()
        plt.axis("equal")
        for row in q_res:
            plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        plt.show()
    conn.close()
    
def plot_boundaries():
    import ast
    
    all_l1_actions = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    conn1 = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c1 = conn1.cursor()
    ag_id = 43
    q_string = "select distinct l1_action from GENERATED_TRAJECTORY_INFO where agent_id="+str(ag_id)
    c.execute(q_string)
    res = c.fetchall()
    all_l1_actions = [row[0] for row in res]        
    emp_path = utils.get_path(ag_id)
    plt.plot([x[0] for x in emp_path],[x[1] for x in emp_path],'blue')
    for l1_act in all_l1_actions:
        q_string = "SELECT DISTINCT TRAJ_ID FROM GENERATED_TRAJECTORY_INFO where l1_action='"+l1_act+"' and agent_id="+str(ag_id)
        c.execute(q_string)
        res = tuple(row[0] for row in c.fetchall())
        
        q_string = "select * from GENERATED_BOUNDARY_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(res)) + ' UNION ' + \
                     "select * from GENERATED_BASELINE_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(res))+" order by trajectory_id,time"
        print(q_string)
        c.execute(q_string)
        traj_dict = dict()
        res = c.fetchall()
        ct = 0
        for row in res:
            ct += 1
            if row[0] not in traj_dict:
                traj_dict[row[0]] = [ [ row[3] ], [ row[4] ] ]
            else:
                traj_dict[row[0]][0].append(row[3])
                traj_dict[row[0]][1].append(row[4])
        for k,v in traj_dict.items():
            print(k)
            plt.plot(v[0],v[1],'--',color='black')
            
        ax = plt.gca()
        #ax.set_facecolor('black')
        plt.title(l1_act)
        plt.axis('equaL')
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property <> 'left_boundary'"
        c1.execute(q_string)
        q_res = c1.fetchall()
        plt.axis("equal")
        for row in q_res:
            plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        plt.show()
    conn.close()    
    

def plot_baseline_velocities():
    import ast
    
    all_l1_actions = []
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    conn1 = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c1 = conn1.cursor()
    ag_id = 44
    q_string = "select distinct l1_action from GENERATED_TRAJECTORY_INFO"
    c.execute(q_string)
    res = c.fetchall()
    all_l1_actions = [row[0] for row in res]        
    emp_path = utils.get_path(ag_id)
    plt.plot([x[0] for x in emp_path],[x[1] for x in emp_path],'blue')
    for l1_act in all_l1_actions:
        q_string = "SELECT DISTINCT TRAJ_ID FROM GENERATED_TRAJECTORY_INFO where l1_action='"+l1_act+"'"
        c.execute(q_string)
        res = tuple(row[0] for row in c.fetchall())
        
        q_string = "select * from GENERATED_BOUNDARY_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(res)) + ' UNION ' + \
                     "select * from GENERATED_BASELINE_TRAJECTORY where TRAJECTORY_INFO_ID in "+str(tuple(res))+" order by trajectory_id,time"
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
            plt.plot(v[0],v[1],'--',color='black')
            
        ax = plt.gca()
        #ax.set_facecolor('black')
        plt.title(l1_act)
        plt.axis('equaL')
        q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property <> 'left_boundary' and region_property <> 'center_line'"
        c1.execute(q_string)
        q_res = c1.fetchall()
        plt.axis("equal")
        for row in q_res:
            plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        plt.show()
        for k,v in traj_dict.items():
            print(k)
            plt.plot(np.arange(len(v[2])),v[2],color='black',linewidth=0.25)
        plt.title(l1_act+'-velocity')
        plt.show()
    conn.close()


def plot_all_trajectories(traj,ax1,ax2,ax3):
    
    X,Y,V,A = [],[],[],[]
    traj = traj[:,0]
    for traj_slice in traj:
        X = traj_slice[1,:].tolist()
        Y = traj_slice[2,:].tolist()
        V = traj_slice[4,:].tolist()
        A = traj_slice[5,:].tolist()
        ax1.plot(X,Y)
        ax2.plot(np.arange(len(V)),V)
        ax3.plot(np.arange(len(A)),A)
        
def plot_traffic_regions():
    import ast
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property <> 'left_boundary'"
    c.execute(q_string)
    q_res = c.fetchall()
    plt.axis("equal")
    for row in q_res:
        #plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        X,Y = ast.literal_eval(row[4]),ast.literal_eval(row[5])
        if len(X) == 2:
            plt.arrow(X[0], Y[0], X[1]-X[0], Y[1]-Y[0], head_width=0.5, head_length=0.5)
            f=1
        else:
            for i,j in zip(np.arange(len(X)-1), np.arange(1,len(Y))):
                plt.arrow(X[i], Y[i], X[j]-X[i], Y[j]-Y[i], head_width=0.5, head_length=0.5)
                f=1
        #plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
    '''
    q_string = "SELECT X_POSITION,Y_POSITION FROM CONFLICT_POINTS"
    c.execute(q_string)
    q_res = c.fetchall()
    plt.plot([x[0] for x in q_res], [x[1] for x in q_res], 'x')
    plt.show()   
    '''
    #plt.ylim(4813961, 4814065)
    #plt.xlim(538776, 538897)
    #plt.show()
        
def verify_traffic_regions():
    import ast
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property <> 'left_boundary' and region_property <> 'gate_line'"
    c.execute(q_string)
    q_res = c.fetchall()
    plt.axis("equal")
    N = len(q_res)
    for idx,row in enumerate(q_res):
        plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]),'--',color='black')
        for idx2,row2 in enumerate(q_res):
            if idx!=idx2 and (row[4]!=row2[4] and row[5] != row2[5]): 
                plt.plot(ast.literal_eval(row2[4]),ast.literal_eval(row2[5]))
        plt.title(row[0]+' , '+row[3]+' '+'['+str(idx)+'/'+str(N)+']')
        plt.show()
        
    '''
    q_string = "SELECT X_POSITION,Y_POSITION FROM CONFLICT_POINTS"
    c.execute(q_string)
    q_res = c.fetchall()
    plt.plot([x[0] for x in q_res], [x[1] for x in q_res], 'x')
    plt.show()   
    '''
        
def plot_exit_yaws():
    import ast
    from all_utils import utils
    import math
    import statistics
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    all_exits = dict()
    c = conn.cursor()
    q_string = "SELECT * FROM TRAFFIC_REGIONS_DEF where shape <> 'point' and region_property == 'exit_boundary'"
    c.execute(q_string)
    q_res = c.fetchall()
    plt.axis("equal")
    for row in q_res:
        plt.plot(ast.literal_eval(row[4]),ast.literal_eval(row[5]))
        if row[0] not in all_exits:
            all_exits[row[0]] = [ast.literal_eval(row[4]),ast.literal_eval(row[5])] 
    yaws = utils.get_mean_yaws_for_segments(list(all_exits.keys()))
    for k,v in yaws.items():
        if k in all_exits:
            plt.arrow(statistics.mean(all_exits[k][0]), statistics.mean(all_exits[k][1]), 5 * math.cos(v), 5 * math.sin(v), fc='r', ec='k', head_width=.25, head_length=.25)
    plt.show()
    
      
def show_baseline_animation():
    
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    q_string = "select * from GENERATED_TRAJECTORY_INFO order by time"
    c.execute(q_string)
    res = c.fetchall()
    traj_info_dict = OrderedDict()
    ct,N = 0,len(res)
    for row in res:
        ct += 1
        log.info("loading "+str(ct)+'/'+str(N))
        if row[6] not in traj_info_dict:
            traj_info_dict[row[6]] = dict()
            traj_info_dict[row[6]][(row[2],row[3])] = row[1]
        else:
            if (row[2],row[3]) not in traj_info_dict[row[6]]:
                traj_info_dict[row[6]][(row[2],row[3])] = row[1]
    ct,N = 0,len(traj_info_dict)    
    for k,v in traj_info_dict.items():
        ct += 1
        log.info("loading "+str(ct)+'/'+str(N))
        for ag,traj_info_id in v.items():
            q_string = "select * from GENERATED_GAUSSIAN_TRAJECTORY where GENERATED_GAUSSIAN_TRAJECTORY.TRAJECTORY_INFO_ID="+str(traj_info_id)+" order by TRAJECTORY_ID,TIME"
            print(traj_info_id)
            c.execute(q_string)
            res = c.fetchall()
            selected_traj_id = res[0][0]
            traj_info_dict[k][ag] = [(row[3],row[4]) for row in res if row[0]==selected_traj_id]
    
    
    #plt.xlim(538777, 538900)
    #plt.ylim(483960, 4814070)
    fig, ax = plt.subplots()
    plot_traffic_regions()
    plt.xlim(538777, 538900)
    plt.ylim(4813960, 4814065)
    title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    
    #line1, = ax.plot([], [], "ro-")
    #line2, = ax.plot([], [], "ro-")
    
    def connect(p):
        i,time_ts = p[0],p[1]
        lines = dict()
        #start=max((i-5,0))
        curr_line = []
        for ag,traj in traj_info_dict[time_ts].items():
            if ag not in lines:
                col = 'red' if ag[1] == 0 else 'blue'
                lines[ag], = ax.plot([], [], "o-", c=col)
            start_i = max(0,i-5)
            #line1.set_data(sample_path1[start:i,0],sample_path1[start:i,1])
            #line2.set_data(sample_path2[start:i,0],sample_path2[start:i,1])
            if i+1 <= len(traj):
                lines[ag].set_data([x[0] for x in traj[start_i:i+1]],[x[1] for x in traj[start_i:i+1]])
                curr_line.append(lines[ag])
        title.set_text(str((time_ts,i)))
        return tuple(curr_line+[title])
    frame_list = []
    for k,v in traj_info_dict.items():
        max_len = max([len(t) for _,t in v.items()])
        this_frame_list = [(x,k)for x in np.arange(max_len)]
        frame_list.extend(this_frame_list)
    frame_list.sort(key=lambda tup: tup[1])
    ani = animation.FuncAnimation(fig, connect, frames=frame_list, interval=100, blit=True, repeat=False)
    plt.show()
    

      
    
if __name__ == '__main__':     
    show_baseline_animation()