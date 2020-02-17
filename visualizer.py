'''
Created on Jan 8, 2020

@author: Atrisha
'''
import csv
import matplotlib.pyplot as plt
import numpy as np

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
