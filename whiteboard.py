'''
Created on Jan 24, 2020

@author: Atrisha
'''
import numpy as np
import math
import utils
import matplotlib.pyplot as plt
import sqlite3



def translate(pt1,pt2,yaw,pos):
    yaw = 1.5
    ''' r*cos(yaw-slope + 90) = d '''
    d = utils.distance_numpy([pt1[0],pt1[1]], [pt2[0],pt2[1]], [pos[0],pos[1]])
    slope = math.atan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
    r = d / (math.cos(yaw - slope + (.5*math.pi)))
    print(r)
    print(d)
    plt.plot([pt1[0],pt2[0]],[pt1[1],pt1[1]],'ro')
    plt.plot([538842.32],[4814000.61],'bx')
    plt.plot()
    plt.show()
#translate((538837.55,4814002.03), (538842.14,4814005.39))



def show_dist_to_lane_boundary():
    traj = [(538842.31999999995, 4814000.6100000003), (538842.30861410464, 4814000.6249298444), (538842.29628636944, 4814000.6424344024), (538842.28268724121, 4814000.6641580509), (538842.26748709544, 4814000.6915962473), (538842.25035639771, 4814000.7260995833), (538842.23096586776, 4814000.7688778136), (538842.20898664091, 4814000.8210038962), (538842.18409043085, 4814000.8834180245), (538842.15594969213, 4814000.9569316721), (538842.1242377829, 4814001.0422316222), (538842.08862912701, 4814001.1398840053), (538842.0487993767, 4814001.2503383486), (538842.00442557526, 4814001.3739315914), (538841.95518631896, 4814001.5108921425), (538841.90076192026, 4814001.6613439051), (538841.84083456965, 4814001.825310315), (538841.77508849942, 4814002.0027183862), (538841.70321014349, 4814002.1934027346), (538841.62488830346, 4814002.3971096277), (538841.53981430794, 4814002.6135010133), (538841.44768217718, 4814002.8421585597), (538841.34818878444, 4814003.0825876901), (538841.24103401857, 4814003.3342216248), (538841.12592094752, 4814003.5964254132), (538841.00255597918, 4814003.8684999738), (538840.87064902543, 4814004.1496861298), (538840.72991366358, 4814004.4391686469), (538840.58006729907, 4814004.7360802675), (538840.42083132896, 4814005.0395057537), (538840.25193130306, 4814005.3484859159), (538840.07309708686, 4814005.6620216593), (538839.88406302442, 4814005.9790780144), (538839.68456810038, 4814006.2985881744), (538839.47435610299, 4814006.6194575373), (538839.253175786, 4814006.9405677319), (538839.02078103158, 4814007.2607806725), (538838.77693101286, 4814007.5789425764), (538838.52139035589, 4814007.8938880134), (538838.25392930245, 4814008.20444394), (538837.97432387306, 4814008.5094337389), (538837.68235602882, 4814008.8076812439), (538837.37781383377, 4814009.0980147952), (538837.060491618, 4814009.3792712623), (538836.7301901402, 4814009.6503000893), (538836.38671674882, 4814009.9099673228), (538836.02988554654, 4814010.1571596628), (538835.65951755166, 4814010.3907884806), (538835.27544086007, 4814010.6097938791), (538834.87749080907, 4814010.813148709), (538834.46551013901, 4814010.9998626169), (538834.03934915562, 4814011.1689860784), (538833.59886589355, 4814011.3196144402), (538833.14392627717, 4814011.4508919492), (538832.67440428515, 4814011.5620157914), (538832.19018211146, 4814011.6522401413), (538831.69115032826, 4814011.7208801769), (538831.17720804794, 4814011.7673161356), (538830.64826308715, 4814011.7909973441), (538830.10423212696, 4814011.7914462499), (538829.545040878, 4814011.7682624692), (538828.97062424058, 4814011.7211268181), (538828.38092646853, 4814011.6498053493), (538827.77590133168, 4814011.5541533865), (538827.15551227739, 4814011.4341195701), (538826.51973259437, 4814011.2897498887), (538825.86854557379, 4814011.1211917102), (538825.20194467332, 4814010.928697831), (538824.51993367844, 4814010.7126305047), (538823.82252686517, 4814010.4734654836), (538823.10974916245, 4814010.211796049), (538822.38163631549, 4814009.9283370571), (538821.63823504758, 4814009.6239289688), (538820.87960322201, 4814009.2995418897), (538820.10581000592, 4814008.9562796094), (538819.31693603192, 4814008.595383632), (538818.51307356055, 4814008.2182372222), (538817.69432664302, 4814007.8263694346), (538816.86081128404, 4814007.4214591486), (538816.01265560358, 4814007.0053391196), (538815.15000000002, 4814006.5799999991)]
    r = [538841,4814010]
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    q_string = "SELECT X_POSITIONS,Y_POSITIONS FROM TRAFFIC_REGIONS_DEF WHERE NAME = 'l_s_n' and REGION_PROPERTY='left_boundary' and SHAPE='line'"
    c.execute(q_string)
    res = c.fetchall()
    x,y = [],[]
    for row in res:
        x = ast.literal_eval(row[0])
        y = ast.literal_eval(row[1])
    plt.plot([x[0] for x in traj],[x[1] for x in traj],'b-')
    plt.plot(x,y,'r-')
    plt.show()
    for t in traj:
        print(utils.distance_numpy([x[0],y[0]], [x[1],y[1]], [t[0],t[1]]))
    print(utils.distance_numpy([x[0],y[0]], [x[1],y[1]], [r[0],r[1]]))

import math 

def show_poly():
    a1,a2,a3,a4,a5 = 538842.39, -0.071381561358092, 0.0147584958045, -0.00264942905716, 7.43072092442e-05
    X = np.arange(0,15,.1)
    Y = [0 + (a1*t) + (a2*t**2) + (a3*t**3) + (a4*t**4) + (a5*t**5) for t in X]
    plt.plot(X,Y)
    plt.show()



def q_p():
    a_0 = 0
    a_1 = 10
    a_2 = .5
    a_3 = 0
    a_norm = 1.4
    v_f = a_1 + a_norm * 5
    print(v_f)
    ct = 0
    coeff_matrix = [] 
    for a_4 in np.arange(0,.002,.001):
        for a_5 in np.arange(0,0.002,0.001):
            ct += 1
            T = np.arange(0,5,.1)
            Y = [a_1 + (2*a_2*t) + (3*a_3*t**2) + (4*a_4*t**3) + (5*a_5*t**4) for t in T]
            max_jerk = max([(6*a_3) + (24*a_4*t) + (60*a_5*t**2) for t in T])
            max_acc = max([(2*a_2) + (6*a_3*t) + (12*a_4*t**2) + (20*a_5*t**3) for t in T])
            v_list = [a_1 + (2*a_2*t) + (3*a_3*t**2) + (4*a_4*t**3) + (5*a_5*t**4) for t in T]
            linear_a = (v_list[-1] - 14.7)/5
            if math.floor(v_list[-1]) == math.floor(v_f):
                #if max_acc >= 2*a_2:
                coeff_matrix.append((v_f,max_acc,max_jerk,(a_4,a_5)))
                print('nums',len(coeff_matrix),max_acc)
                
            #print(ct,a_4,a_5,a_1,max_acc,max_jerk,v_list[-1],linear_a)
            if ct == 1000:
                break
        if ct == 1000:
            break
            #plt.plot(T,Y)
            #plt.show()
    print(coeff_matrix)       

def pad_test():
    z = str(198)
    print(z.zfill(3))

#def panda_test():
import re
def regex_test():
    s = '769011......._10..$'
    f1 = '7690110000202_1000'
    f2 = '7690110000301_10010'
    for _f in [f1,f2]:
        if re.match(s, _f):
            print(_f,True)
        else:
            print(_f,False)


def ruptures_test():
    q_string = "select SPEED from trajectories_0769 where track_id=11 order by time"
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber.db')
    c = conn.cursor()
    c.execute(q_string)
    res = c.fetchall()
    v_signal = []
    for r in res:
        v_signal.append(float(r[0]))
    v_signal = np.asarray(v_signal)
    model = "rbf"  # "l1", "rbf", "linear", "normal", "ar"
    #algo = rpt.Binseg(model=model).fit(v_signal)
    #algo = rpt.Window(width=50, model=model).fit(v_signal)
    #algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(v_signal)
    #result = algo.predict(n_bkps=3)
    #algo = rpt.Pelt(model="rbf").fit(v_signal)
    #result = algo.predict(pen=10)
    tru_brkpts = []
    thresh_kph = [.2,40]
    for t in thresh_kph:
        for _i,v in enumerate(v_signal):
            if v > t:
                tru_brkpts.append(_i)
                break
    tru_brkpts.append(len(v_signal)-1)
    #rpt.display(v_signal, tru_brkpts, result)
    plt.axvspan(8, 14, alpha=0.5, color='red')
    plt.show()
'''
plt.axis('equal')
plt.plot( [538798.3, 538796.58],  [4813995.35, 4813998.65], '-')
plt.plot([538814.15,538816.08], [4814007.58,4814004.26], 'b-')
plt.plot([538852.12],[4813985.8],'o')
plt.plot([538814.87],[4814004.97],'x')
plt.show()


from collections import OrderedDict    
hpx =  [538794.04715, 538794.04715, 538794.81398, 538796.05269]
hpy =  [4813996.09035, 4813996.09035, 4813996.48035, 4813997.11033]
#_l = list(sorted(set(zip(hpx,hpy)),key=lambda tup: tup[0]))
_d = OrderedDict(sorted(list(zip(hpx,hpy)),key=lambda tup: tup[0]))
hpx,hpy = list(_d.keys()),list(_d.values())
print(hpx,hpy)
'''
def db_test():
    import sqlite3
    conn = sqlite3.connect('D:\\intersections_dataset\\dataset\\uni_weber_generated_trajectories.db')
    c = conn.cursor()
    c.execute("INSERT INTO AA VALUES(NULL,?)",(4,))
    conn.commit()
    id = int(c.lastrowid)
    print(id)
    
import itertools
'''    
print(np.divide([.4,.3,.2,.1],[.1,.2,.3,.4]))
plt.plot([1,2,3,4],[.04,.05,.7,.8],c='black')
plt.plot([1,2,3,4],[.1,.2,.3,.4],'b')  
plt.plot([1,2,3,4],np.divide([.04,.05,.7,.8],[.1,.2,.3,.4]),'r')
plt.show()
'''
'''
tm_keys = list(itertools.product([('wait','cancel'),('wait','cont.'),('merge','cancel'),('merge','cont.')],\
                             [('slow','cont. speed'),('speed','cont. speed'),('slow','slow'),('speed','slow')]))

toy_merge_dict = {k:[None,None] for k in tm_keys}
print(toy_merge_dict) 
'''
'''
bel = np.arange(0,1.1,1)
payoffs = [-5,10]

plt.plot([0,1],[0,1]) 
plt.plot([0,1],[0,0]) 
plt.show()
'''
def sym_sol():
    import sympy
    from sympy.solvers import solve
    from sympy import Symbol
    a_0 = Symbol('a_0')
    a_1 = Symbol('a_1')
    a_2 = Symbol('a_2')
    a_3 = Symbol('a_3')
    p_0 = Symbol('p_0')
    p_1 = Symbol('p_1')
    p_2 = Symbol('p_2')
    p_3 = Symbol('p_3')
    s_f = Symbol('p_4')
    e = [a_0-p_0, (a_3*s_f**3)+(a_2*s_f**2)+(a_1*s_f)+a_0-p_3, ((a_3*s_f**3)/27)+((a_2*s_f**2)/9)+((a_1*s_f)/3)+a_0-p_1, ((8*a_3*s_f**3)/27)+((4*a_2*s_f**2)/9)+((2*a_1*s_f)/3)+a_0-p_2]  
    s = solve(e,[a_0,a_1,a_2,a_3])
    print(s)
sym_sol()