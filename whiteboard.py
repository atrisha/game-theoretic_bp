'''
Created on Jan 24, 2020

@author: Atrisha
'''
import numpy as np
import math
import all_utils
import matplotlib.pyplot as plt
import sqlite3



def translate(pt1,pt2,yaw,pos):
    yaw = 1.5
    ''' r*cos(yaw-slope + 90) = d '''
    d = all_utils.distance_numpy([pt1[0],pt1[1]], [pt2[0],pt2[1]], [pos[0],pos[1]])
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
        print(all_utils.distance_numpy([x[0],y[0]], [x[1],y[1]], [t[0],t[1]]))
    print(all_utils.distance_numpy([x[0],y[0]], [x[1],y[1]], [r[0],r[1]]))

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


def debug_traj():
    import all_utils
    path,v_s,a_s,max_acc,max_jerk,v_g,dt,acc = [(538842.39, 4814000.65), (538842.29, 4814000.810362602), (538842.1900000001, 4814000.969194056), (538842.0900000001, 4814001.126501106), (538841.9900000001, 4814001.282290499), (538841.8900000001, 4814001.436568978), (538841.7900000002, 4814001.589343293), (538841.6900000002, 4814001.740620183), (538841.5900000002, 4814001.890406398), (538841.4900000002, 4814002.03870868), (538841.3900000002, 4814002.185533778), (538841.2900000003, 4814002.330888433), (538841.1900000003, 4814002.4747793935), (538841.0900000003, 4814002.617213403), (538840.9900000003, 4814002.758197208), (538840.8900000004, 4814002.897737553), (538840.7900000004, 4814003.035841184), (538840.6900000004, 4814003.172514845), (538840.5900000004, 4814003.307765282), (538840.4900000005, 4814003.441599239), (538840.3900000005, 4814003.574023464), (538840.2900000005, 4814003.7050447), (538840.1900000005, 4814003.834669692), (538840.0900000005, 4814003.962905188), (538839.9900000006, 4814004.089757931), (538839.8900000006, 4814004.215234666), (538839.7900000006, 4814004.33934214), (538839.6900000006, 4814004.462087096), (538839.5900000007, 4814004.583476282), (538839.4900000007, 4814004.7035164405), (538839.3900000007, 4814004.8222143175), (538839.2900000007, 4814004.93957666), (538839.1900000008, 4814005.055610212), (538839.0900000008, 4814005.170321718), (538838.9900000008, 4814005.283717925), (538838.8900000008, 4814005.395805577), (538838.7900000009, 4814005.506591419), (538838.6900000009, 4814005.616082196), (538838.5900000009, 4814005.724284656), (538838.4900000009, 4814005.831205541), (538838.390000001, 4814005.936851598), (538838.290000001, 4814006.041229572), (538838.190000001, 4814006.144346208), (538838.090000001, 4814006.246208252), (538837.990000001, 4814006.346822448), (538837.8900000011, 4814006.446195542), (538837.7900000011, 4814006.544334278), (538837.6900000011, 4814006.641245404), (538837.5900000011, 4814006.736935664), (538837.4900000012, 4814006.831411801), (538837.3900000012, 4814006.924680564), (538837.2900000012, 4814007.016748696), (538837.1900000012, 4814007.107622942), (538837.0900000012, 4814007.197310048), (538836.9900000013, 4814007.28581676), (538836.8900000013, 4814007.373149822), (538836.7900000013, 4814007.459315979), (538836.6900000013, 4814007.5443219775), (538836.5900000014, 4814007.628174563), (538836.4900000014, 4814007.71088048), (538836.3900000014, 4814007.792446473), (538836.2900000014, 4814007.872879287), (538836.1900000015, 4814007.95218567), (538836.0900000015, 4814008.030372366), (538835.9900000015, 4814008.107446118), (538835.8900000015, 4814008.183413673), (538835.7900000016, 4814008.258281779), (538835.6900000016, 4814008.332057176), (538835.5900000016, 4814008.4047466125), (538835.4900000016, 4814008.476356832), (538835.3900000016, 4814008.546894582), (538835.2900000017, 4814008.616366607), (538835.1900000017, 4814008.684779652), (538835.0900000017, 4814008.752140462), (538834.9900000017, 4814008.818455781), (538834.8900000018, 4814008.883732357), (538834.7900000018, 4814008.947976934), (538834.6900000018, 4814009.011196257), (538834.5900000018, 4814009.07339707), (538834.4900000019, 4814009.13458612), (538834.3900000019, 4814009.194770153), (538834.2900000019, 4814009.253955913), (538834.1900000019, 4814009.312150146), (538834.090000002, 4814009.369359597), (538833.990000002, 4814009.425591008), (538833.890000002, 4814009.48085113), (538833.790000002, 4814009.535146705), (538833.690000002, 4814009.588484477), (538833.5900000021, 4814009.640871194), (538833.4900000021, 4814009.692313602), (538833.3900000021, 4814009.742818443), (538833.2900000021, 4814009.792392463), (538833.1900000022, 4814009.84104241), (538833.0900000022, 4814009.8887750255), (538832.9900000022, 4814009.935597057), (538832.8900000022, 4814009.98151525), (538832.7900000022, 4814010.026536349), (538832.6900000023, 4814010.070667099), (538832.5900000023, 4814010.113914246), (538832.4900000023, 4814010.156284534), (538832.3900000023, 4814010.197784709), (538832.2900000024, 4814010.238421517), (538832.1900000024, 4814010.278201702), (538832.0900000024, 4814010.317132011), (538831.9900000024, 4814010.355219187), (538831.8900000025, 4814010.392469978), (538831.7900000025, 4814010.428891127), (538831.6900000025, 4814010.464489378), (538831.5900000025, 4814010.499271479), (538831.4900000026, 4814010.533244176), (538831.3900000026, 4814010.566414212), (538831.2900000026, 4814010.598788333), (538831.1900000026, 4814010.630373283), (538831.0900000026, 4814010.66117581), (538830.9900000027, 4814010.691202656), (538830.8900000027, 4814010.7204605695), (538830.7900000027, 4814010.748956295), (538830.6900000027, 4814010.776696575), (538830.5900000028, 4814010.803688156), (538830.4900000028, 4814010.829937786), (538830.3900000028, 4814010.855452209), (538830.2900000028, 4814010.880238168), (538830.1900000029, 4814010.904302409), (538830.0900000029, 4814010.92765168), (538829.9900000029, 4814010.950292724), (538829.8900000029, 4814010.972232286), (538829.790000003, 4814010.993477111), (538829.690000003, 4814011.014033946), (538829.590000003, 4814011.033909535), (538829.490000003, 4814011.053110624), (538829.390000003, 4814011.071643958), (538829.2900000031, 4814011.089516281), (538829.1900000031, 4814011.10673434), (538829.0900000031, 4814011.12330488), (538828.9900000031, 4814011.139234645), (538828.8900000032, 4814011.154530382), (538828.7900000032, 4814011.169198835), (538828.6900000032, 4814011.183246751), (538828.5900000032, 4814011.196680873), (538828.4900000033, 4814011.209507946), (538828.3900000033, 4814011.2217347175), (538828.2900000033, 4814011.233367932), (538828.1900000033, 4814011.244414334), (538828.0900000033, 4814011.2548806695), (538827.9900000034, 4814011.264773683), (538827.8900000034, 4814011.27410012), (538827.7900000034, 4814011.282866726), (538827.6900000034, 4814011.291080248), (538827.5900000035, 4814011.298747428), (538827.4900000035, 4814011.305875013), (538827.3900000035, 4814011.312469747), (538827.2900000035, 4814011.318538378), (538827.1900000036, 4814011.32408765), (538827.0900000036, 4814011.329124305), (538826.9900000036, 4814011.333655092), (538826.8900000036, 4814011.337686757), (538826.7900000036, 4814011.341226042), (538826.6900000037, 4814011.344279695), (538826.5900000037, 4814011.3468544595), (538826.4900000037, 4814011.348957081), (538826.3900000037, 4814011.350594305), (538826.2900000038, 4814011.351772878), (538826.1900000038, 4814011.352499544), (538826.0900000038, 4814011.352781047), (538825.9900000038, 4814011.352624134), (538825.8900000039, 4814011.35203555), (538825.7900000039, 4814011.351022041), (538825.6900000039, 4814011.349590352), (538825.5900000039, 4814011.347747225), (538825.490000004, 4814011.34549941), (538825.390000004, 4814011.34285365), (538825.290000004, 4814011.33981669), (538825.190000004, 4814011.336395276), (538825.090000004, 4814011.332596152), (538824.9900000041, 4814011.328426066), (538824.8900000041, 4814011.323891761), (538824.7900000041, 4814011.318999982), (538824.6900000041, 4814011.313757475), (538824.5900000042, 4814011.308170985), (538824.4900000042, 4814011.302247259), (538824.3900000042, 4814011.29599304), (538824.2900000042, 4814011.289415074), (538824.1900000043, 4814011.282520107), (538824.0900000043, 4814011.2753148815), (538823.9900000043, 4814011.267806146), (538823.8900000043, 4814011.260000645), (538823.7900000043, 4814011.251905123), (538823.6900000044, 4814011.243526325), (538823.5900000044, 4814011.234870997), (538823.4900000044, 4814011.225945885), (538823.3900000044, 4814011.216757733), (538823.2900000045, 4814011.207313287), (538823.1900000045, 4814011.19761929), (538823.0900000045, 4814011.187682491), (538822.9900000045, 4814011.177509633), (538822.8900000046, 4814011.167107462), (538822.7900000046, 4814011.156482723), (538822.6900000046, 4814011.1456421595), (538822.5900000046, 4814011.134592519), (538822.4900000046, 4814011.123340548), (538822.3900000047, 4814011.111892988), (538822.2900000047, 4814011.100256586), (538822.1900000047, 4814011.088438089), (538822.0900000047, 4814011.076444241), (538821.9900000048, 4814011.064281786), (538821.8900000048, 4814011.051957469), (538821.7900000048, 4814011.039478038), (538821.6900000048, 4814011.0268502375), (538821.5900000049, 4814011.01408081), (538821.4900000049, 4814011.001176504), (538821.3900000049, 4814010.988144064), (538821.2900000049, 4814010.974990234), (538821.190000005, 4814010.96172176), (538821.090000005, 4814010.948345388), (538820.990000005, 4814010.934867863), (538820.890000005, 4814010.921295928), (538820.790000005, 4814010.90763633), (538820.6900000051, 4814010.893895816), (538820.5900000051, 4814010.880081129), (538820.4900000051, 4814010.866199016), (538820.3900000051, 4814010.852256218), (538820.2900000052, 4814010.838259486), (538820.1900000052, 4814010.824215561), (538820.0900000052, 4814010.810131192), (538819.9900000052, 4814010.7960131215), (538819.8900000053, 4814010.781868093), (538819.7900000053, 4814010.767702857), (538819.6900000053, 4814010.753524155), (538819.5900000053, 4814010.739338733), (538819.4900000053, 4814010.725153337), (538819.3900000054, 4814010.71097471), (538819.2900000054, 4814010.696809601), (538819.1900000054, 4814010.682664752), (538819.0900000054, 4814010.66854691), (538818.9900000055, 4814010.65446282), (538818.8900000055, 4814010.640419226), (538818.7900000055, 4814010.626422876), (538818.6900000055, 4814010.612480512), (538818.5900000056, 4814010.598598883), (538818.4900000056, 4814010.5847847285), (538818.3900000056, 4814010.571044799), (538818.2900000056, 4814010.557385838), (538818.1900000056, 4814010.543814591), (538818.0900000057, 4814010.530337803), (538817.9900000057, 4814010.516962218), (538817.8900000057, 4814010.503694585), (538817.7900000057, 4814010.490541645), (538817.6900000058, 4814010.477510147), (538817.5900000058, 4814010.464606833), (538817.4900000058, 4814010.451838449), (538817.3900000058, 4814010.439211742), (538817.2900000059, 4814010.426733456), (538817.1900000059, 4814010.414410336), (538817.0900000059, 4814010.402249127), (538816.9900000059, 4814010.390256575), (538816.890000006, 4814010.378439427), (538816.790000006, 4814010.366804425), (538816.690000006, 4814010.355358316), (538816.590000006, 4814010.344107844), (538816.490000006, 4814010.333059756), (538816.3900000061, 4814010.322220797), (538816.2900000061, 4814010.311597712), (538816.1900000061, 4814010.301197245), (538816.0900000061, 4814010.2910261415), (538815.9900000062, 4814010.281091149), (538815.8900000062, 4814010.271399011), (538815.7900000062, 4814010.261956473), (538815.6900000062, 4814010.2527702795), (538815.5900000063, 4814010.243847176), (538815.4900000063, 4814010.23519391), (538815.3900000063, 4814010.226817224), (538815.2900000063, 4814010.218723865), (538815.1900000063, 4814010.210920576), (538815.0900000064, 4814010.203414106), (538814.9900000064, 4814010.1962111965), (538814.8900000064, 4814010.189318595), (538814.7900000064, 4814010.182743046), (538814.6900000065, 4814010.176491295), (538814.5900000065, 4814010.170570087), (538814.4900000065, 4814010.164986167), (538814.3900000065, 4814010.159746281)], 0.1291111111111111, -0.0003, 2.5, 2, 8.8, 0.1, True
    tx = np.arange(0,5.1,.1)
    V, f_path = all_utils.generate_baseline_trajectory(tx,path,v_s,a_s,max_acc,max_jerk,v_g,dt,acc)
    plt.plot([path[0][0],path[-1][0]], [path[0][1],path[-1][1]], 'x')
    plt.plot([x[0] for x in f_path], [x[1] for x in f_path])
    plt.show()
    plt.plot(tx[:-2],V)
    plt.show()


'''
import scipy.special
import constants
def dist_payoffs(dist_arr):
    return scipy.special.erf((dist_arr - constants.DIST_COST_MEAN) / (constants.DIST_COST_SD * math.sqrt(2)))

plt.plot(np.arange(0,50,.1),dist_payoffs(np.arange(0,50,.1)))
plt.show()
'''


'''    
import all_utils
import numpy as np
A1,B1 = [538856.03,4814002.42],[538855.83,4814005.40]
A2,B2 = (538833.73,4813993.36),(538836.72,4813992.06)
pt = (538859.98,4814009.83)
#pt = [538854.94,4814004.24]
#pt = [538830.43,4813993.59]
#pt = [538831.95,4813989.97]
A1,B1,A2,B2,pt = np.asarray(A1),np.asarray(B1),np.asarray(A2),np.asarray(B2),np.asarray(pt)
#print(all_utils.distance_numpy(A1, B1, pt))
#print(all_utils.distance_numpy(A2, B2, pt))
vector_1,vector_2 = pt-B1,pt-A2
unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)
print(np.rad2deg(angle))
print(all_utils.distance_numpy([538856.03,4814002.42], [538836.72,4813992.06], pt))
print(all_utils.distance_numpy([538855.83,4814005.40], [538833.73,4813993.36], pt))
print(A1-B1)
'''
'''
from scipy.interpolate import CubicSpline
from collections import OrderedDict
veh_pos = (5,5)
veh_yaw = math.pi/4
veh = [veh_pos,(veh_pos[0] + (1*np.cos(veh_yaw)), veh_pos[1] + (1*np.sin(veh_yaw)))]

def generate_path(hpx,hpy):
    path = all_utils.split_in_n((hpx[0],hpy[0]), (hpx[1],hpy[1]), 2)
    #path = [(hpx[0],hpy[0])]
    s_x = [0] + [p2-p1 for p1,p2 in list(zip(hpx[:-1],hpx[1:]))]
    s_x = [0] + [sum(s_x[:i+1]) for i in np.arange(1,len(s_x))]
    s_y = [0] + [p2-p1 for p1,p2 in list(zip(hpy[:-1],hpy[1:]))]
    s_y = [0] + [sum(s_y[:i+1]) for i in np.arange(1,len(s_y))]
    indx = np.arange(len(s_x))
    cs_x = CubicSpline(indx,s_x)
    cs_y = CubicSpline(indx,s_y)
    for i_a in np.arange(indx[1],indx[-1]+.1,.1):
        path.append(((path[1][0]+cs_x(i_a)), ((path[1][1]+cs_y(i_a)))))
    max_coeff = max(np.max(np.abs(cs_x.c[-1,1:])), np.max(np.abs(cs_y.c[-1,1:])))
    return path,max_coeff
    
l1 = [(6,8),(7,9),(8,10),(10,12)]
selected_path = ([],np.inf)
cl_list = [l1]
for d in np.arange(0,1.16,.1):
    pl1,pl2 = all_utils.add_parallel(l1, d)
    cl_list.append(pl1)
    cl_list.append(pl2)

for m_idx,l in enumerate(cl_list):
    hpx = [veh_pos[0], veh_pos[0] + (1*np.cos(veh_yaw))]
    hpy = [veh_pos[1], veh_pos[1] + (1*np.sin(veh_yaw))]
    hpx = hpx + [x[0] for x in l]
    hpy = hpy + [x[1] for x in l]
    plt.plot(hpx,hpy,'x')
    path,max_coeff = generate_path(hpx, hpy)
    if max_coeff < selected_path[1]:
        selected_path = (path,max_coeff)
    print(max_coeff)
plt.plot([x[0] for x in path], [x[1] for x in path],'green')
print(max_coeff)
plt.show()
import itertools
'''
'''
pl1,pl2 = all_utils.add_parallel(l1, 2)
for l in [l1,pl1,pl2]:
    plt.plot([x[0] for x in l], [x[1] for x in l])
plt.show()
''' 
'''
seg_info_segments = {1:{'lateral_waypoints':[[1,2,5],[2,0],[3,5]]}, 2:{'lateral_waypoints':[[4],[5],[6]]}, 3:{'lateral_waypoints':[[],[1],[]]}}
z = [list(itertools.chain.from_iterable(x)) for x in zip(*[x['lateral_waypoints'] for x in seg_info_segments.values()])]
l = [[[1,2,5],[2,0],[3,5]], [[4],[5],[6]], [[],[1],[]] ]
z1 = [list(itertools.chain.from_iterable(x)) for x in zip(*l)]
z2 = []
for k,v in seg_info_segments.items():
    lwp = v['lateral_waypoints']
    for idx,wp in enumerate(lwp):
        if len(z2) <= idx:
            z2.append([(x,0) for x in wp])
        else:
            z2[idx].extend([(x,0) for x in wp])

for i,_z2 in enumerate(z2):
    n_j = []
    for j in _z2:
        if j[0] != i+1:
            n_j.append(j)
    z2[i] = n_j
        
                   
print(z)
print(z1)
print(z2)
'''
'''
import ast
import sys
import scipy.special
def dist_payoffs(dist_arr,c,s):
    return scipy.special.erf((dist_arr - c) / (s * math.sqrt(2)))
def dist_payoffs_exp(dist_arr,c,s):
    return scipy.special.erf((dist_arr - c) / (s * 2))
print(dist_payoffs_exp(np.inf, 10, 2))
a = np.linspace(start=0, stop=20, num=300)
p = list(zip([10,7.5,6.25],[2.2,1,1]))
param_dict = {(5,15):(10,2.2),(5,10):(7.5,1),(3.5,9):(6.25,1)}
for par in p:
    plt.figure()
    plt.title(par)
    plt.plot(a,dist_payoffs(a,par[0],par[1]),color='blue')
    plt.plot(a,dist_payoffs_exp(a,par[0],par[1]),color='red')
plt.show()
'''
import os
print(os.path.join(os.path.abspath(__file__),"CACHE"))
    
    
    
    