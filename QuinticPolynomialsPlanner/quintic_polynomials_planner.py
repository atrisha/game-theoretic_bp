"""

Quintic Polynomials Planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Local Path planning And Motion Control For Agv In Positioning](http://ieeexplore.ieee.org/document/637936/)

"""

import math

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import utils
from scipy.stats import halfnorm
import constants
import sys

# parameter


show_animation = False
show_simple_plot = False
show_log = False

class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt
    
    def print_coeff(self,str):
        print(str,self.a0,self.a1,self.a3,self.a4,self.a5)

def quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt,lane_boundary):
    #print('called with')
    plan_type = 'QP'
    MAX_T = 15.0  # maximum time to the goal [s]
    MIN_T = 1  # minimum time to the goal[s]
    T_STEP = 0.1
    #print(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, lane_boundary)
    if show_log:
        print(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, lane_boundary)
        print('target_pos',(gx,gy),'target vel', gv, 'target acc' , ga)
        print('max acc', max_accel, 'max jerk', max_jerk)
        print('yaw',gyaw)
    acc = True
    if max_accel < 0:
        acc = False
    if lane_boundary is not None:
        lb_xs = lane_boundary[0]
        lb_ys = lane_boundary[1]
    else:
        lb_xs,lb_ys = None,None
    """
    quintic polynomial planner

    input
        sx: start x position [m]
        sy: start y position [m]
        syaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]

    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list

    """

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)
    
    #linear_plan_x = utils.linear_planner(sx, vxs, axs, gx, vxg, axg, max_accel, max_jerk, dt)
    #linear_plan_y = utils.linear_planner(sy, vys, ays, gy, vyg, ayg, constants.MAX_LAT_ACC_NORMAL, constants.MAX_ACC_JERK_AGGR, dt)
    
    
    time, rx, ry, ryaw, rv, ra, rj, d2g = [], [], [], [], [], [], [], []
    traj_found = False
    for T in np.arange(MIN_T, MAX_T, T_STEP):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)
        #xqp.print_coeff('x')
        #yqp.print_coeff('y')
        time, rx, ry, ryaw, rv, ra, rj, d2g = [], [], [], [], [], [], [], []
        within_lane = True
        goal_reached = False
        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))
            dist_to_goal = math.hypot(rx[-1] - gx , ry[-1] - gy)
            if dist_to_goal <= constants.CAR_WIDTH/2:
                goal_reached = True
            d2g.append(dist_to_goal)
            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)
            
        
        if lane_boundary is not None:
            goal_sign = np.sign(utils.distance_numpy([lb_xs[0],lb_ys[0]], [lb_xs[1],lb_ys[1]], [gx,gy]))
            dist_to_lane_b = np.sign([utils.distance_numpy([lb_xs[0],lb_ys[0]], [lb_xs[1],lb_ys[1]], [t[0],t[1]]) for t in list(zip(rx,ry))])
            within_lane = np.all(dist_to_lane_b == goal_sign)
        else:
            within_lane = True
        if within_lane:
            if max([abs(i) for i in ra]) <= max_accel:
                if max([abs(i) for i in rj]) <= max_jerk:
                    if show_log:
                        print("found path!!",len(rx)*dt)
                    traj_found = True
                    break
                else:
                    if show_log:
                        print('couldnt find path for',T,'max jerk:',max([abs(i) for i in rj]))
            else:
                    if show_log:
                        print('couldnt find path for',T,'max acc/dec:',max([abs(i) for i in ra]))
        else:
            if show_log:
                print('trajectory outside lane boundary')
    if show_animation:  # pragma: no cover
        for i, _ in enumerate(time):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.grid(True)
            plt.axis("equal")
            
            plot_arrow(sx, sy, syaw)
            plot_arrow(gx, gy, gyaw)
            plot_arrow(rx[i], ry[i], ryaw[i])
            
            plt.title("Time[s]:" + str(time[i])[0:4] +
                      " v[m/s]:" + str(rv[i])[0:4] +
                      " a[m/ss]:" + str(ra[i])[0:4] +
                      " jerk[m/sss]:" + str(rj[i])[0:4],
                      )
            plt.pause(0.001)
        plt.show()
    if traj_found:
        if show_log:
            print('trajectory horizon',T,'secs')
    if not traj_found:
        return None
    else:
        return np.array(time), np.array(rx), np.array(ry), np.array(ryaw), np.array(rv), np.array(ra), np.array(rj), T, plan_type

def check_traj_safety(rx,vx,ax,jx,max_accel,max_jerk,dt,axis):
    T = len(rx)*dt
    if max([abs(i) for i in ax]) <= max_accel:
        if max([abs(i) for i in jx]) <= max_jerk:
            if show_log:
                print("found path!!",T)
            return True
        else:
            if show_log:
                print('couldnt find',axis,' path for',T,'max jerk:',max([abs(i) for i in jx]))
            return False
            
    else:
        if show_log:
            print('couldnt find',axis,' path for',T,'max acc:',max([abs(i) for i in ax]))
        return False

def check_collision(rx,dist_to_lead,vxg,lvax,dt):
    lx = [dist_to_lead]
    for i in np.arange(len(rx)):
        dist_to_lead += (vxg*dt + (0.5*lvax*dt**2))
        lx.append(dist_to_lead)
    dist = [math.hypot(x, y) for x,y in zip(rx,lx)]
    #print('dist values',min(dist),max(dist)) 
    

            
        
         

def car_following_planner(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_val, max_jerk, dt,lane_boundary,center_line):
    #max_accel_long = constants.MAX_LONG_ACC_NORMAL if accel_param is 'NORMAL' else constants.MAX_LONG_ACC_AGGR
    #max_accel_lat = constants.MAX_LAT_ACC_NORMAL if accel_param is 'NORMAL' else constants.MAX_LAT_ACC_AGGR
    max_accel_long,max_accel_lat = accel_val[0],accel_val[1]
    #print('called with')
    #print(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_val, max_jerk, dt,lane_boundary,center_line)
    plan_type = 'QP'
    '''
    if show_log:
        print(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, lane_boundary)
        print('target_pos',(gx,gy),'target vel', gv, 'target acc' , ga)
        print('max acc', max_accel, 'max jerk', max_jerk)
        print('yaw',gyaw)
    '''
    acc = True
    if max_accel_long < 0:
        acc = False
    if lane_boundary is not None:
        lb_xs = lane_boundary[0]
        lb_ys = lane_boundary[1]
    else:
        lb_xs,lb_ys = None,None
    """
    quintic polynomial planner

    input
        sx: start x position [m]
        sy: start y position [m]
        syaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]

    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list

    """
    lead_vehicle_present = False
    if lvx is not None:
        lead_vehicle_present = True
        MAX_TY = 20.0  # maximum time to the goal [s]
        MIN_T = 1  # minimum time to the goal[s]
        MAX_TX = 20.0
        T_STEP = 0.1
    else:
        MAX_TY = 20.0  # maximum time to the goal [s]
        MIN_T = .01  # minimum time to the goal[s]
        MAX_TX = 20.0
        T_STEP = 0.1
    
    center_line_angle = math.atan2((center_line[0][1]-center_line[1][1]),(center_line[0][0]-center_line[1][0]))
    if center_line_angle > 0:
        center_line_angle = center_line_angle - math.pi
    dist_to_centerline = utils.distance_numpy([center_line[0][0],center_line[0][1]],[center_line[1][0],center_line[1][1]],[sx,sy])
    sv_angle_with_cl = syaw - center_line_angle
    sv_angle_in_map = math.atan2(sy,sx)
    fresnet_origin = (sx - dist_to_centerline*math.cos(sv_angle_in_map), sy - dist_to_centerline*math.sin(sv_angle_in_map))
    vxs = sv * math.cos(sv_angle_with_cl)
    vys = sv * math.sin(sv_angle_with_cl)
    axs = sax * math.cos(sv_angle_with_cl)
    ays = say
    axg = 0
    ayg = 0
    if lead_vehicle_present:
        lv_angle_with_cl = lvyaw - center_line_angle
        dist_to_lead = math.hypot(lvx-sx, lvy-sy)
        dist_to_lead_cl_proj = dist_to_lead
        vxg = lvv * math.cos(lv_angle_with_cl)
        vyg = lvv * math.sin(lv_angle_with_cl)
        #lvax = 
        goal_x = dist_to_lead_cl_proj - constants.CAR_FOLLOWING_SAFETY_DISTANCE
        if acc > 0:
            axg_list = [0]
            axg_list = np.arange(0,acc+.5,0.5)
        else:
            axg_list = [0]
            axg_list = np.arange(0,acc-.5,-0.5)
    else:
        vyg = constants.TARGET_VEL*math.sin(sv_angle_with_cl)
        if constants.TARGET_VEL*math.cos(sv_angle_with_cl) - vxs > 0:
            time_to_target_vel = (constants.TARGET_VEL*math.cos(sv_angle_with_cl) - vxs)/constants.MAX_LONG_ACC_NORMAL
            acc = constants.MAX_LONG_ACC_NORMAL
        else:
            time_to_target_vel = (constants.TARGET_VEL*math.cos(sv_angle_with_cl) - vxs)/constants.MAX_LONG_DEC_NORMAL
            acc = constants.MAX_LONG_DEC_NORMAL
        dist_to_target_vel = vxs*time_to_target_vel + (0.5*acc*time_to_target_vel**2)
        goal_x = dist_to_target_vel
        vxg = constants.TARGET_VEL * math.cos(sv_angle_with_cl)
        if acc > 0:
            axg_list = [0]
            axg_list = np.arange(0,acc+.5,0.5)
        else:
            axg_list = [0]
            axg_list = np.arange(0,acc-.5,-0.5)
    goal_y = 0
    
    traj_found = False
    for axg in axg_list:
        for TX in np.arange(MIN_T, MAX_TX, T_STEP):
            if lead_vehicle_present:
                goal_x_upper = goal_x + max(0,vxg*TX + 0.5*lvax*(TX**2))
                end_states = [goal_x_upper - x for x in halfnorm.rvs(loc=0,scale=10,size=10)]
            else:
                end_states = [goal_x + x for x in np.arange(.25,25,.25)]
            for gx in end_states:
                xqp = QuinticPolynomial(0, vxs, axs, gx, vxg, axg, TX)
                time_x, rx, rvx, rax, rjx = [], [], [], [], []
                traj_x_found = False
                for t in np.arange(0.0, TX + dt, dt):
                    time_x.append(t)
                    rx.append(xqp.calc_point(t))
                    vx = xqp.calc_first_derivative(t)
                    rvx.append(vx)
                    ax = xqp.calc_second_derivative(t)
                    rax.append(ax)
                    jx = xqp.calc_third_derivative(t)
                    rjx.append(jx)
                if check_traj_safety(rx,rvx,rax,rjx,max_accel_long,max_jerk,dt,'lon'):
                    traj_found = True
                    #print('found path for',gx,TX)
                    break
            if traj_found:
                if lead_vehicle_present:
                    #check_collision(rx,dist_to_lead,vxg,lvax,dt)
                    traj_x_found = True
                    break
                else:
                    traj_x_found = True
                    break
        if traj_found:
            break
    if not traj_x_found:
        #print(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_param, max_jerk, dt,lane_boundary,center_line)
        #print('lead vehicle present',lead_vehicle_present)
        time_x, rx, rvx, rax, rjx = utils.linear_planner(0, vxs, axs, goal_x, vxg, axg, max_accel_long, max_jerk, dt)
        plan_type = 'LP'
        traj_x_found = True
        #sys.exit('car following planner trajectory not found')       
    for TY in np.arange(MIN_T, MAX_TY, T_STEP):
        yqp = QuinticPolynomial(dist_to_centerline, vys, ays, goal_y, vyg, ayg, TY)
        time_y, ry, rvy, ray, rjy = [], [], [], [], []
        traj_y_found = False
        for t in np.arange(0.0, TY + dt, dt):
            time_y.append(t)
            ry.append(yqp.calc_point(t))
            vy = yqp.calc_first_derivative(t)
            rvy.append(vy)
            ay = yqp.calc_second_derivative(t)
            ray.append(ay)
            jy = yqp.calc_third_derivative(t)
            rjy.append(jy)
        if check_traj_safety(ry,rvy,ray,rjy,max_accel_lat,max_jerk,dt,'lat'):
            traj_y_found = True
            break
    if show_log and traj_x_found and traj_y_found:
        print('times are',TX,TY,axg)
    
    if not traj_y_found:
        #print(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_param, max_jerk, dt,lane_boundary,center_line)
        #print('lead vehicle present',lead_vehicle_present)
        #sys.exit('car following planner trajectory not found')
        time_y, ry, rvy, ray, rjy = utils.linear_planner(dist_to_centerline, vys, ays, goal_y, vyg, ayg, max_accel_lat, max_jerk, dt)
        plan_type = 'LP'
        traj_y_found = True       
    T = 0
    if traj_x_found and traj_y_found:
        ''' padding lateral and longitudinal trajectory for the same length '''
        if len(rx) > len(ry):
            T = TX
            ry += [0]*(len(rx)-len(ry))
            time_y += [time_y[-1]]*(len(time_x)-len(time_y))
            rvy += [rvy[-1]]*(len(rvx)-len(rvy))
            ray += [ray[-1]]*(len(rax)-len(ray))
            rjy += [rjy[-1]]*(len(rjx)-len(rjy))
        elif len(rx) < len(ry):
            T = TY
            for i in np.arange(0,len(ry)-len(rx)+1):
                rx.append(rx[-1] + rvx[-1]*(i+1)*dt)
            time_x += [time_x[-1]]*(len(time_y)-len(time_x))
            rvx += [rvx[-1]]*(len(rvy)-len(rvx))
            rax += [0]*(len(ray)-len(rax))
            rjx += [0]*(len(rjy)-len(rjx))
        
        time, ryaw, rv, ra, rj = [], [], [], [], [] 
        ''' set the goal states since they have been reached anyway '''
        ry[-1],rvy[-1] = 0.0,0.0
        ''' merge lateral and longitudinal into a single trajectory'''
        for vx,vy in zip(rvx,rvy):
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)
            
        for ax,ay in zip(rax,ray):
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)
            
        for jx,jy in zip(rjx,rjy):
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)
        time = time_x
        rx_map, ry_map = utils.fresnet_to_map(fresnet_origin[0], fresnet_origin[1], rx, ry, center_line_angle)    
        ryaw_map = [y - abs(center_line_angle) for y in ryaw]
        #plt.plot([center_line[0][0],center_line[1][0]],[center_line[0][1],center_line[1][1]],'b-')
        #show_summary(time, rx_map, ry_map, ryaw_map, rv, ra, rj)
        #print('traj found', T)
        return np.array(time), np.array(rx_map), np.array(ry_map), np.array(ryaw_map), np.array(rv), np.array(ra), np.array(rj), T, plan_type
    else:
        print(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, accel_val, max_jerk, dt,lane_boundary,center_line)
        return None
    
    
def plot_arrow(x, y, yaw, length=0.5, width=0.25, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def show_summary(time, x, y, yaw, v, a, j):
        if len(x) > len(y):
            y += [y[-1]]*(len(x)-len(y))
        elif len(x) < len(y):
            x += [x[-1]]*(len(y)-len(x))
        plt.axis('equal')
        plt.plot(x, y, "-r")
        if yaw is not None:
            plt.subplots()
            plt.plot(time, [np.rad2deg(i) for i in yaw], "-r")
            plt.xlabel("Time[s]")
            plt.ylabel("Yaw[deg]")
            plt.grid(True)

        plt.subplots()
        plt.plot(time, v, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[m/s]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, a, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("accel[m/ss]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, j, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("jerk[m/sss]")
        plt.grid(True)

        plt.show()


def main():
    print(__file__ + " start!!")

    sx,sy,syaw,sv,sa,gx,gy,gyaw,gv,ga,max_accel,max_jerk,dt,lane_boundary = \
    0.0, 0.0, 0.0, 17.19, 0, 46.0, 0.0, 0.0, 18.45, 1.13, 2.47, 10, 0.1, None
    '''
     = 538842.32  # start x position [m]
    sy = 4814000.61  # start y position [m]
    syaw = 2.23  # start yaw angle [rad]
    sv = 0.18  # start speed [m/s]
    sa = 0.1  # start accel [m/ss]
    gx = 538815.15  # goal x position [m]
    gy = 4814006.58  # goal y position [m]
    gyaw = 3.6  # goal yaw angle [rad]
    gv = 9.7  # goal speed [m/s]
    ga = 1.6  # goal accel [m/ss]
    max_accel = 3.07  # max accel [m/ss]
    max_jerk = 5  # max jerk [m/sss]
    dt = 0.1  # time tick [s]
    '''

    time, x, y, yaw, v, a, j = quintic_polynomials_planner(
        sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
    
    if show_animation:  # pragma: no cover
        plt.plot(x, y, "-r")

        plt.subplots()
        plt.plot(time, [np.rad2deg(i) for i in yaw], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Yaw[deg]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, v, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[m/s]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, a, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("accel[m/ss]")
        plt.grid(True)

        plt.subplots()
        plt.plot(time, j, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("jerk[m/sss]")
        plt.grid(True)

        plt.show()
        
''' with a lead vehicle '''    
#sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, max_accel, max_jerk, dt,lane_boundary,center_line = 538879.7122240972,4814019.592224097,5.274310542614675,16.986472222222222,0.0,-0.0,538835.08,4814005.55,5.3053,17.876722222222224,0.6537,0.0555,'NORMAL',2,0.1,None,[(538822.54,4814023.19),(538842.94,4813993.87)]
''' without a lead vehicle '''
#sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, max_accel, max_jerk, dt,lane_boundary,center_line = 538839.93,4813997.24,5.3094,14.766527777777778,0.0014616499114032647,-0.0021502510403426916,None,None,None,None,None,None,(3.6,6),2,0.1,None,[(538822.54,4814023.19),(538842.94,4813993.87)]

#car_following_planner(sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, max_accel, max_jerk, dt, lane_boundary, center_line)

''' left turn '''
#sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt,lane_boundary = 538842.39,4814000.65,2.1566,0.1291111111111111,-0.0003,538814.15,4814007.58,-2.765017735489607,7.50979619831,0.884725431993,3.6,2,0.1,[[538827.81,538847.55],[4814025.31,4813996.34]]
#quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt, lane_boundary)