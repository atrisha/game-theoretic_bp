import math

import matplotlib.pyplot as plt
import numpy as np
import sqlite3

# parameter
MAX_T = 10.0  # maximum time to the goal [s]
MIN_T = 1.0  # minimum time to the goal[s]

show_animation = True
show_simple_plot = False

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # s_i,v_i,a_i , s_f,v_f,a_f
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4],
                      [3 * time ** 2, 4 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t 

        return xt


def quartic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt,lane_boundary):
    #print('called with')
    print('starts pos,vel,acc',(sx,sy,sv,sa),'target_pos',(gx,gy),'target vel', gv, 'target acc' , ga, max_accel, max_jerk, dt)
    acc = True
    if max_accel < 0:
        acc = False
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

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []
    
    px_l,py_l,vx_l,vy_l = [],[],[],[]
    traj_found = False
    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuarticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuarticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []
        px_l,py_l,vx_l,vy_l = [],[],[],[]
        
        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))
            px_l.append(rx[-1])
            py_l.append(ry[-1])

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)
            yaw = math.atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)
            vx_l.append(vx)
            vy_l.append(vy)
            

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
        
        if max([abs(i) for i in ra]) <= max_accel:
            if max([abs(i) for i in rj]) <= max_jerk:
                print("found path!!",T)
                traj_found = True
                break
            else:
                f = 1
                print('couldnt find path for',T,'max jerk:',max([abs(i) for i in rj]))
        else:
            f = 1
            print('couldnt find path for',T,'max acc/dec:',max([abs(i) for i in ra]))

    if show_animation and traj_found:  # pragma: no cover
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
    
    if T == MAX_T-1 and not traj_found:
        return None
    else:
        return time, rx, ry, ryaw, rv, ra, rj, px_l, py_l, vx_l, vy_l


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
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

def show_summary(time, x, y, yaw, v, a, j, px_l, py_l, vx_l, vy_l):
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
        
        plt.subplots()
        plt.plot(time, px_l, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("p_x[m]")
        plt.grid(True)
        
        plt.subplots()
        plt.plot(time, py_l, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("p_y[m]")
        plt.grid(True)
        
        plt.subplots()
        plt.plot(time, vx_l, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("v_x[m/s]")
        plt.grid(True)
        
        plt.subplots()
        plt.plot(time, vy_l, "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("v_y[m/s]")
        plt.grid(True)
        
        plt.show()


def main():
    print(__file__ + " start!!")
    '''
    sx = 538842.32  # start x position [m]
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
    
    #gx_l = np.random.normal(538840.044317,1,10)
    #gy_l = np.random.normal(4813992.77308,1,10)
    #gv_l = np.random.normal(17.19,3,100)
    ct = 0
    sx, sy, syaw, sv, sax, say, lvx, lvy, lvyaw, lvv, lvax, lvay, max_accel, max_jerk, dt,lane_boundary,center_line = 538839.93,4813997.24,5.3094,14.766527777777778,0.0014616499114032647,-0.0021502510403426916,None,None,None,None,None,None,'NORMAL',2,0.1,None,[(538822.54,4814023.19),(538842.94,4813993.87)]
    for tv in np.arange(18,22,.1):
        res = quartic_polynomials_planner(
            0, 0, 0, 14.76, 0.00018545, 5, 0, 0, tv, 0.0001, 1.47, 2, 0.1,lane_boundary)
        if res is not None:
            print('found path for',5,0,0)
            time, x, y, yaw, v, a, j, px_l, py_l, vx_l, vy_l = res
            show_summary(time, x, y, yaw, v, a, j, px_l, py_l, vx_l, vy_l)
        else:
            print('cannot find path for',5,0,0)
        ct += 1
        print(ct)
    
    '''
    if show_animation or show_simple_plot:  # pragma: no cover
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
    '''
if __name__ == '__main__':
    main()
