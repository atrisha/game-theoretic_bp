'''
Created on Mar 11, 2020

@author: authorA
'''


def linear_planner(sx, vxs, axs, gx, vxg, axg, max_accel,max_jerk,dt):
    goal_reached = False
    t = 0
    time_x, rx, rvx, rax, rjx = [], [], [], [], []
    time_x.append(t)
    rx.append(sx)
    rvx.append(vxs)
    rax.append(axs)
    rjx.append(0)
    a = axs
    v = vxs
    while not goal_reached:
        t += dt
        time_x.append(t)
        if vxg > vxs:
            a = a + (max_jerk*dt)
            if a > max_accel:
                a = max_accel
            d = sx + (vxs * dt) + (0.5*a*dt**2)
            v = v + a*dt
            rx.append(rx[-1]+d)
            rvx.append(v)
            rax.append(a)
            rjx.append((abs(rax[-2]-rax[-1]))/dt)
            if rx[-1] > gx or v > vxg:
                goal_reached = True
        else:
            a = a - (max_jerk*dt)
            if abs(a) > max_accel:
                a = -max_accel
            d = sx + (vxs * dt) + (0.5*a*dt**2)
            v = v + a*dt
            rx.append(rx[-1]+d)
            rvx.append(v)
            rax.append(a)
            rjx.append((abs(rax[-2]-rax[-1]))/dt)
            if rx[-1] > gx or v < vxg:
                goal_reached = True
    return time_x, rx, rvx, rax, rjx
