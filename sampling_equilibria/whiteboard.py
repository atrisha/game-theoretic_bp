'''
Created on Sep 29, 2020

@author: Atrisha
'''


import numpy as np

def calc_bhattacharya_distance(mu1,sigma1,mu2,sigma2):
    return (0.25*np.log(0.25*((sigma1**2/sigma2**2)+(sigma2**2/sigma1**2)+2))) + (0.25*(((mu1-mu2)**2)/( (sigma1**2)+(sigma2**2) )))




import numpy as np
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


class AnalogPlot:
    def __init__(self, data, display_len):
        self.buff = deque(np.zeros(display_len))
        self.display_len = display_len
        self.data = data

        # set up the plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, xlim=(0, t_max), ylim=(-1, 1))
        self.ax.set_xticks((0,t_max))
        self.lines = self.ax.plot([], [])

        # setup the animation
        self.cur_frame = 0
        self.anim = animation.FuncAnimation(self.fig, self._update,
                                            interval=1.0)

        # setup the animation control
        self.anim_running = True


    def _add_to_buff(self, buf, val):
        if len(buf) < self.display_len:
            buf.appendLeft(val)
        else:
            buf.popleft()
            buf.append(val)

    def _update(self, frame):
        frame = self.cur_frame
        self._add_to_buff(self.buff, self.data[frame:frame+1])
        self.lines[0].set_data(range(self.display_len), self.buff)

        self.ax.set_xticklabels((str(frame), str(frame+self.display_len)))

        self.time_slider.eventson = False
        self.time_slider.set_val(frame)
        self.time_slider.eventson = True

        self.cur_frame += 1

        return self.lines

    def _pause(self, event):
        if self.anim_running:
            self.pause_button.label.set_text('play')
            self.anim.event_source.stop()
            self.anim_running = False
        else:
            self.pause_button.label.set_text('pause')
            self.anim.event_source.start()
            self.anim_running = True

    def _reset(self, event):
        self._set_val(0)


    def _set_val(self, frame=0):
        frame = int(frame)
        self.cur_frame = frame
        new_start = frame - self.display_len
        if new_start >= 0:
            self.buff = deque(self.data[new_start:frame])
        else:
            self.buff = deque(np.concatenate((np.zeros(np.abs(new_start)), self.data[:frame])))

        self.anim.event_source.stop()
        self.anim = animation.FuncAnimation(self.fig, self._update,
                                            interval=1.0)
        self.anim_running = True


    def animate(self):
        pause_ax = self.fig.add_axes((0.7, 0.025, 0.1, 0.04))
        pause_button = Button(pause_ax, 'pause', hovercolor='0.975')
        pause_button.on_clicked(self._pause)
        self.pause_button = pause_button
        
        reset_ax = self.fig.add_axes((0.8, 0.025, 0.1, 0.04))
        reset_button = Button(reset_ax, 'reset', hovercolor='0.975')
        reset_button.on_clicked(self._reset)

        slider_ax = self.fig.add_axes((0.1, 0.025, 0.5, 0.04))
        self.time_slider = Slider(slider_ax, label='Time',
                                  valmin=0, valmax=self.data.shape[0],
                                  valinit=0.0)

        self.time_slider.on_changed(self._set_val)
        
        plt.show()

'''
t_max = 100
lin_sig = np.linspace(0, 1, 1000)
analog_plot = AnalogPlot(lin_sig, t_max)
analog_plot.animate()
'''
'''
import sqlite3
import matplotlib.pyplot as plt
file_id = '769'
conn_traj = sqlite3.connect('D:\\intersections_dataset\\dataset\\'+file_id+'\\uni_weber_generated_trajectories_'+file_id+'.db')
c_traj = conn_traj.cursor()
'''
'''        
q_string = "select * from GENERATED_BASELINE_TRAJECTORY where GENERATED_BASELINE_TRAJECTORY.TRAJECTORY_INFO_ID=71316 order by time"
c_traj.execute(q_string)
res = c_traj.fetchall()
plt.plot([x[2] for x in res],[x[6] for x in res])
plt.show()
'''
x = [1,2,3]
x += [2]*1
print(x)