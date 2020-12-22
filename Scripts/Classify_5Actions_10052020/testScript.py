import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import quaternion as qt
import skinematics as ski
import numba

# Calculate the angular velocity from quaternions

q = ski.quat.Quaternion([0,0.2,0.1])
# rm = q.export()
# fick = q.export('Fick')

q = np.array([[-0.144,0.97,0.05,-0.16],
     [-0.107,0.9572, -0.05,0.26]])
q = qt.as_quat_array(q)
t = np.array([19.3025,19.3225])
#
# t1 = 19.3
# t2 = 19.32
# t_rate = 1/ (t2 - t1)

#omega = ski.quat.calc_angvel(q,rate=t_rate, winSize=1, order=2)
omega = qt.quaternion_time_series.angular_velocity(q,t)
