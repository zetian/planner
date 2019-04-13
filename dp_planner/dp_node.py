# import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../math/")
from polynomial_curve1d import *

class DPNode:
    def __init__(self, s, l, t):
        self.s_ = s
        self.l_ = l
        self.t_ = t


# Test QuarticPolynomialCurve1d
start = [0, 10, 0]
end = [50, 0, 0]
time_end = 8
quartic_poly = QuinticPolynomialCurve1d(start, end, time_end)
time = np.linspace(0, time_end, 200)
pos = quartic_poly.Evaluate(0, time)
vel = quartic_poly.Evaluate(1, time)
accel = quartic_poly.Evaluate(2, time)

plt.figure()
plt.subplot(311)
plt.plot(time, pos, color = 'black')
plt.title('Quartic Polynomial Curve')
plt.ylabel('Position(m)')
plt.xlim(0, time_end) 
plt.subplot(312)
plt.plot(time, vel, color = 'red')
plt.ylabel('Velocity(m/s)')
plt.xlim(0, time_end) 
plt.subplot(313)
plt.plot(time, accel)
plt.xlabel('time(s)')
plt.ylabel('Acceleration(m/s^2)')
plt.xlim(0, time_end) 
plt.show()

