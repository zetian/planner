import math
import numpy as np
import matplotlib.pyplot as plt

class QuarticPolynomialCurve1d:
    """
    Quartic Polynomial class
    """
    
    start = []
    end = []
    coef = [0.0]*5

    def __init__(self, start, end, param):
        self.start = start
        self.end = end
        self.time_span = param
        self.ComputeCoefficients(start[0], start[1], start[2], end[0], end[1], param)

    def Evaluate(self, order, p):
        if order == 0:
            return (((self.coef[4]*p + self.coef[3])*p + self.coef[2])*p + self.coef[1])*p + self.coef[0]
        elif order == 1:
            return ((4*self.coef[4]*p + 3*self.coef[3])*p + 2*self.coef[2])*p + self.coef[1]
        elif order == 2:
            return (12*self.coef[4]*p + 6*self.coef[3])*p + 2*self.coef[2]
        elif order == 3:
            return 24*self.coef[4]*p + 6*self.coef[3]
        elif order == 4:
            return 24*self.coef[4]
        return 0

    def ComputeCoefficients(self, x0, dx0, ddx0, dx1, ddx1, p):
        self.coef[0] = x0
        self.coef[1] = dx0
        self.coef[2] = ddx0/2.0
        b0 = dx1 - ddx0*p - dx0
        b1 = ddx1 - ddx0
        p2 = p*p
        p3 = p2*p
        self.coef[3] = (3.0*b0 - b1*p)/(3.0*p2)
        self.coef[4] = (-2.0*b0 + b1*p)/(4.0*p3)

class QuinticPolynomialCurve1d:
    """
    Quintic Polynomial class
    """
    start = []
    end = []
    coef = [None]*6

    def __init__(self, start, end, param):
        self.start = start
        self.end = end
        self.time_span = param
        self.ComputeCoefficients(start[0], start[1], start[2], end[0], end[1], end[2], param)
    
    def Evaluate(self, order, p):
        if order == 0:
            return ((((self.coef[5]*p + self.coef[4])*p + self.coef[3])*p + self.coef[2])*p + self.coef[1])*p + self.coef[0]
        elif order == 1:
            return (((5*self.coef[5]*p + 4*self.coef[4])*p + 3*self.coef[3])*p + 2 * self.coef[2])*p + self.coef[1]
        elif order == 2:
            return  (((20*self.coef[5] *p + 12*self.coef[4])*p) + 6*self.coef[3])*p + 2*self.coef[2]
        elif order == 3:
            return (60*self.coef[5]*p + 24*self.coef[4]) * p + 6*self.coef[3]
        elif order == 4:
            return 120*self.coef[5]*p + 24*self.coef[4]
        elif order == 5:
            return 120*self.coef[5]
        return 0

    def ComputeCoefficients(self, x0, dx0, ddx0, x1, dx1, ddx1, p):
        self.coef[0] = x0
        self.coef[1] = dx0
        self.coef[2] = ddx0/2.0
        p2 = p*p
        p3 = p*p2
        c0 = (x1 - 0.5*p2*ddx0 - dx0*p - x0)/p3
        c1 = (dx1 - ddx0*p - dx0)/p2
        c2 = (ddx1 - ddx0)/p
        self.coef[3] = 0.5*(20.0*c0 - 8.0*c1 + c2)
        self.coef[4] = (-15.0*c0 + 7.0*c1 - c2)/p
        self.coef[5] = (6.0*c0 - 3.0*c1 + 0.5*c2)/p2

if __name__ == "__main__":
    # Test QuarticPolynomialCurve1d
    start = [0, 10, 0]
    end = [20, 0]
    time_end = 8
    quartic_poly = QuarticPolynomialCurve1d(start, end, time_end)
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

    # Test QuinticPolynomialCurve1d
    start = [0, 0, 0]
    end = [10, 0, 0]
    time_end = 20
    quintic_poly = QuinticPolynomialCurve1d(start, end, time_end)
    time = np.linspace(0, time_end, 200)
    pos = quintic_poly.Evaluate(0, time)
    vel = quintic_poly.Evaluate(1, time)
    accel = quintic_poly.Evaluate(2, time)

    plt.figure()
    plt.subplot(311)
    plt.plot(time, pos, color = 'black')
    plt.title('Quintic Polynomial Curve')
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