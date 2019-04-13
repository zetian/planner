import math
import numpy as np
import matplotlib.pyplot as plt

class QuarticPolynomialCurve1d:
    """
    Quartic Polynomial class
    """
    
    start_ = []
    end_ = []
    coef_ = [None]*5
    # time_span_

    def __init__(self, start, end, param):
        self.start_ = start
        self.end_ = end
        self.time_span_ = param
        self.ComputeCoefficients(start[0], start[1], start[2], end[0], end[1], param)

    def Evaluate(self, order, p):
        if order == 0:
            return (((self.coef_[4]*p + self.coef_[3])*p + self.coef_[2])*p + self.coef_[1])*p + self.coef_[0]
        elif order == 1:
            return ((4*self.coef_[4]*p + 3*self.coef_[3])*p + 2*self.coef_[2])*p + self.coef_[1]
        elif order == 2:
            return (12*self.coef_[4]*p + 6*self.coef_[3])*p + 2*self.coef_[2]
        elif order == 3:
            return 24*self.coef_[4]*p + 6*self.coef_[3]
        elif order == 4:
            return 24*self.coef_[4]
        return 0

    def ComputeCoefficients(self, x0, dx0, ddx0, dx1, ddx1, p):
        self.coef_[0] = x0
        self.coef_[1] = dx0
        self.coef_[2] = ddx0/2.0
        b0 = dx1 - ddx0*p - dx0
        b1 = ddx1 - ddx0
        p2 = p*p
        p3 = p2*p
        self.coef_[3] = (3*b0 - b1*p)/(3*p2)
        self.coef_[4] = (-2*b0 + b1*p)/(4*p3)

class QuinticPolynomialCurve1d:
    """
    Quintic Polynomial class
    """
    start_ = []
    end_ = []
    coef_ = [None]*6

    def __init__(self, start, end, param):
        self.start_ = start
        self.end_ = end
        self.time_span_ = param
        self.ComputeCoefficients(start[0], start[1], start[2], end[0], end[1], end[2], param)
    
    def Evaluate(self, order, p):
        if order == 0:
            return ((((self.coef_[5]*p + self.coef_[4])*p + self.coef_[3])*p + self.coef_[2])*p + self.coef_[1])*p + self.coef_[0]
        elif order == 1:
            return (((5*self.coef_[5]*p + 4*self.coef_[4])*p + 3*self.coef_[3])*p + 2 * self.coef_[2])*p + self.coef_[1]
        elif order == 2:
            return  (((20*self.coef_[5] *p + 12*self.coef_[4])*p) + 6*self.coef_[3])*p + 2*self.coef_[2]
        elif order == 3:
            return (60*self.coef_[5]*p + 24*self.coef_[4]) * p + 6*self.coef_[3]
        elif order == 4:
            return 120*self.coef_[5]*p + 24*self.coef_[4]
        elif order == 5:
            return 120*self.coef_[5]
        return 0


    def ComputeCoefficients(self, x0, dx0, ddx0, x1, dx1, ddx1, p):
        self.coef_[0] = x0
        self.coef_[1] = dx0
        self.coef_[2] = ddx0/2.0

        p2 = p*p
        p3 = p*p2

        c0 = (x1 - 0.5*p2*ddx0 - dx0*p - x0)/p3
        c1 = (dx1 - ddx0*p - dx0)/p2
        c2 = (ddx1 - ddx0)/p
        self.coef_[3] = 0.5*(20.0 * c0 - 8.0 * c1 + c2)
        self.coef_[4] = (-15.0*c0 + 7.0*c1 - c2)/p
        self.coef_[5] = (6.0*c0 - 3.0*c1 + 0.5*c2)/p2



if __name__ == "__main__":
    start = [0, 0, 0]
    end = [10, 0]
    time_end = 20
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

    start = [0, 0, 0]
    end = [10, 0, 0]
    quintic_poly = QuinticPolynomialCurve1d(start, end, time_end)
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