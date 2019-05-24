import numpy as np
import scipy.linalg
from systems import Car, DubinsCar
import scipy.linalg as la
from matplotlib import pyplot as plt

def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn

def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
    
    #first, try to solve the ricatti equation
    P = np.matrix(solve_DARE_with_iteration(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    
    # eigVals, eigVecs = scipy.linalg.eig(A-B*P)
    
    return K


if __name__ == '__main__':
    dt = 1
    Q = np.matrix("1.0 0 0; 0 1.0 0; 0 0 1.0")
    R = np.matrix("1.0 0.0; 0 100.0")
    # Q = np.diag([1.0, 1.0, 1.0])
    # R = np.diag([1.0, 100.0])
    # x = np.array([[1], [1], [1]])
    # x = np.array([1, 1, 1])
    x = np.matrix("-10.0 ; -10.0 ; 0.1")
    # print(x)
    # print(x[2,0])
    # print(x[0][0])
    # A = np.zeros((3, 3))
    # while (True):
    A = np.matrix("1.0 0 0; 0 1.0 0; 0 0 1.0")
    pos_x = []
    pos_y = []
    for i in range(20):
        B = np.array([[np.cos(x[2, 0])*dt, 0], [np.sin(x[2, 0])*dt, 0], [0, 1]])
        K = dlqr(A, B, Q, R)
        u = -K*x
        # print(u[0, 0])
        # u[0, 0] = min(max(u[0, 0], 0.0), 2)
        # u[1, 0] = min(max(u[0, 0], -1), 1)
        # print(u)
        x = A*x + B*u
        pos_x.append(x[0, 0])
        pos_y.append(x[1, 0])
    plt.figure()
    plt.plot(pos_x, pos_y)
    plt.show()