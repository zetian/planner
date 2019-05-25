import numpy as np
import scipy
import random
from numpy.linalg import inv
from matplotlib import pyplot as plt
import timeit
from systems import Car

"iterative LQR with Quadratic cost"


class iterative_LQR_quadratic_cost:
    """
    iterative LQR can be used as a controller/trajectory optimizer.
    Reference:
    Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization
    https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems
    https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf
    cost function: x'Qx + u'Ru
    """

    def __init__(self, sys, target_states, dt):
        """
        iLQR initilization
        """
        self.target_states = target_states
        self.prediction_horizon = self.target_states.shape[1]
        self.dt = dt
        self.converge = False
        self.system = sys
        self.n_states = sys.state_size
        self.m_inputs = sys.control_size
        self.Q = sys.Q
        self.R = sys.R
        self.Qf = sys.Q_f
        self.maxIter = 30
        self.LM_parameter = 0.0
        self.states = np.zeros(
            (self.n_states, self.prediction_horizon))
        self.inputs = np.zeros(
            (self.m_inputs, self.prediction_horizon - 1))

    def cost(self):
        states_diff = self.states - self.target_states
        cost = 0.0
        for i in range(self.prediction_horizon-1):
            state = np.reshape(states_diff[:, i], (-1, 1))
            control = np.reshape(self.inputs[:, i], (-1, 1))
            cost += np.dot(np.dot(state.T, self.Q), state) + \
                np.dot(np.dot(control.T, self.R), control)
        state = np.reshape(states_diff[:, -1], (-1, 1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        return cost

    def compute_dl_dx(self, x, xr):
        dl_dx = 2.0*np.dot(self.Q, x - xr)
        return dl_dx

    def compute_dl_dxdx(self, x, xr):
        dl_dxdx = 2.0 * self.Q
        return dl_dxdx

    def compute_dl_du(self, u):
        dl_du = 2.0*np.dot(self.R, u)
        return dl_du

    def compute_dl_dudu(self, u):
        dl_dudu = 2.0*self.R
        return dl_dudu

    def compute_dl_dudx(self, x, u):
        dl_dudx = np.zeros((self.m_inputs, self.n_states))
        return dl_dudx

    def forward_pass(self):
        prev_states = np.copy(self.states)
        prev_inputs = np.copy(self.inputs)
        prev_cost = self.cost()
        alpha = 1.0
        while (True):
            for i in range(0, self.prediction_horizon-1):
                self.inputs[:, i] = self.inputs[:, i] + alpha*np.reshape(self.k[i, :, :], (-1,)) + np.reshape(
                    np.dot(self.K[i, :, :], np.reshape(self.states[:, i] - prev_states[:, i], (-1, 1))), (-1,))
                if self.system.control_limited:
                    for j in range(self.m_inputs):
                        self.inputs[j, i] = min(max(
                            self.inputs[j, i], self.system.control_limit[j, 0]), self.system.control_limit[j, 1])
                self.states[:, i+1] = self.system.model_f(
                    self.states[:, i], self.inputs[:, i])
            cost = self.cost()
            if cost < prev_cost:
                # print('cost decreased after this pass. learning_rate: ', alpha)
                break
            elif alpha < 1e-4:
                self.converge = True
                # print(
                #     'learning_rate below threshold. Unable to reduce cost. learning_rate: ', alpha)
                break
            else:
                alpha /= 2.
                self.states = np.copy(prev_states)
                self.inputs = np.copy(prev_inputs)

    def backward_pass(self):
        npts = self.prediction_horizon
        self.k = np.zeros((npts-1, self.m_inputs, 1))
        self.K = np.zeros((npts-1, self.m_inputs, self.n_states))
        Vx = 2.0 * \
            np.dot(
                self.Qf, self.states[:, -1] - self.target_states[:, -1])
        Vxx = 2.0*self.Qf
        dl_dxdx = self.compute_dl_dxdx(None, None)
        dl_dudu = self.compute_dl_dudu(None)
        dl_dudx = self.compute_dl_dudx(None, None)
        for i in range(npts - 2, -1, -1):
            df_du = self.system.compute_df_du(
                self.states[:, i], self.inputs[:, i])
            df_dx = self.system.compute_df_dx(
                self.states[:, i], self.inputs[:, i])
            dl_dx = self.compute_dl_dx(
                self.states[:, i], self.target_states[:, i])
            dl_du = self.compute_dl_du(self.inputs[:, i])
            Qx = dl_dx + np.dot(df_dx.T, Vx)
            Qu = dl_du + np.dot(df_du.T, Vx)
            Vxx_augmented = Vxx + self.LM_parameter * np.eye(self.n_states)
            Qxx = dl_dxdx + np.dot(np.dot(df_dx.T, Vxx_augmented), df_dx)
            Quu = dl_dudu + np.dot(np.dot(df_du.T, Vxx_augmented), df_du)
            Qux = dl_dudx + np.dot(np.dot(df_du.T, Vxx_augmented), df_dx)
            Quu_inv = inv(Quu)
            k = -np.dot(Quu_inv, Qu)
            K = -np.dot(Quu_inv, Qux)
            self.k[i, :, :] = np.reshape(k, (-1, 1), order='F')
            self.K[i, :, :] = K
            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot(K.T, Qux)

    def __call__(self):
        for iter in range(self.maxIter):
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass()
        return self.states


if __name__ == '__main__':
    ntimesteps = 100
    target_states = np.zeros((4, ntimesteps))
    noisy_targets = np.zeros((4, ntimesteps))
    ref_vel = np.zeros(ntimesteps)
    dt = 0.2
    curv = 0.1
    a = 1.5
    v_max = 11
    for i in range(40, ntimesteps):
        if ref_vel[i - 1] > v_max:
            a = 0
        ref_vel[i] = ref_vel[i - 1] + a*dt
    for i in range(1, ntimesteps):
        target_states[0, i] = target_states[0, i-1] + \
            np.cos(target_states[3, i-1])*dt*ref_vel[i - 1]
        target_states[1, i] = target_states[1, i-1] + \
            np.sin(target_states[3, i-1])*dt*ref_vel[i - 1]
        target_states[2, i] = ref_vel[i]
        target_states[3, i] = target_states[3, i-1] + curv*dt
        noisy_targets[0, i] = target_states[0, i] + random.uniform(0, 5.0)
        noisy_targets[1, i] = target_states[1, i] + random.uniform(0, 5.0)
        noisy_targets[2, i] = target_states[2, i]
        noisy_targets[3, i] = target_states[3, i] + random.uniform(0, 1.0)

    car_system = Car()
    car_system.set_dt(dt)
    car_system.set_cost(
        np.diag([5.0, 5.0, 1000.0, 0.0]), np.diag([1000.0, 1000.0]))
    car_system.set_control_limit(np.array([[-1.5, 1.5], [-0.3, 0.3]]))
    myiLQR = iterative_LQR_quadratic_cost(
        car_system, noisy_targets, dt)

    for i in range(myiLQR.prediction_horizon-1):
        myiLQR.inputs[0, i] = (
            target_states[2, i + 1] - target_states[2, i])/dt
        myiLQR.inputs[1, i] = (
            target_states[3, i+1]-target_states[3, i])/dt

    start_time = timeit.default_timer()
    myiLQR()
    elapsed = timeit.default_timer() - start_time
    print("elapsed time: ", elapsed)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(myiLQR.target_states[0, :],
             myiLQR.target_states[1, :], '--r', label='Target', linewidth=2)
    plt.plot(myiLQR.states[0, :], myiLQR.states[1, :],
             '-+b', label='iLQR', linewidth=1.0)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: state vs. time.  ')
    plt.plot(myiLQR.states[2, :], '-b', linewidth=1.0, label='speed')
    plt.plot(ref_vel, '-r', linewidth=1.0, label='target speed')
    plt.ylabel('speed')
    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.title('iLQR: inputs vs. time.  ')
    plt.plot(myiLQR.inputs[0, :], '-b',
             linewidth=1.0, label='Acceleration')
    plt.plot(myiLQR.inputs[1, :], '-r',
             linewidth=1.0, label='turning rate')
    plt.ylabel('acceleration and turning rate input')
    plt.show()
