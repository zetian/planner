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
