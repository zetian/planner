import numpy as np
from matplotlib import pyplot as plt

class System:
    def __init__(self, state_size, control_size):
        self.state_size = state_size
        self.control_size = control_size
    def set_cost(self, Q, R):
        # one step cost = x.T * Q * x + u.T * R * u
        self.Q = Q
        self.R = R

    def set_final_cost(self, Q_f):
        self.Q_f = Q_f
    def calculate_cost(self, x, u):
        return 0.5*((x-self.goal).T.dot(self.Q).dot(x-self.goal) + u.T.dot(self.R).dot(u))
    def calculate_final_cost(self, x):
        return 0.5*(x-self.goal).T.dot(self.Q_f).dot(x-self.goal)
    def set_goal(self, x_goal):
        self.goal = x_goal

class Car(System):
    def __init__(self):
        super().__init__(4, 2)
        self.dt = 0.05

    def set_dt(self, dt):
        self.dt = dt
    
    def model_f(self, x, u):
        assert (x.shape == (self.state_size,1) or x.shape == (self.state_size,) ), "state dimension inconsistent with setup."
        assert (u.shape == (self.control_size,1) or u.shape == (self.control_size,) ), "input dimension inconsistent with setup."
        theta = x[3]
        v = x[2]
        acc = u[0]
        theta_rate = u[1]
        x_next = np.reshape(x, (-1,1), order = 'F') + np.array( [ [v*np.cos(theta)], [v*np.sin(theta)], [acc], [theta_rate] ] )*self.dt
        return np.reshape(x_next, (-1,))

    def compute_df_dx(self, x, u):
        assert (x.shape[0] == self.state_size), "state dimension inconsistent with setup."
        theta = x[3]
        v = x[2]
        acc = u[0]
        theta_rate = u[1]
        # df_dx = np.array([ [1.0, 0.0,  -np.sin(theta)*v*self.dt],
        #                    [0.0, 1.0,   np.cos(theta)*v*self.dt],
        #                    [0.0, 0.0,  1.0] 
        #                 ])
        df_dx = np.array([ [1.0, 0.0, np.cos(theta)*self.dt, -np.sin(theta)*v*self.dt],
                           [0.0, 1.0, np.sin(theta)*self.dt,  np.cos(theta)*v*self.dt],
                           [0.0, 0.0,  1.0, 0.0],
                           [0.0, 0.0,  0.0, 1.0] 
                        ])
        return df_dx

    def compute_df_du(self, x, u):
        # assert (u.shape == (self.m_inputs,1)), "state dimension inconsistent with setup."
        # theta = x[2]
        # v = u[0]
        # curvature = u[1]
        # df_du = np.array([ [np.cos(theta), 0.0],
        #                    [np.sin(theta), 0.0],
        #                    [curvature, v ]])*self.dt
        df_du = np.array([ [0.0, 0.0],
                           [0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0]])*self.dt
        return df_du

    def draw_trajectories(self, x_trajectories):
        ax = plt.subplot(111)
        circle1 = plt.Circle((1, 1), 0.5, color=(0, 0.8, 0.8))
        circle2 = plt.Circle((2, 2), 1, color=(0, 0.8, 0.8))
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        for i in range(0, x_trajectories.shape[1]-1, 5):
            circle_car = plt.Circle((x_trajectories[0, i], x_trajectories[1, i]), 0.1, facecolor='none')
            ax.add_patch(circle_car)
            ax.arrow(x_trajectories[0, i], x_trajectories[1, i], 0.1*np.sin(x_trajectories[2, i]), 0.1 * np.cos(x_trajectories[2, i]), head_width=0.05, head_length=0.1, fc='k', ec='k')
        ax.set_aspect("equal")
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        plt.show()