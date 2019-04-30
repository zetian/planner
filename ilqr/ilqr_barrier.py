import numpy as np
import scipy
import random
import pylab as pl
from numpy.linalg import inv
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from polynomial_curve1d import *


"iterative LQR with Quadratic cost"
class iterative_LQR_quadratic_cost:
    """
    iterative LQR can be used as a controller/trajectory optimizer.
    Reference:
    Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization
    https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
    Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems
    https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf
    Constrained iterative LQR for on-road autonomous driving motion planning
    
    cost function: x'Qx + u'Ru
    """

    def __init__(self, target_state_sequence, dt):
        """
        iLQR initilization
        """
        self.target_state_sequence = target_state_sequence
        self.prediction_horizon = self.target_state_sequence.shape[1]
        self.dt = dt
        self._curvature_limit = 0.3
        self.converge = False
        self.set_params()
        

    def set_params(self):
        self.n_states = 4
        self.m_inputs = 2
        self.Q  = np.diag([5.0, 5.0, 1000.0, 0.0])
        self.R  = np.diag([1000.0, 1000.0])
        self.obs_w = 100
        self.Qf = 100*self.Q #100*np.diag( [1.0, 1.0, 1.0] )
        self.maxIter = 30
        self.LM_parameter= 0.0
        self.state_sequence = np.zeros((self.n_states, self.prediction_horizon))
        self.input_sequence= np.zeros((self.m_inputs, self.prediction_horizon - 1))
        self.obstacle_weight = 0.1
        self.obs_list = []
    
    def set_obstacles(self, obs):
        self.obs_list = obs

    def cost(self):
        state_sequence_diff = self.state_sequence - self.target_state_sequence
        cost = 0.0
        for i in range(self.prediction_horizon-1):
            state = np.reshape(state_sequence_diff[:,i], (-1,1))
            control = np.reshape( self.input_sequence[:,i], (-1,1) )
            cost += np.dot(np.dot(state.T, self.Q), state) + np.dot(np.dot(control.T, self.R), control)
            for obs in self.obs_list:
                # if (self.state_sequence[0, i] - obs[0])**2 + (self.state_sequence[1, i] - obs[1])**2 - obs[2]**2 > 0:
                #     print(np.exp (-( (self.state_sequence[0, i] - obs[0])**2 + (self.state_sequence[1, i] - obs[1])**2 - obs[2]**2 )*self.obstacle_weight))
            
                cost += self.obs_w*np.exp(self.obstacle_weight*(obs[2]**2 - (self.state_sequence[0, i] - obs[0])**2 - (self.state_sequence[1, i] - obs[1])**2))

        state = np.reshape(state_sequence_diff[:,-1], (-1,1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        for obs in self.obs_list:
                # cost += self.obs_w*np.exp (-( (self.state_sequence[0, -1] - obs[0])**2 + (self.state_sequence[1, -1] - obs[1])**2 - obs[2]**2 )*self.obstacle_weight)
                cost += self.obs_w*np.exp(self.obstacle_weight*(obs[2]**2 - (self.state_sequence[0, -1] - obs[0])**2 - (self.state_sequence[1, -1] - obs[1])**2))
        return cost

    def model_f(self, x, u):
        assert (x.shape == (self.n_states,1) or x.shape == (self.n_states,) ), "state dimension inconsistent with setup."
        assert (u.shape == (self.m_inputs,1) or u.shape == (self.m_inputs,) ), "input dimension inconsistent with setup."
        theta = x[3]
        v = x[2]
        acc = u[0]
        theta_rate = u[1]
        x_next = np.reshape(x, (-1,1), order = 'F') + np.array( [ [v*np.cos(theta)], [v*np.sin(theta)], [acc], [theta_rate] ] )*self.dt
        return np.reshape(x_next, (-1,))

    def compute_df_dx(self, x, u):
        assert (x.shape[0] == self.n_states), "state dimension inconsistent with setup."
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

    def compute_dl_dx(self, x, xr):
        assert (x.shape[0] == self.n_states), "state dimension inconsistent with setup."
        dl_dx = 2.0*np.dot(self.Q, x - xr)
        # a = np.array([1, 3, 4, 5])
        # dl_dx_obs = np.array([[0.0], [0.0], [0.0], [0.0]])
        dl_dx_obs = np.zeros(4, dtype = 'float')
        for obs in self.obs_list:
            dl_dx_obs += np.array([-2.0*(x[0] - obs[0]), 
                                  -2.0*(x[1] - obs[1]),
                                  0.0,
                                  0.0])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))*self.obstacle_weight
        dl_dx += dl_dx_obs
        return dl_dx

    def compute_dl_dxdx(self, x, xr):
        # assert (x.shape == (self.n_states,1)), "state dimension inconsistent with setup."
        dl_dxdx = 2.0* self.Q
        dl_dxdx_obs = np.zeros(shape = (4, 4), dtype = 'float')
        for obs in self.obs_list:
            # dl_dxdx_obs += np.array([[-2*self.obstacle_weight + 4*self.obstacle_weight**2*(x[0] - obs[0])**2, 4*self.obstacle_weight**2*(x[1] - obs[1])*(x[0] - obs[0]), 0.0, 0.0],
            #                       [4*self.obstacle_weight**2*(x[1] - obs[1])*(x[0] - obs[0]), -2*self.obstacle_weight + 4*self.obstacle_weight**2*self.obstacle_weight*(x[1] - obs[1])**2, 0.0, 0.0],
            #                       [0.0, 0.0, 0.0, 0.0],
            #                       [0.0, 0.0, 0.0, 0.0]])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))
            dl_dxdx_obs += np.array([[4*(x[0] - obs[0])**2, 4*(x[1] - obs[1])*(x[0] - obs[0]), 0.0, 0.0],
                                  [4*(x[1] - obs[1])*(x[0] - obs[0]), 4*(x[1] - obs[1])**2, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0]])*np.exp(self.obstacle_weight*(obs[2]**2 - (x[0] - obs[0])**2 - (x[1] - obs[1])**2))*self.obstacle_weight**2
        
        dl_dxdx += dl_dxdx_obs
        return dl_dxdx

    def compute_dl_du(self, u):
        assert (u.shape[0] == self.m_inputs ), "input dimension inconsistent with setup."
        dl_du = 2.0*np.dot(self.R, u)
        return dl_du
    def compute_dl_dudu(self, u):
        # assert (u.shape == (self.m_inputs,1)), "input dimension inconsistent with setup."
        dl_dudu = 2.0*self.R
        return dl_dudu
    def compute_dl_dudx(self, x, u):
        # assert (x.shape == (self.n_states,1)), "state dimension inconsistent with setup."
        # assert (u.shape == (self.m_inputs,1)), "input dimension inconsistent with setup."
        dl_dudx = np.zeros((self.m_inputs, self.n_states))
        return dl_dudx

    def forward_pass(self, iteration):
        prev_state_sequence = np.copy(self.state_sequence)
        prev_input_sequence = np.copy(self.input_sequence)
        prev_cost = self.cost()
        print('prev_cost', prev_cost)
        alpha = 1.0
        while (True):
            for i in range(0, self.prediction_horizon-1):
                self.input_sequence[:, i] = self.input_sequence[:, i] + alpha*np.reshape(self.k_sequence[i,:,:], (-1,)) + np.reshape(
                                                  np.dot(self.K_sequence[i,:,:], np.reshape(self.state_sequence[:,i] - prev_state_sequence[:,i], (-1,1)) ), (-1,))
                self.input_sequence[0,i] = min(max(self.input_sequence[0,i], -1.5), 1.5)
                self.input_sequence[1,i] = min(max(self.input_sequence[1,i], -self._curvature_limit), self._curvature_limit)
                self.state_sequence[:,i+1] = self.model_f(self.state_sequence[:,i], self.input_sequence[:,i])
            cost = self.cost()
            if cost < prev_cost:
                print ('cost decreased after this pass. learning_rate: ', alpha)
                break
            elif alpha < 1e-4:
                self.converge = True
                print ('learning_rate below threshold. Unable to reduce cost. learning_rate: ', alpha)
                break
            else:
                alpha /= 2.
                self.state_sequence = np.copy(prev_state_sequence)
                self.input_sequence = np.copy(prev_input_sequence)

    def backward_pass(self):
        npts = self.prediction_horizon
        self.k_sequence = np.zeros( (npts-1, self.m_inputs, 1 )  )
        self.K_sequence = np.zeros( (npts-1, self.m_inputs, self.n_states )  )
        Vx  = 2.0*np.dot(self.Qf, self.state_sequence[:,-1] - self.target_state_sequence[:,-1])
        Vxx = 2.0*self.Qf
        
        # dl_dxdx = self.compute_dl_dxdx( self.state_sequence[:,-1], self.target_state_sequence[:,-1] )
        dl_dudu = self.compute_dl_dudu( None )
        dl_dudx = self.compute_dl_dudx( None, None )
        for i in range(npts-2, -1, -1):
            df_du = self.compute_df_du( self.state_sequence[:,i], self.input_sequence[:,i])
            df_dx = self.compute_df_dx( self.state_sequence[:,i], self.input_sequence[:,i] )
            dl_dx = self.compute_dl_dx( self.state_sequence[:,i], self.target_state_sequence[:,i] )
            dl_du = self.compute_dl_du( self.input_sequence[:,i] )
            dl_dxdx = self.compute_dl_dxdx( self.state_sequence[:,i], self.target_state_sequence[:,i] )
            Qx = dl_dx + np.dot( df_dx.T, Vx )
            Qu = dl_du + np.dot( df_du.T, Vx )
            Vxx_augmented = Vxx + self.LM_parameter *np.eye(self.n_states)
            
            Qxx = dl_dxdx + np.dot( np.dot( df_dx.T, Vxx_augmented ), df_dx )
            Quu = dl_dudu + np.dot( np.dot( df_du.T, Vxx_augmented ), df_du )
            Qux = dl_dudx + np.dot( np.dot( df_du.T, Vxx_augmented ), df_dx )

            Quu_inv = inv(Quu )
            k = -np.dot(Quu_inv, Qu)
            K = -np.dot(Quu_inv, Qux)
            self.k_sequence[ i,:,: ] = np.reshape(k, (-1,1), order = 'F')
            self.K_sequence[ i,:,: ] = K

            Vx = Qx + np.dot(K.T, Qu)
            Vxx = Qxx + np.dot( K.T, Qux)
        # print 'One backward pass completed.'
    def __call__(self, show_conv = False):
        "iterative LQR with quadratic cost function"
        assert (self.target_state_sequence is not None), "trajectory is not set yet."
        self.state_sequence[:,0] = self.target_state_sequence[:,0]
        for i in range(1, self.prediction_horizon):
            self.state_sequence[:,i] = self.model_f(self.state_sequence[:,i-1], self.input_sequence[:,i-1])
        if show_conv:
            pl.plot(self.target_state_sequence[0,:], self.target_state_sequence[1,:], 'r--+',linewidth=2.0, label = 'Target')
            pl.plot(self.state_sequence[0,:], self.state_sequence[1,:], '--',linewidth=1.5, label = 'iLQR')
            pl.grid('on')
            pl.axis('equal')
            pl.xlabel('x')
            pl.ylabel('y')
     
        for iteration in range(self.maxIter):
            if (self.converge):
                break
            self.backward_pass()
            self.forward_pass(iteration)
            # print (iteration)
            if show_conv:
                pl.plot(self.state_sequence[0,:], self.state_sequence[1,:], '-',linewidth=0.5, label = str(iteration) )
                pl.grid('on')
                pl.xlabel('x')
                pl.ylabel('y')
                pl.legend(framealpha=0.5)
                pl.pause(0.01)
        # pl.show()
        return self.state_sequence
        "iterative LQR with quadratic cost function ----------- end"


if __name__ == '__main__':

    obs_list = []
    obs_1 = [-10, 0, 10]
    obs_list.append(obs_1)


    ntimesteps = 300
    target_state_sequence = np.zeros((4, ntimesteps))
    noisy_target_sequence = np.zeros((4, ntimesteps))
    v_sequence = np.zeros(ntimesteps)
    dt = 0.2
    v = 1.0
    curv = 0.1

    a = 1.5
    v_max = 11

    v_sequence = np.ones(ntimesteps)*v_max

    poly_start = [0, 0, 0]
    poly_end = [v_max, 0]
    poly_time = 20
    quartic_poly = QuarticPolynomialCurve1d(poly_start, poly_end, poly_time)
    time_list = np.linspace(0, poly_time, int(poly_time/dt))
    pos = quartic_poly.Evaluate(0, time_list)
    vel = quartic_poly.Evaluate(1, time_list)
    accel = quartic_poly.Evaluate(2, time_list)

    # Use quartic_poly for speed profile
    v_sequence[0:vel.size] = vel



    for i in range(1, ntimesteps):
        target_state_sequence[0,i] = target_state_sequence[0,i-1] + np.cos(target_state_sequence[3,i-1])*dt*v_sequence[i - 1]
        target_state_sequence[1,i] = target_state_sequence[1,i-1] + np.sin(target_state_sequence[3,i-1])*dt*v_sequence[i - 1]
        target_state_sequence[2,i] = v_sequence[i]
        target_state_sequence[3,i] = target_state_sequence[3,i-1] + curv*dt
        noisy_target_sequence[0,i] = target_state_sequence[0, i] + random.uniform(0, 5.0)
        noisy_target_sequence[1,i] = target_state_sequence[1, i] + random.uniform(0, 5.0)
        noisy_target_sequence[2,i] = target_state_sequence[2, i]
        noisy_target_sequence[3,i] = target_state_sequence[3, i] + random.uniform(0, 1.0)
        
    test_seq = np.zeros((4, ntimesteps))
    time = np.linspace(0, np.pi, 1000)
    test_x = []
    test_y = []

    dx = 0.1
    test_x = [0]
    test_y = [0]
    for i in range(1, 300):
        test_x.append(dx + test_x[-1])
        test_y.append(0)
    test_x.pop(0)
    test_y.pop(0)
    test_x.reverse()

    d_theta = 0.01
    R = 10
    for i in range(1, ntimesteps):
        test_x.append(-R + R*np.cos(i*d_theta))
        test_y.append(R*np.sin(i*d_theta))

    funny = np.zeros((4, len(test_x)))
    for i in range(0, len(test_x)):
        funny[0, i] = test_x[i]
        funny[1, i] = test_y[i]
        funny[2, i] = -1
        funny[3, i] = 0

    # plt.plot(test_x,test_y)
    # plt.show()

    # myiLQR = iterative_LQR_quadratic_cost(target_state_sequence, dt)
    myiLQR = iterative_LQR_quadratic_cost(funny, 0.1)
    myiLQR.set_obstacles(obs_list)
    # for i in range(myiLQR.prediction_horizon-1):
    #     myiLQR.input_sequence[0,i] = (target_state_sequence[2, i + 1] - target_state_sequence[2, i])/dt
    #     myiLQR.input_sequence[1,i] = (target_state_sequence[3,i+1]-target_state_sequence[3,i])/dt
    for i in range(myiLQR.prediction_horizon-1):
        myiLQR.input_sequence[0,i] = -1
        myiLQR.input_sequence[1,i] = 0

    # init_sequence = np.zeros((4, ntimesteps))
    # for i in range(1, myiLQR.prediction_horizon):
    #     init_sequence[:,i] = myiLQR.model_f(init_sequence[:,i-1], myiLQR.input_sequence[:,i-1])

    # plt.figure()
    # plt.plot(init_sequence[0,:], init_sequence[1,:], '--',linewidth=1.5, label = 'init state')
    # plt.show()
    
    myiLQR(show_conv = False)
    

    plt.figure(figsize=(8*1.1, 6*1.1))
    ax = plt.gca()
    plt.suptitle('iLQR: 2D, x and y.  ')
    plt.axis('equal')
    plt.plot(myiLQR.target_state_sequence[0,:], myiLQR.target_state_sequence[1,:], '--r', label = 'Target', linewidth=2)
    plt.plot(myiLQR.state_sequence[0,:], myiLQR.state_sequence[1,:], '-+b', label = 'iLQR', linewidth=1.0)
    for obs in obs_list:
        print(obs[0], obs[1])
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='r')
        ax.add_artist(circle)
    plt.grid('on')
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.legend(fancybox=True, framealpha=0.2)

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.suptitle('iLQR: state vs. time.  ')

    plt.plot(myiLQR.state_sequence[2,:], '-b', linewidth=1.0, label='speed')
    # pl.plot(myiLQR.state_sequence[3,:], '-r', linewidth=1.0, label='yaw')
    plt.plot(v_sequence, '-r', linewidth=1.0, label='target speed')
    
    plt.grid('on')
    # pl.xlabel('x (meters)')
    plt.ylabel('speed')
    plt.legend(fancybox=True, framealpha=0.2)
    plt.tight_layout()

    plt.figure(figsize=(8*1.1, 6*1.1))
    plt.suptitle('iLQR: inputs vs. time.  ')

    plt.plot(myiLQR.input_sequence[0,:], '-b', linewidth=1.0, label='Acceleration')
    plt.plot(myiLQR.input_sequence[1,:], '-r', linewidth=1.0, label='turning rate')
    plt.grid('on')
    # pl.xlabel('x (meters)')
    plt.ylabel('acceleration and turning rate input')
    plt.legend(fancybox=True, framealpha=0.2)
    plt.tight_layout()


    plt.show()