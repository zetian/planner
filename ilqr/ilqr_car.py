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
        self.n_states = 5
        self.m_inputs = 2
        self.Q  = np.diag([5.0, 5.0, 1000.0, 1000, 0.0])
        self.R  = np.diag([1000.0, 1000.0])
        self.Qf = 100*self.Q #100*np.diag( [1.0, 1.0, 1.0] )
        self.maxIter = 30
        self.LM_parameter= 0.0
        self.state_sequence = np.zeros((self.n_states, self.prediction_horizon))
        self.input_sequence= np.zeros((self.m_inputs, self.prediction_horizon - 1))

    def cost(self):
        state_sequence_diff = self.state_sequence - self.target_state_sequence
        cost = 0.0
        for i in range(self.prediction_horizon-1):
            state = np.reshape(state_sequence_diff[:,i], (-1,1))
            control = np.reshape( self.input_sequence[:,i], (-1,1) )
            cost += np.dot(np.dot(state.T, self.Q), state) + np.dot(np.dot(control.T, self.R), control)
        state = np.reshape(state_sequence_diff[:,-1], (-1,1))
        cost += np.dot(np.dot(state.T, self.Qf), state)
        return cost

    def model_f(self, x, u):
        assert (x.shape == (self.n_states,1) or x.shape == (self.n_states,) ), "state dimension inconsistent with setup."
        assert (u.shape == (self.m_inputs,1) or u.shape == (self.m_inputs,) ), "input dimension inconsistent with setup."
        theta = x[3]
        v = x[2]
        acc = x[4]
        jerk = u[0]
        theta_rate = u[1]
        x_next = np.reshape(x, (-1,1), order = 'F') + np.array( [ [v*np.cos(theta)], [v*np.sin(theta)], [acc], [theta_rate], [jerk] ] )*self.dt
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
        df_dx = np.array([ [1.0, 0.0, np.cos(theta)*self.dt, -np.sin(theta)*v*self.dt, 0.0],
                           [0.0, 1.0, np.sin(theta)*self.dt,  np.cos(theta)*v*self.dt, 0.0],
                           [0.0, 0.0,  1.0, 0.0, self.dt],
                           [0.0, 0.0,  0.0, 1.0, 0.0],
                           [0.0, 0.0,  0.0, 0.0, 1.0]
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
                           [0.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 0.0]])*self.dt
        return df_du

    def compute_dl_dx(self, x, xr):
        assert (x.shape[0] == self.n_states), "state dimension inconsistent with setup."
        dl_dx = 2.0*np.dot(self.Q, x - xr)
        return dl_dx
    def compute_dl_dxdx(self, x, xr):
        # assert (x.shape == (self.n_states,1)), "state dimension inconsistent with setup."
        dl_dxdx = 2.0* self.Q
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
        
        dl_dxdx = self.compute_dl_dxdx( None, None )
        dl_dudu = self.compute_dl_dudu( None )
        dl_dudx = self.compute_dl_dudx( None, None )
        for i in range(npts-2, -1, -1):
            df_du = self.compute_df_du( self.state_sequence[:,i], self.input_sequence[:,i])
            df_dx = self.compute_df_dx( self.state_sequence[:,i], self.input_sequence[:,i] )
            dl_dx = self.compute_dl_dx( self.state_sequence[:,i], self.target_state_sequence[:,i] )
            dl_du = self.compute_dl_du( self.input_sequence[:,i] )

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

    
    ntimesteps = 100
    target_state_sequence = np.zeros((5, ntimesteps))
    noisy_target_sequence = np.zeros((5, ntimesteps))
    v_sequence = np.zeros(ntimesteps)
    dt = 0.2
    v = 1.0
    curv = 0.1

    a = 1.5
    v_max = 11

    # v_sequence = np.ones(ntimesteps)*v_max

    poly_start = [0, 0, 0]
    poly_end = [v_max, 0]
    poly_time = 20
    quartic_poly = QuarticPolynomialCurve1d(poly_start, poly_end, poly_time)
    time_list = np.linspace(0, poly_time, int(poly_time/dt))
    pos = quartic_poly.Evaluate(0, time_list)
    vel = quartic_poly.Evaluate(1, time_list)
    accel = quartic_poly.Evaluate(2, time_list)

    # Use quartic_poly for speed profile
    # v_sequence[0:vel.size] = vel
    # pl.figure()
    # pl.plot(pos, '-r', linewidth=1.0, label='target speed')
    # pl.show()

    for i in range(40, ntimesteps):
        if v_sequence[i - 1] > v_max:
            a = 0
        v_sequence[i] = v_sequence[i - 1] + a*dt

    # plt.figure()
    # plt.plot(v_sequence)
    # plt.show()

    for i in range(1, ntimesteps):
        target_state_sequence[0,i] = target_state_sequence[0,i-1] + np.cos(target_state_sequence[3,i-1])*dt*v_sequence[i - 1]
        target_state_sequence[1,i] = target_state_sequence[1,i-1] + np.sin(target_state_sequence[3,i-1])*dt*v_sequence[i - 1]
        target_state_sequence[2,i] = v_sequence[i]
        target_state_sequence[3,i] = target_state_sequence[3,i-1] + curv*dt
        target_state_sequence[4,i] = 0.0
        noisy_target_sequence[0,i] = target_state_sequence[0, i] + random.uniform(0, 5.0)
        noisy_target_sequence[1,i] = target_state_sequence[1, i] + random.uniform(0, 5.0)
        noisy_target_sequence[2,i] = target_state_sequence[2, i]
        noisy_target_sequence[3,i] = target_state_sequence[3, i] + random.uniform(0, 1.0)
        noisy_target_sequence[4,i] = target_state_sequence[4, i] + random.uniform(0, 1.0)
        

    # myiLQR = iterative_LQR_quadratic_cost(target_state_sequence, dt)
    myiLQR = iterative_LQR_quadratic_cost(noisy_target_sequence, dt)
    for i in range(myiLQR.prediction_horizon-1):
        # myiLQR.input_sequence[0,i] = (target_state_sequence[2, i + 1] - target_state_sequence[2, i])/dt
        myiLQR.input_sequence[0,i] = 0.0
        myiLQR.input_sequence[1,i] = (target_state_sequence[3,i+1]-target_state_sequence[3,i])/dt

    # init_sequence = np.zeros((4, ntimesteps))
    # for i in range(1, myiLQR.prediction_horizon):
    #     init_sequence[:,i] = myiLQR.model_f(init_sequence[:,i-1], myiLQR.input_sequence[:,i-1])

    # plt.figure()
    # plt.plot(init_sequence[0,:], init_sequence[1,:], '--',linewidth=1.5, label = 'init state')
    # plt.show()
    
    myiLQR(show_conv = False)

    pl.figure(figsize=(8*1.1, 6*1.1))
    pl.suptitle('iLQR: 2D, x and y.  ')
    pl.axis('equal')
    pl.plot(myiLQR.target_state_sequence[0,:], myiLQR.target_state_sequence[1,:], '--r', label = 'Target', linewidth=2)
    pl.plot(myiLQR.state_sequence[0,:], myiLQR.state_sequence[1,:], '-+b', label = 'iLQR', linewidth=1.0)
    pl.grid('on')
    pl.xlabel('x (meters)')
    pl.ylabel('y (meters)')
    pl.legend(fancybox=True, framealpha=0.2)



    pl.figure(figsize=(8*1.1, 6*1.1))
    pl.suptitle('iLQR: state vs. time.  ')

    pl.plot(myiLQR.state_sequence[2,:], '-b', linewidth=1.0, label='speed')
    pl.plot(myiLQR.state_sequence[4,:], '-g', linewidth=1.0, label='acceleration')
    pl.plot(v_sequence, '-r', linewidth=1.0, label='target speed')
    
    pl.grid('on')
    # pl.xlabel('x (meters)')
    pl.ylabel('speed')
    pl.legend(fancybox=True, framealpha=0.2)
    pl.tight_layout()

    pl.figure(figsize=(8*1.1, 6*1.1))
    pl.suptitle('iLQR: inputs vs. time.  ')

    pl.plot(myiLQR.input_sequence[0,:], '-b', linewidth=1.0, label='Jerk')
    pl.plot(myiLQR.input_sequence[1,:], '-r', linewidth=1.0, label='Turning rate')
    pl.grid('on')
    # pl.xlabel('x (meters)')
    pl.ylabel('acceleration and turning rate input')
    pl.legend(fancybox=True, framealpha=0.2)
    pl.tight_layout()


    pl.show()