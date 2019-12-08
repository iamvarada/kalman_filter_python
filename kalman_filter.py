#! usr/bin/python3

# Basic implementation of Kalman filter in python
# Developer : Krishna Varadarajan

import numpy as np
import numpy.linalg as lin

class BasicKalmanFilter :   
    """ Python class for basic kalman filter implementation"""

    dt_ = 0.0
    t_iniital_ = 0.0
    t_current_ = 0.0
    last_predict_step_time_ = 0.0
    last_udpate_step_time_ = 0.0
    filter_initialized_ = False
    num_states_ = 0
    num_outputs_ = 0
    X_ = np.zeros((num_states_, 1))
    P_ = np.zeros((num_states_, num_states_))
    A_ = np.zeros((num_states_, num_states_))
    C_ = np.zeros((num_outputs_, num_states_))

    def __init__(self, A, C) :
        """ Class constructor (default) """
        """ \param[in] A : system numpy matrix; dims: (num_states, num_states) """
        """ \param[in] C : output numpy matrix; dims: (num_outputs, num_states)"""

        self.A_ = A
        self.C_ = C
        self.num_states_ = A.shape[0]
        self.num_outputs_ = C.shape[0]


    def initializer_filter (self, X_init, P_init, t_init, dt) : 
        """ Initialize the filter zero initial states """
        """ \param[in] X_init : initial states of the filter; dims: (num_states, 1) """
        """ \param[in] P_init : initial estimation error covariance of the filter; dims: (num_states, num_states) """
        """ \param[in] t_init : start time of the filter """
        """ \param[dt] dt : time-step of the filter """

        self.X_ = X_init
        self.filter_initialized_ = True
        self.dt_ = dt
        self.P_ = P_init
        self.t_iniital_ = t_init
        self.t_current_ = self.t_iniital_
        self.last_udpate_step_time_ = self.t_iniital_
        self.last_predict_step_time_ = self.t_iniital_
        return

    def predict_filter_step (self, Q) :
        """ Predict step for the filter when measurement is not received """
        """ \param[in] Q : process noise covariance (flexibility to change it at every prediction step) """

        if (self.filter_initialized_ == False) :
            print ("\nFilter not initialized")
            return 
        self.X_ =  np.dot(self.A_, self.X_)
        self.P_ = np.dot(self.A_, np.dot(self.P_, self.A_.T)) + Q

        self.t_current_ = self.t_current_ + self.dt_
        self.last_predict_step_time_ = self.t_current_
        return

    def update_filter_step(self, Y, R) :
        """ Update step for the filter when a measurement is received """
        """ \param[in] Y : received measurment """
        """ \param[in] R : measurement noise covariance (flexibility to change it at every prediction step) """

        if (self.filter_initialized_ == False) :
            print ("\nFilter not initialized")
            return
        temp = np.dot(self.C_, np.dot(self.P_, self.C_.T)) + R
        temp_inv = lin.inv(temp)
        K_gain = np.dot(self.P_, np.dot(self.C_.T, temp_inv))
        residual = Y - np.dot(self.C_, self.X_)
        self.X_ = self.X_ + np.dot(K_gain, residual)
        self.P_ = np.identity(self.num_states_, self.num_states_) - np.dot(K_gain, self.C_)

        self.t_current_ = self.t_current_ + self.dt_
        self.last_udpate_step_time_ = self.t_current_
        return K_gain
    
    def get_lastest_estimated_state(self) : 
        return self.X_
  
    def get_current_time(self) :
        return self.t_current_
