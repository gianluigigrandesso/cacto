import gym
import math
import random
import numpy as np
import tensorflow as tf
import pinocchio as pin
from gym.spaces import Box
from utils import *

class Env:
    def __init__(self, conf):
        '''    
        :input conf :                           (Configuration file)

            :param robot :                      (RobotWrapper instance) 
            :param simu :                       (RobotSimulator instance)
            :param x_init_min :                 (float array) State lower bound initial configuration array
            :param x_init_max :                 (float array) State upper bound initial configuration array
            :param x_min :                      (float array) State lower bound vector
            :param x_max :                      (float array) State upper bound vector
            :param u_min :                      (float array) Action lower bound array
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param dt :                         (float) Timestep
            :param end_effector_frame_id :      (str) Name of EE-frame

            # Cost function parameters
            :param TARGET_STATE :               (float array) Target position
            :param cost_funct_param             (float array) Cost function scale and offset factors
            :param soft_max_param :             (float array) Soft parameters array
            :param obs_param :                  (float array) Obtacle parameters array
    '''
        
        self.conf = conf

        self.observation_space = Box(self.conf.x_min, self.conf.x_max, shape=(self.conf.nb_state,), dtype=np.float32)
        self.action_space = Box(self.conf.u_min, self.conf.u_max, shape=(self.conf.nb_action,), dtype=np.float32) 

        self.nq = self.conf.robot.nq
        self.nv = self.conf.robot.nv
        self.nx = self.nq + self.nv
        self.nu = self.conf.robot.na

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

    def reset(self, options=None):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state) 

        time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 

        state[-1] = self.conf.dt*round(time/self.conf.dt)

        return time, state

    def step(self, weights, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward(weights, state, action)

        return (state_next, reward)

    def simulate(self, state, action):
        ''' Simulate dynamics '''
        state_next = np.zeros(self.nx+1)

        # Simulate control action
        self.conf.simu.simulate(np.copy(state[:-1]), action, self.conf.dt, 1) #Explicit Euler

        # Return next state
        state_next[:self.nq], state_next[self.nq:self.nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt
        
        return state_next
    
    def derivative(self, state, action):
        ''' Compute the derivative '''
        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = state[:self.nq]
        v_init = state[self.nq:self.nx]

        # Dynamics gradient w.r.t control (1st order euler)
        pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, np.copy(q_init), np.copy(v_init), action)       

        Fu = np.zeros((self.nx+1, self.nu))
        Fu[self.nv:-1, :] = self.conf.robot.data.Minv
        Fu[:self.nx, :] *= self.conf.dt

        if self.conf.NORMALIZE_INPUTS:
            Fu[:-1] *= (1/self.conf.state_norm_arr[:-1,None])  

        return Fu
    
    def simulate_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        state_next = np.zeros((self.conf.BATCH_SIZE, self.conf.nb_state)) 

        for sample_indx in range(self.conf.BATCH_SIZE):
            # Simulate control action
            state_next[sample_indx] = self.simulate(state[sample_indx],action[sample_indx])             

        return tf.convert_to_tensor(state_next, dtype=tf.float32)
        
    def derivative_batch(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        Fu  = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state,self.conf.nb_action))      

        for sample_indx in range(self.conf.BATCH_SIZE):
            # Dynamics gradient w.r.t control (1st order euler)
            Fu[sample_indx,:,:] = self.derivative(state[sample_indx],action[sample_indx])     

        return tf.convert_to_tensor(Fu, dtype=tf.float32)
    
    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        q = state[:self.nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p
    



class Manipulator(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):
    
        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = [self.get_end_effector_position(state)[i] for i in range(2)] 

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        # Term penalizing the FINAL joint velocity
        if np.linalg.norm(state[-1] - self.conf.dt*self.conf.NSTEPS) < 1e-3:
            vel_cost = state[self.nq:self.nx].dot(state[self.nq:self.nx])
        else:    
            vel_cost = 0

        if action is not None:
            u_cost = action.dot(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset) #- weights[2]*vel_cost 

        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.zeros(self.conf.BATCH_SIZE)

        for sample_indx in range(self.conf.BATCH_SIZE):
            # Compute not-action related reward
            partial_reward[sample_indx] = self.reward(weights[sample_indx,:], state[sample_indx,:])

        # Redefine action-related cost in tensorflow version
        u_cost = action[:,0]**2 + action[:,1]**2 + action[:,2]**2 
    
        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])
    


class DoubleIntegrator(Env):
    '''
    :param cost_function_parameters :
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):

        self.conf = conf

        super().__init__(conf)

        # Rename reward parameters
        self.offset = self.conf.cost_funct_param[0]
        self.scale = self.conf.cost_funct_param[1]

        self.alpha = self.conf.soft_max_param[0]
        self.alpha2 = self.conf.soft_max_param[1]

        self.XC1 = self.conf.obs_param[0]
        self.YC1 = self.conf.obs_param[1]
        self.XC2 = self.conf.obs_param[2]
        self.YC2 = self.conf.obs_param[3]
        self.XC3 = self.conf.obs_param[4]
        self.YC3 = self.conf.obs_param[5]
        
        self.A1 = self.conf.obs_param[6]
        self.B1 = self.conf.obs_param[7]
        self.A2 = self.conf.obs_param[8]
        self.B2 = self.conf.obs_param[9]
        self.A3 = self.conf.obs_param[10]
        self.B3 = self.conf.obs_param[11]

        self.TARGET_STATE = self.conf.TARGET_STATE
    
    def reward(self, weights, state, action=None):
        ''' Compute reward '''
        # End-effector coordinates
        x_ee, y_ee = [self.get_end_effector_position(state)[i] for i in range(2)] 

        # Penalties for the ellipses representing the obstacle
        ell1_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_cost = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_rew = np.math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        if action is not None:
            u_cost = action.dot(action)
        else:
            u_cost = 0

        dist_cost = (x_ee-self.TARGET_STATE[0])**2 + (y_ee-self.TARGET_STATE[1])**2

        r = self.scale*(- weights[0]*dist_cost + weights[1]*peak_rew - weights[3]*ell1_cost - weights[4]*ell2_cost - weights[5]*ell3_cost - weights[6]*u_cost + self.offset)
        
        return r
    
    def reward_batch(self, weights, state, action):
        ''' Compute reward using tensors. Batch-wise computation '''
        partial_reward = np.zeros(self.conf.BATCH_SIZE)

        for sample_indx in range(self.conf.BATCH_SIZE):
            # Compute not-action related reward
            partial_reward[sample_indx] = self.reward(weights[sample_indx,:], state[sample_indx,:])

        # Redefine action-related cost in tensorflow version
        u_cost = action[:,0]**2+action[:,1]**2 
    
        r = self.scale*(- weights[:,6]*u_cost) + tf.convert_to_tensor(partial_reward, dtype=tf.float32)

        return tf.reshape(r, [r.shape[0], 1])