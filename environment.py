import gym
import math
import random
import numpy as np
import tensorflow as tf
import pinocchio as pin
from gym.spaces import Box
from utils import *

class Manipulator(gym.Env):
    '''
    :param robot :                  (RobotWrapper instance) 
    :param simu :                   (RobotSimulator instance)
    :param x_init_min :             (float array) State lower bound initial configuration array
    :param x_init_max :             (float array) State upper bound initial configuration array
    :param x_min :                  (float array) State lower bound vector
    :param x_max :                  (float array) State upper bound vector
    :param u_min :                  (float array) Action lower bound array
    :param u_max :                  (float array) Action upper bound array
    :param nb_state :               (int) State size (robot state size + 1)
    :param nb_action :              (int) Action size (robot action size)
    :param dt :                     (float) Timestep
    :param self.TARGET_STATE :           (float array) Target position
    :param soft_max_param :         (float array) Soft parameters array
    :param obs_param :              (float array) Obtacle parameters array
    :param weight :                 (float array) Weights array
    :param end_effector_frame_id :  (str)
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):
    
        self.conf = conf

        self.observation_space = Box(self.conf.x_min, self.conf.x_max, shape=(self.conf.nb_state,), dtype=np.float32)
        self.action_space = Box(self.conf.u_min, self.conf.u_max, shape=(self.conf.nb_action,), dtype=np.float32)  

        self.window = None
        self.clock = None

        #rename reward parameters
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

        self.w_d = self.conf.weight[0]
        self.w_u = self.conf.weight[1]
        self.w_peak = self.conf.weight[2]
        self.w_ob1 = self.conf.weight[3]
        self.w_ob2 = self.conf.weight[3]
        self.w_ob3 = self.conf.weight[3]
        self.w_v = self.conf.weight[4]

        self.TARGET_STATE = self.conf.TARGET_STATE

    def reset(self, options=None):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state) 

        time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        state[-1] = self.conf.dt*round(time/self.conf.dt)

        return time, state

    def step(self, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward(state, action)

        return (state_next, reward)
    
    def simulate(self, state, action):
        ''' Simulate dynamics '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq + nv

        state_next = np.zeros(nx+1)

        # Simulate control action
        self.conf.simu.simulate(np.copy(state), action, self.conf.dt, 1)
       
        # Return next state
        state_next[:nq], state_next[nq:nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt

        return state_next
    
    def reward(self, state, action=None):
        ''' Compute reward '''
        nv = self.conf.robot.nv
        nq = self.conf.robot.nq
        nx = nv + nq

        # End-effector coordinates 
        x_ee = self.conf.x_base + self.conf.l*(math.cos(state[0]) + math.cos(state[0]+state[1]) + math.cos(state[0]+state[1]+state[2]))
        y_ee = self.conf.y_base + self.conf.l*(math.sin(state[0]) + math.sin(state[0]+state[1]) + math.sin(state[0]+state[1]+state[2]))  

        # Penalties for the ellipses representing the obstacle
        ell1_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_reward = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        # Term penalizing the FINAL joint velocity
        if state[-1] == self.conf.dt*self.conf.NSTEPS:
            vel_joint = state[nq:nx].dot(state[nq:nx]) - 10000/self.w_v
        else:    
            vel_joint = 0

        r = (self.w_d*(-(x_ee-self.TARGET_STATE[0])**2 -(y_ee-self.TARGET_STATE[1])**2) + self.w_peak*peak_reward - self.w_v*vel_joint - self.w_ob1*ell1_pen - self.w_ob2*ell2_pen - self.w_ob3*ell3_pen - self.w_u*(action.dot(action)))/100 
        
        return r
    
    def simulate_and_derivative_tf(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        state_next = np.zeros((self.conf.BATCH_SIZE, self.conf.nb_state))
        Fu = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state,self.conf.nb_action))      

        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq+nv

        nu = self.conf.nb_action

        for sample_indx in range(self.conf.BATCH_SIZE):
            state_np = np.copy(state[sample_indx]) 
            
            # Create robot model in Pinocchio with q_init as initial configuration
            q_init = state_np[:nq]
            v_init = state_np[nq:nx]
            
            # Dynamics gradient w.r.t control (1st order euler)
            pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, q_init, v_init, action[sample_indx])       

            Fu_sample = np.zeros((nx+1, nu))
            Fu_sample[nv:-1, :] = self.conf.robot.data.Minv
            Fu_sample[:nx, :] *= self.conf.dt

            Fu[sample_indx] = Fu_sample  

            # Simulate control action
            self.conf.simu.simulate(np.concatenate((q_init, v_init)), action[sample_indx], self.conf.dt, 1)

            state_next[sample_indx,:self.conf.robot.nq] = np.copy(self.conf.simu.q)
            state_next[sample_indx,self.conf.robot.nq:self.conf.robot.nv+self.conf.robot.nq] = np.copy(self.conf.simu.v)
            state_next[sample_indx,-1] = state_np[-1] + self.conf.dt

        return tf.convert_to_tensor(state_next, dtype=tf.float32), Fu 
    
    def reward_tf(self, state, action, BATCH_SIZE, last_ts):
        ''' Compute reward using tensors. Batch-wise computation '''    
        x_ee = self.conf.x_base + self.conf.l*(tf.math.cos(state[:,0]) + tf.math.cos(state[:,0]+state[:,1]) + tf.math.cos(state[:,0]+state[:,1]+state[:,2]))
        y_ee = self.conf.y_base + self.conf.l*(tf.math.sin(state[:,0]) + tf.math.sin(state[:,0]+state[:,1]) + tf.math.sin(state[:,0]+state[:,1]+state[:,2]))

        ell1_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC1)**2)/((self.A1/2)**2) + ((y_ee[:]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC2)**2)/((self.A2/2)**2) + ((y_ee[:]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC3)**2)/((self.A3/2)**2) + ((y_ee[:]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha
    
        peak_reward = tf.math.log(tf.math.exp(self.alpha2*-(tf.math.sqrt((x_ee[:]-self.TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-self.TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/self.alpha2
    
        vel_joint_list = []
        for i in range(BATCH_SIZE):
            if last_ts[i][0] == 1.0:
                vel_joint_list.append(state[i,3]**2 + state[i,4]**2 + state[i,5]**2 - 10000/self.w_v)
            else:    
                vel_joint_list.append(0)
        vel_joint = tf.cast(tf.stack(vel_joint_list),tf.float32)
    
        r = (self.w_d*(-(x_ee[:]-self.TARGET_STATE[0])**2 -(y_ee[:]-self.TARGET_STATE[1])**2) + self.w_peak*peak_reward -self.w_v*vel_joint - self.w_ob1*ell1_pen - self.w_ob2*ell2_pen - self.w_ob3*ell3_pen - self.w_u*(action[:,0]**2+action[:,1]**2+action[:,2]**2))/100 
        r = tf.reshape(r, [r.shape[0], 1])

        return r 

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        q = state[:nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p
    



class DoubleIntegrator(gym.Env):
    '''
    :param robot :                  (RobotWrapper instance) 
    :param simu :                   (RobotSimulator instance)
    :param x_init_min :             (float array) State lower bound initial configuration array
    :param x_init_max :             (float array) State upper bound initial configuration array
    :param x_min :                  (float array) State lower bound vector
    :param x_max :                  (float array) State upper bound vector
    :param u_min :                  (float array) Action lower bound array
    :param u_max :                  (float array) Action upper bound array
    :param nb_state :               (int) State size (robot state size + 1)
    :param nb_action :              (int) Action size (robot action size)
    :param dt :                     (float) Timestep
    :param self.TARGET_STATE :           (float array) Target position
    :param soft_max_param :         (float array) Soft parameters array
    :param obs_param :              (float array) Obtacle parameters array
    :param weight :                 (float array) Weights array
    :param end_effector_frame_id :  (str)
    '''

    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, conf):

        self.conf = conf

        self.observation_space = Box(self.conf.x_min, self.conf.x_max, shape=(self.conf.nb_state,), dtype=np.float32)
        self.action_space = Box(self.conf.u_min, self.conf.u_max, shape=(self.conf.nb_action,), dtype=np.float32) 

        self.window = None
        self.clock = None

        #rename reward parameters
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

        self.w_d = self.conf.weight[0]
        self.w_u = self.conf.weight[1]
        self.w_peak = self.conf.weight[2]
        self.w_ob1 = self.conf.weight[3]
        self.w_ob2 = self.conf.weight[3]
        self.w_ob3 = self.conf.weight[3]
        self.w_v = self.conf.weight[4]

        self.TARGET_STATE = self.conf.TARGET_STATE
    
    def reset(self, options=None):
        ''' Choose initial state uniformly at random '''
        state = np.zeros(self.conf.nb_state)    

        time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1]) 
        for i in range(self.conf.nb_state-1): 
            state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        state[-1] = self.conf.dt*round(time/self.conf.dt)

        observation = state

        return time, observation

    def step(self, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)
        
        # compute reward
        reward = self.reward(state, action)
        
        return (state_next, reward)

    def simulate(self, state, action):
        ''' Simulate dynamics '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq + nv

        state_next = np.zeros(nx+1)

        # Simulate control action
        self.conf.simu.simulate(np.copy(state), action, self.conf.dt, 1)

        # Return next state
        state_next[:nq], state_next[nq:nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt
        
        return state_next
    
    def reward(self, state, action=None):
        ''' Compute reward '''
        nv = self.conf.robot.nv
        nq = self.conf.robot.nq
        nx = nv + nq

        # End-effector coordinates 
        x_ee = state[0]
        y_ee = state[1] 

        # Penalties for the ellipses representing the obstacle
        ell1_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC1)**2)/((self.A1/2)**2) + ((y_ee-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC2)**2)/((self.A2/2)**2) + ((y_ee-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_pen = math.log(math.exp(self.alpha*-(((x_ee-self.XC3)**2)/((self.A3/2)**2) + ((y_ee-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_reward = math.log(math.exp(self.alpha2*-(math.sqrt((x_ee-self.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-self.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/self.alpha2

        r = (self.w_d*(-(x_ee-self.TARGET_STATE[0])**2 -(y_ee-self.TARGET_STATE[1])**2) + self.w_peak*peak_reward - self.w_ob1*ell1_pen - self.w_ob2*ell2_pen - self.w_ob3*ell3_pen - self.w_u*(action.dot(action)) + 10000)/100 

        return r

    def simulate_and_derivative_tf(self, state, action):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''
        state_next = np.zeros((self.conf.BATCH_SIZE, self.conf.nb_state))
        Fu = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state,self.conf.nb_action))      

        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq+nv

        nu = self.conf.nb_action

        for sample_indx in range(self.conf.BATCH_SIZE):
            state_np = np.copy(state[sample_indx]) 
            
            # Create robot model in Pinocchio with q_init as initial configuration
            q_init = state_np[:nq]
            v_init = state_np[nq:nx]
            
            # Dynamics gradient w.r.t control (1st order euler)
            pin.computeABADerivatives(self.conf.robot.model, self.conf.robot.data, q_init, v_init, action[sample_indx])       

            Fu_sample = np.zeros((nx+1, nu))
            Fu_sample[nv:-1, :] = self.conf.robot.data.Minv
            Fu_sample[:nx, :] *= self.conf.dt

            Fu[sample_indx] = Fu_sample

            # Simulate control action
            self.conf.simu.simulate(np.concatenate((q_init, v_init)), action[sample_indx], self.conf.dt, 1)

            state_next[sample_indx,:self.conf.robot.nq] = np.copy(self.conf.simu.q)
            state_next[sample_indx,self.conf.robot.nq:self.conf.robot.nv+self.conf.robot.nq] = np.copy(self.conf.simu.v)
            state_next[sample_indx,-1] = state_np[-1] + self.conf.dt

        return tf.convert_to_tensor(state_next, dtype=tf.float32), Fu 

    def reward_tf(self, state, action, BATCH_SIZE, last_ts):
        ''' Compute reward using tensors. Batch-wise computation '''
        x_ee = state[:,0]
        y_ee = state[:,1] 

        ell1_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC1)**2)/((self.A1/2)**2) + ((y_ee[:]-self.YC1)**2)/((self.B1/2)**2) - 1.0)) + 1)/self.alpha
        ell2_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC2)**2)/((self.A2/2)**2) + ((y_ee[:]-self.YC2)**2)/((self.B2/2)**2) - 1.0)) + 1)/self.alpha
        ell3_pen = tf.math.log(tf.math.exp(self.alpha*-(((x_ee[:]-self.XC3)**2)/((self.A3/2)**2) + ((y_ee[:]-self.YC3)**2)/((self.B3/2)**2) - 1.0)) + 1)/self.alpha
    
        peak_reward = tf.math.log(tf.math.exp(self.alpha2*-(tf.math.sqrt((x_ee[:]-self.TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-self.TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/self.alpha2
    
        r = (self.w_d*(-(x_ee[:]-self.TARGET_STATE[0])**2 -(y_ee[:]-self.TARGET_STATE[1])**2) + self.w_peak*peak_reward - self.w_ob1*ell1_pen - self.w_ob2*ell2_pen - self.w_ob3*ell3_pen - self.w_u*(action[:,0]**2+action[:,1]**2) + 10000)/100 
        r = tf.reshape(r, [r.shape[0], 1])

        return r 

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        q = state[:nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p