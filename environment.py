import gym
from gym.spaces import Box
import numpy as np
import math
import tensorflow as tf
import pinocchio as pin
from gym.utils import seeding
import sys
import random

class Manipulator(gym.Env):
    '''
    :param robot :                  (RobotWrapper instance) 
    :param simu :                   (RobotSimulator instance)
    :param x_init_min :             (float array) State lower bound initial configuration array
    :param x_init_max :             (float array) State upper bound initial configuration array
    :param x_min :                  (float array) State lower bound vector
    :param x_max :                  (float array) State upper bound vector
    :param state_norm_arr :         (float array) Array used to normalize states
    :param u_min :                  (float array) Action lower bound array
    :param u_max :                  (float array) Action upper bound array
    :param nb_state :               (int) State size (robot state size + 1)
    :param nb_action :              (int) Action size (robot action size)
    :param dt :                     (float) Timestep
    :param TARGET_STATE :           (float array) Target position
    :param soft_max_param :         (float array) Soft parameters array
    :param obs_param :              (float array) Obtacle parameters array
    :param weight :                 (float array) Weights array
    :param end_effector_frame_id :  (str)
    :param NORMALIZE_INPUTS :       (bool)
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
    
    def reset(self, options=None):
        ''' Choose initial state uniformly at random '''
        self._state = np.zeros(self.conf.nb_state) 

        rand_time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1])
        for i in range(self.conf.nb_state-1): 
            self._state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        self._state[-1] = self.conf.dt*round(rand_time/self.conf.dt)
        observation = self._state

        return rand_time, observation

    def step(self, rand_time, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)

        # compute reward
        reward = self.reward(rand_time, state_next, action)

        return (state_next, reward)

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        nv = self.conf.robot.nv
        nq = self.conf.robot.nq
        q = state[:nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p

    def reward(self, rand_time, state_next, action=None):
        ''' Compute reward '''
        nv = self.conf.robot.nv
        nq = self.conf.robot.nq
        nx = nv + nq

        # End-effector coordinates 
        x_ee = self.conf.x_base + self.conf.l*(math.cos(state_next[0]) + math.cos(state_next[0]+state_next[1]) + math.cos(state_next[0]+state_next[1]+state_next[2]))
        y_ee = self.conf.y_base + self.conf.l*(math.sin(state_next[0]) + math.sin(state_next[0]+state_next[1]) + math.sin(state_next[0]+state_next[1]+state_next[2]))  
        
        #rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        TARGET_STATE = self.conf.TARGET_STATE

        # Penalties for the ellipses representing the obstacle
        ell1_pen = math.log(math.exp(alpha*-(((x_ee-XC1)**2)/((A1/2)**2) + ((y_ee-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = math.log(math.exp(alpha*-(((x_ee-XC2)**2)/((A2/2)**2) + ((y_ee-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = math.log(math.exp(alpha*-(((x_ee-XC3)**2)/((A3/2)**2) + ((y_ee-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_reward = math.log(math.exp(alpha2*-(math.sqrt((x_ee-TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/alpha2

        # Term penalizing the FINAL joint velocity
        if state_next[-1] == self.conf.dt*round(rand_time/self.conf.dt):
            vel_joint = state_next[nq:nx].dot(state_next[nq:nx]) - 10000/w_v
        else:    
            vel_joint = 0

        r = (w_d*(-(x_ee-TARGET_STATE[0])**2 -(y_ee-TARGET_STATE[1])**2) + w_peak*peak_reward - w_v*vel_joint - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(action.dot(action)))/100 

        return r

    def reward_tf(self,state_next,action,BATCH_SIZE,last_ts):
        ''' Compute reward using tensors. Batch-wise computation '''    
        # De-normalize state_next because it is normalized if self.conf.NORMALIZE_INPUTS=1. (Mask trick needed because TensorFlow's autodifferentiation doesn't work if tensors' elements are directly modified by accessing them)
        if self.conf.NORMALIZE_INPUTS:
            state_next_not_norm = self.de_normalize(state_next, BATCH_SIZE)
        else:
            state_next_not_norm = state_next 
    
        x_ee = self.conf.x_base + self.conf.l*(tf.math.cos(state_next_not_norm[:,0]) + tf.math.cos(state_next_not_norm[:,0]+state_next_not_norm[:,1]) + tf.math.cos(state_next_not_norm[:,0]+state_next_not_norm[:,1]+state_next_not_norm[:,2]))
        y_ee = self.conf.y_base + self.conf.l*(tf.math.sin(state_next_not_norm[:,0]) + tf.math.sin(state_next_not_norm[:,0]+state_next_not_norm[:,1]) + tf.math.sin(state_next_not_norm[:,0]+state_next_not_norm[:,1]+state_next_not_norm[:,2]))
        
        #rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        TARGET_STATE = self.conf.TARGET_STATE

        ell1_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC1)**2)/((A1/2)**2) + ((y_ee[:]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC2)**2)/((A2/2)**2) + ((y_ee[:]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC3)**2)/((A3/2)**2) + ((y_ee[:]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha
    
        peak_reward = tf.math.log(tf.math.exp(alpha2*-(tf.math.sqrt((x_ee[:]-TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/alpha2
    
        vel_joint_list = []
        for i in range(BATCH_SIZE):
            if last_ts[i][0] == 1.0:
                vel_joint_list.append(state_next_not_norm[i,3]**2 + state_next_not_norm[i,4]**2 + state_next_not_norm[i,5]**2 - 10000/w_v)
            else:    
                vel_joint_list.append(0)
        vel_joint = tf.cast(tf.stack(vel_joint_list),tf.float32)
    
        r = (w_d*(-(x_ee[:]-TARGET_STATE[0])**2 -(y_ee[:]-TARGET_STATE[1])**2) + w_peak*peak_reward -w_v*vel_joint - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(action[:,0]**2+action[:,1]**2+action[:,2]**2))/100 
        r = tf.reshape(r, [r.shape[0], 1])

        return r 

    # Simulate dynamics to get next state
    def simulate(self, state, action):
        ''' Simulate dynamics '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq + nv

        state_next = np.zeros(nx+1)

        # Simulate control action
        self.conf.simu.simulate(state, action, self.conf.dt, 1)
       
        # Return next state
        state_next[:nq], state_next[nq:nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt
        return state_next

    # Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation
    def simulate_and_derivative_tf(self,state,action, BATCH_SIZE, der=1):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''        
        q0_next, q1_next, q2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        v0_next, v1_next, v2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)

        if der==1:
            Fu = np.zeros((BATCH_SIZE,self.conf.nb_state,self.conf.nb_action))      

        for sample_indx in range(BATCH_SIZE):
            state_np = state[sample_indx]    

            # Create robot model in Pinocchio with q_init as initial configuration
            q_init = np.zeros(self.conf.robot.nq)
            for state_index in range(self.conf.robot.nq):
                q_init[state_index] = state_np[state_index]
            q_init = q_init.T    
            v_init = np.zeros(self.conf.robot.nv)
            for state_index in range(self.conf.robot.nv):
                v_init[state_index] = state_np[self.conf.robot.nq +state_index]
            v_init = v_init.T
            robot = self.conf.robot    
            
            if der ==1:
                # Dynamics gradient w.r.t control (1st order euler)
                nq = robot.nq
                nv = robot.nv
                nu = robot.na
                nx = nq+nv
                pin.computeABADerivatives(robot.model, robot.data, q_init, v_init, action[sample_indx])        
                Fu_sample = np.zeros((nx, nu))
                Fu_sample[nv:, :] = robot.data.Minv
                Fu_sample *= self.conf.dt
                if self.conf.NORMALIZE_INPUTS:
                    Fu_sample *= (1/self.conf.state_norm_arr[:-1,None]) 
    
                Fu[sample_indx] = np.vstack((Fu_sample, np.zeros(self.conf.nb_action)))    

            # Simulate control action
            self.conf.simu.simulate(np.concatenate((q_init, v_init)), action[sample_indx], self.conf.dt, 1)
            q_next, v_next, = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
            q0_next_sample, q1_next_sample, q2_next_sample = q_next[0],q_next[1],q_next[2]
            v0_next_sample, v1_next_sample, v2_next_sample = v_next[0],v_next[1],v_next[2]
            q0_next[sample_indx] = q0_next_sample
            q1_next[sample_indx] = q1_next_sample
            q2_next[sample_indx] = q2_next_sample
            v0_next[sample_indx] = v0_next_sample
            v1_next[sample_indx] = v1_next_sample
            v2_next[sample_indx] = v2_next_sample    

        t_next = state[:,-1] + self.conf.dt    
        
        if self.conf.NORMALIZE_INPUTS:
            q0_next = q0_next / self.conf.state_norm_arr[0]
            q1_next = q1_next / self.conf.state_norm_arr[1]
            q2_next = q2_next / self.conf.state_norm_arr[2]
            v0_next = v0_next / self.conf.state_norm_arr[3]
            v1_next = v1_next / self.conf.state_norm_arr[4]
            v2_next = v2_next / self.conf.state_norm_arr[5]
            t_next = 2*t_next/self.conf.state_norm_arr[-1] -1     

        if der==1:
            Fu = tf.convert_to_tensor(Fu,dtype=tf.float32)    

            return tf.cast(tf.stack([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next],1),dtype=tf.float32), Fu
        
        return tf.cast(tf.stack([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next],1),dtype=tf.float32)
    
    def simulate_tf(self,state,action):
        ''' Simulate dynamics using tensors. Batch-wise computation '''
        print('Sobolev training for Manipulator not implemented yet')
        sys.exit()

    def de_normalize(self, state, BATCH_SIZE):
        ''' Retrieve state from normalized state '''
        state_time = tf.stack([np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), (state[:,-1]+1)*self.conf.state_norm_arr[-1]/2],1)
        state_no_time = state * self.conf.state_norm_arr
        mask = tf.cast(tf.stack([np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.zeros(BATCH_SIZE)],1),tf.float32)
        state_not_norm = state_no_time * mask + state_time * (1 - mask)

        return state_not_norm
    



class DoubleIntegrator(gym.Env):
    '''
    :param robot :                  (RobotWrapper instance) 
    :param simu :                   (RobotSimulator instance)
    :param x_init_min :             (float array) State lower bound initial configuration array
    :param x_init_max :             (float array) State upper bound initial configuration array
    :param x_min :                  (float array) State lower bound vector
    :param x_max :                  (float array) State upper bound vector
    :param state_norm_arr :         (float array) Array used to normalize states
    :param u_min :                  (float array) Action lower bound array
    :param u_max :                  (float array) Action upper bound array
    :param nb_state :               (int) State size (robot state size + 1)
    :param nb_action :              (int) Action size (robot action size)
    :param dt :                     (float) Timestep
    :param TARGET_STATE :           (float array) Target position
    :param soft_max_param :         (float array) Soft parameters array
    :param obs_param :              (float array) Obtacle parameters array
    :param weight :                 (float array) Weights array
    :param end_effector_frame_id :  (str)
    :param NORMALIZE_INPUTS :       (bool)
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
    
    def reset(self, options=None):
        ''' Choose initial state uniformly at random '''
        self._state = np.zeros(self.conf.nb_state)    

        rand_time = random.uniform(self.conf.x_init_min[-1], self.conf.x_init_max[-1]) 
        for i in range(self.conf.nb_state-1): 
            self._state[i] = random.uniform(self.conf.x_init_min[i], self.conf.x_init_max[i]) 
        self._state[-1] = self.conf.dt*round(rand_time/self.conf.dt)

        observation = self._state

        return rand_time, observation

    def step(self, rand_time, state, action):
        ''' Return next state and reward '''
        # compute next state
        state_next = self.simulate(state, action)  #observation = self._get_obs()

        # compute reward
        reward = self.reward(rand_time, state_next, action)

        return (state_next, reward)

    def get_end_effector_position(self, state, recompute=True):
        ''' Compute end-effector position '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        q = state[:nq] 

        RF = self.conf.robot.model.getFrameId(self.conf.end_effector_frame_id) 

        H = self.conf.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p

    def reward(self, rand_time, state_next, action=None):
        ''' Compute reward '''
        nv = self.conf.robot.nv
        nq = self.conf.robot.nq
        nx = nv + nq

        # End-effector coordinates 
        x_ee = state_next[0]
        y_ee = state_next[1] 
        
        #rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        TARGET_STATE = self.conf.TARGET_STATE

        # Penalties for the ellipses representing the obstacle
        ell1_pen = math.log(math.exp(alpha*-(((x_ee-XC1)**2)/((A1/2)**2) + ((y_ee-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = math.log(math.exp(alpha*-(((x_ee-XC2)**2)/((A2/2)**2) + ((y_ee-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = math.log(math.exp(alpha*-(((x_ee-XC3)**2)/((A3/2)**2) + ((y_ee-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_reward = math.log(math.exp(alpha2*-(math.sqrt((x_ee-TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/alpha2

        r = (w_d*(-(x_ee-TARGET_STATE[0])**2 -(y_ee-TARGET_STATE[1])**2) + w_peak*peak_reward - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(action.dot(action)) + 10000)/100 

        return r

    def reward_tf(self,state_next,action,BATCH_SIZE,last_ts):
        ''' Compute reward using tensors. Batch-wise computation '''
        # De-normalize state_next because it is normalized if self.conf.NORMALIZE_INPUTS=1. (Mask trick needed because TensorFlow's autodifferentiation doesn't work if tensors' elements are directly modified by accessing them)
        if self.conf.NORMALIZE_INPUTS:
            state_next_not_norm = self.de_normalize(state_next, BATCH_SIZE)
        else:
            state_next_not_norm = state_next 
    
        x_ee = state_next_not_norm[:,0]
        y_ee = state_next_not_norm[:,1] 

        #rename reward parameters
        alpha = self.conf.soft_max_param[0]
        alpha2 = self.conf.soft_max_param[1]

        XC1 = self.conf.obs_param[0]
        YC1 = self.conf.obs_param[1]
        XC2 = self.conf.obs_param[2]
        YC2 = self.conf.obs_param[3]
        XC3 = self.conf.obs_param[4]
        YC3 = self.conf.obs_param[5]
        
        A1 = self.conf.obs_param[6]
        B1 = self.conf.obs_param[7]
        A2 = self.conf.obs_param[8]
        B2 = self.conf.obs_param[9]
        A3 = self.conf.obs_param[10]
        B3 = self.conf.obs_param[11]

        w_d = self.conf.weight[0]
        w_u = self.conf.weight[1]
        w_peak = self.conf.weight[2]
        w_ob1 = self.conf.weight[3]
        w_ob2 = self.conf.weight[3]
        w_ob3 = self.conf.weight[3]
        w_v = self.conf.weight[4]

        TARGET_STATE = self.conf.TARGET_STATE

        ell1_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC1)**2)/((A1/2)**2) + ((y_ee[:]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC2)**2)/((A2/2)**2) + ((y_ee[:]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC3)**2)/((A3/2)**2) + ((y_ee[:]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha
    
        peak_reward = tf.math.log(tf.math.exp(alpha2*-(tf.math.sqrt((x_ee[:]-TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/alpha2
    
        r = (w_d*(-(x_ee[:]-TARGET_STATE[0])**2 -(y_ee[:]-TARGET_STATE[1])**2) + w_peak*peak_reward - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(action[:,0]**2+action[:,1]**2) + 10000)/100 
        r = tf.reshape(r, [r.shape[0], 1])

        return r 

    def simulate(self, state, action):
        ''' Simulate dynamics '''
        nq = self.conf.robot.nq
        nv = self.conf.robot.nv
        nx = nq + nv

        state_next = np.zeros(nx+1)

        # Simulate control action
        self.conf.simu.simulate(state, action, self.conf.dt, 1)

        # Return next state
        state_next[:nq], state_next[nq:nx] = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
        state_next[-1] = state[-1] + self.conf.dt
        
        return state_next

    def simulate_and_derivative_tf(self,state,action,BATCH_SIZE, der=1):
        ''' Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation '''
        q0_next, q1_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        v0_next, v1_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        if der==1:
            Fu = np.zeros((BATCH_SIZE,self.conf.nb_state,self.conf.nb_action))      

        for sample_indx in range(BATCH_SIZE):
            state_np = state[sample_indx]    

            # Create robot model in Pinocchio with q_init as initial configuration
            q_init = np.zeros(self.conf.robot.nq)
            for state_index in range(self.conf.robot.nq):
                q_init[state_index] = state_np[state_index]
            q_init = q_init.T    
            v_init = np.zeros(self.conf.robot.nv)
            for state_index in range(self.conf.robot.nv):
                v_init[state_index] = state_np[self.conf.robot.nq +state_index]
            v_init = v_init.T
            robot = self.conf.robot    

            if der==1:
                # Dynamics gradient w.r.t control (1st order euler)
                nq = robot.nq
                nv = robot.nv
                nu = robot.na
                nx = nq+nv
                pin.computeABADerivatives(robot.model, robot.data, q_init, v_init, action[sample_indx])        
                Fu_sample = np.zeros((nx, nu))
                Fu_sample[nv:, :] = robot.data.Minv
                Fu_sample *= self.conf.dt
                if self.conf.NORMALIZE_INPUTS:
                    Fu_sample *= (1/self.conf.state_norm_arr[:-1,None]) 

                Fu[sample_indx] = np.vstack((Fu_sample, np.zeros(self.conf.nb_action)))    

            # Simulate control action
            self.conf.simu.simulate(np.concatenate((q_init, v_init)), action[sample_indx], self.conf.dt, 1)
            q_next, v_next, = np.copy(self.conf.simu.q), np.copy(self.conf.simu.v)
            q0_next_sample, q1_next_sample = q_next[0],q_next[1]
            v0_next_sample, v1_next_sample = v_next[0],v_next[1]
            q0_next[sample_indx] = q0_next_sample
            q1_next[sample_indx] = q1_next_sample
            v0_next[sample_indx] = v0_next_sample
            v1_next[sample_indx] = v1_next_sample   

        t_next = state[:,-1] + self.conf.dt    
        
        if self.conf.NORMALIZE_INPUTS:
            q0_next = q0_next / self.conf.state_norm_arr[0]
            q1_next = q1_next / self.conf.state_norm_arr[1]
            v0_next = v0_next / self.conf.state_norm_arr[2]
            v1_next = v1_next / self.conf.state_norm_arr[3]
            t_next = 2*t_next/self.conf.state_norm_arr[-1] -1     

        if der==1:
            Fu = tf.convert_to_tensor(Fu,dtype=tf.float32)    

            return tf.cast(tf.stack([q0_next,q1_next,v0_next,v1_next,t_next],1),dtype=tf.float32), Fu
        
        return tf.cast(tf.stack([q0_next,q1_next,v0_next,v1_next,t_next],1),dtype=tf.float32)

    def simulate_tf(self,state,action):
        ''' Simulate dynamics using tensors. Batch-wise computation '''
        q0_next = state[:,0] + self.conf.dt*state[:,2] + 0.5*action[:,0]*self.conf.dt**2
        q1_next = state[:,1] + self.conf.dt*state[:,3] + 0.5*action[:,1]*self.conf.dt**2
        v0_next = state[:,2] + self.conf.dt*action[:,0]
        v1_next = state[:,3] + self.conf.dt*action[:,1]
        t_next  = state[:,-1]+ self.conf.dt
        if self.conf.NORMALIZE_INPUTS:
            q0_next = q0_next / self.conf.state_norm_arr[0]
            q1_next = q1_next / self.conf.state_norm_arr[1]
            v0_next = v0_next / self.conf.state_norm_arr[2]
            v1_next = v1_next / self.conf.state_norm_arr[3]
            t_next = 2*t_next / self.conf.state_norm_arr[-1] -1  

        return tf.stack([q0_next, q1_next, v0_next, v1_next, t_next],1)

    def de_normalize(self, state, BATCH_SIZE):
        ''' Retrieve state from normalized state '''
        state_time = tf.stack([np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), (state[:,-1]+1)*self.conf.state_norm_arr[-1]/2],1)
        state_no_time = state * self.conf.state_norm_arr
        mask = tf.cast(tf.stack([np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.zeros(BATCH_SIZE)],1),tf.float32)
        state_not_norm = state_no_time * mask + state_time * (1 - mask)

        return state_not_norm
