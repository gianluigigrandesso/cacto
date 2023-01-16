import gym
from gym.spaces import Box
import numpy as np
import math
import tensorflow as tf
import pinocchio as pin
import manipulator_conf as conf #only for system parameters (M, l, Iz, ...)
from gym.utils import seeding

class Manipulator(gym.Env):
    metadata = {
        "render_modes": [
            "human", "rgb_array"
        ], 
        "render_fps": 4,
    }

    def __init__(self, dt, x_init_min, x_init_max, x_min, x_max,  u_min, u_max, x_target, soft_max_param, obs_param, weight, robot, nb_state, nb_action, NORMALIZE_INPUTS, state_norm_arr, simu):
        self.x_init_min = x_init_min
        self.x_init_max = x_init_max
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.nb_state = nb_state  
        self.nb_action = nb_action 
        self.x_target = x_target
        self.dt = dt
        self.soft_max_param = soft_max_param
        self.obs_param = obs_param
        self.weight = weight
        self.robot = robot
        self.simu = simu
        self.end_effector_frame_id = 'EE'

        self.observation_space = Box(self.x_min, self.x_max, shape=(self.nb_state,), dtype=np.float32)
        self.action_space = Box(self.u_min, self.u_max, shape=(self.nb_action,), dtype=np.float32)  

        self.window = None
        self.clock = None

        self.NORMALIZE_INPUTS = NORMALIZE_INPUTS
        self.state_norm_arr = state_norm_arr
        self.simulation_type = 'euler'      
        self.tau_coulomb_max = 0*np.ones(3)          # Expressed as percentage of torque max


    def step(self, rand_time, state, u):

        # compute next state
        state_next = self.simulate(state, u)  #observation = self._get_obs()

        # compute reward
        reward = self.reward(rand_time, state_next, u)

        return (state_next, reward)

    def get_end_effector_position(self, state, end_effector_frame_id, recompute=True):
        nv = self.robot.nv
        nq = self.robot.nq
        q = state[:nq] 

        RF = self.robot.model.getFrameId(end_effector_frame_id) 

        H = self.robot.framePlacement(q, RF, recompute)
    
        p = H.translation 
        
        return p

    def reward(self, rand_time, x2, u=None):
        nv = self.robot.nv
        nq = self.robot.nq
        nx = nv + nq

        # End-effector coordinates 
        x_ee = conf.x_base + conf.l*(math.cos(x2[0]) + math.cos(x2[0]+x2[1]) + math.cos(x2[0]+x2[1]+x2[2]))
        y_ee = conf.y_base + conf.l*(math.sin(x2[0]) + math.sin(x2[0]+x2[1]) + math.sin(x2[0]+x2[1]+x2[2]))  
        
        #rename reward parameters
        alpha = self.soft_max_param[0]
        alpha2 = self.soft_max_param[1]

        XC1 = self.obs_param[0]
        YC1 = self.obs_param[1]
        XC2 = self.obs_param[2]
        YC2 = self.obs_param[3]
        XC3 = self.obs_param[4]
        YC3 = self.obs_param[5]
        
        A1 = self.obs_param[6]
        B1 = self.obs_param[7]
        A2 = self.obs_param[8]
        B2 = self.obs_param[9]
        A3 = self.obs_param[10]
        B3 = self.obs_param[11]

        w_d = self.weight[0]
        w_u = self.weight[1]
        w_peak = self.weight[2]
        w_ob1 = self.weight[3]
        w_ob2 = self.weight[3]
        w_ob3 = self.weight[3]
        w_v = self.weight[4]

        TARGET_STATE = self.x_target

        # Penalties for the ellipses representing the obstacle
        ell1_pen = math.log(math.exp(alpha*-(((x_ee-XC1)**2)/((A1/2)**2) + ((y_ee-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = math.log(math.exp(alpha*-(((x_ee-XC2)**2)/((A2/2)**2) + ((y_ee-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = math.log(math.exp(alpha*-(((x_ee-XC3)**2)/((A3/2)**2) + ((y_ee-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha

        # Term pushing the agent to stay in the neighborhood of target
        peak_reward = math.log(math.exp(alpha2*-(math.sqrt((x_ee-TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/alpha2

        # Term penalizing the FINAL joint velocity
        if x2[-1] == self.dt*round(rand_time/self.dt):
            vel_joint = x2[nq:nx].dot(x2[nq:nx]) - 10000/w_v
        else:    
            vel_joint = 0

        r = (w_d*(-(x_ee-TARGET_STATE[0])**2 -(y_ee-TARGET_STATE[1])**2) + w_peak*peak_reward - w_v*vel_joint - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(u.dot(u)))/100 

        return r

    #Define reward function with TensorFlow tensors needed in the actor update (to compute its derivatives with TensorFlow autodifferentiation). Batch-wise computation
    def reward_tf(self,x2,u,BATCH_SIZE,last_ts):
    
        # De-normalize x2 because it is normalized if self.NORMALIZE_INPUTS=1. (Mask trick needed because TensorFlow's autodifferentiation doesn't work if tensors' elements are directly modified by accessing them)
        if self.NORMALIZE_INPUTS:
            x2_time = tf.stack([np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), (x2[:,-1]+1)*self.state_norm_arr[-1]/2],1)
            x2_no_time = x2 * self.state_norm_arr
            mask = tf.cast(tf.stack([np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.zeros(BATCH_SIZE)],1),tf.float32)
            x2_not_norm = x2_no_time * mask + x2_time * (1 - mask)
        else:
            x2_not_norm = x2 
    
        x_ee = conf.x_base + conf.l*(tf.math.cos(x2_not_norm[:,0]) + tf.math.cos(x2_not_norm[:,0]+x2_not_norm[:,1]) + tf.math.cos(x2_not_norm[:,0]+x2_not_norm[:,1]+x2_not_norm[:,2]))
        y_ee = conf.y_base + conf.l*(tf.math.sin(x2_not_norm[:,0]) + tf.math.sin(x2_not_norm[:,0]+x2_not_norm[:,1]) + tf.math.sin(x2_not_norm[:,0]+x2_not_norm[:,1]+x2_not_norm[:,2]))
        
        #rename reward parameters
        alpha = self.soft_max_param[0]
        alpha2 = self.soft_max_param[1]

        XC1 = self.obs_param[0]
        YC1 = self.obs_param[1]
        XC2 = self.obs_param[2]
        YC2 = self.obs_param[3]
        XC3 = self.obs_param[4]
        YC3 = self.obs_param[5]
        
        A1 = self.obs_param[6]
        B1 = self.obs_param[7]
        A2 = self.obs_param[8]
        B2 = self.obs_param[9]
        A3 = self.obs_param[10]
        B3 = self.obs_param[11]

        w_d = self.weight[0]
        w_u = self.weight[1]
        w_peak = self.weight[2]
        w_ob1 = self.weight[3]
        w_ob2 = self.weight[3]
        w_ob3 = self.weight[3]
        w_v = self.weight[4]

        TARGET_STATE = self.x_target

        ell1_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC1)**2)/((A1/2)**2) + ((y_ee[:]-YC1)**2)/((B1/2)**2) - 1.0)) + 1)/alpha
        ell2_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC2)**2)/((A2/2)**2) + ((y_ee[:]-YC2)**2)/((B2/2)**2) - 1.0)) + 1)/alpha
        ell3_pen = tf.math.log(tf.math.exp(alpha*-(((x_ee[:]-XC3)**2)/((A3/2)**2) + ((y_ee[:]-YC3)**2)/((B3/2)**2) - 1.0)) + 1)/alpha
    
        peak_reward = tf.math.log(tf.math.exp(alpha2*-(tf.math.sqrt((x_ee[:]-TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/alpha2
    
        vel_joint_list = []
        for i in range(BATCH_SIZE):
            if last_ts[i][0] == 1.0:
                vel_joint_list.append(x2_not_norm[i,3]**2 + x2_not_norm[i,4]**2 + x2_not_norm[i,5]**2 - 10000/w_v)
            else:    
                vel_joint_list.append(0)
        vel_joint = tf.cast(tf.stack(vel_joint_list),tf.float32)
    
        r = (w_d*(-(x_ee[:]-TARGET_STATE[0])**2 -(y_ee[:]-TARGET_STATE[1])**2) + w_peak*peak_reward -w_v*vel_joint - w_ob1*ell1_pen - w_ob2*ell2_pen - w_ob3*ell3_pen - w_u*(u[:,0]**2 + u[:,1]**2 + u[:,2]**2))/100 
    
        return r 

    # Simulate dynamics to get next state
    def simulate(self, state, u):
        nq = self.robot.nq
        nv = self.robot.nv
        nx = nq + nv

        state_next = np.zeros(nx+1)

        # Simulate control u
        self.simu.simulate(state, u, self.dt, 1)
       
        # Return next state
        state_next[:nq], state_next[nq:nx] = np.copy(self.simu.q), np.copy(self.simu.v)
        state_next[-1] = state[-1] + self.dt
        return state_next

    # Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation
    def simulate_and_derivative_tf(self,state,u,BATCH_SIZE):
        
        q0_next, q1_next, q2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        v0_next, v1_next, v2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        Fu = np.zeros((BATCH_SIZE,self.nb_state,self.nb_action))      

        for sample_indx in range(BATCH_SIZE):
            state_np = state[sample_indx]    

            # Create robot model in Pinocchio with q_init as initial configuration
            q_init = np.zeros(self.robot.nq)
            for state_index in range(self.robot.nq):
                q_init[state_index] = state_np[state_index]
            q_init = q_init.T    
            v_init = np.zeros(self.robot.nv)
            for state_index in range(self.robot.nv):
                v_init[state_index] = state_np[self.robot.nq +state_index]
            v_init = v_init.T
            robot = self.robot    

            # Dynamics gradient w.r.t control (1st order euler)
            nq = robot.nq
            nv = robot.nv
            nu = robot.na
            nx = nq+nv
            pin.computeABADerivatives(robot.model, robot.data, q_init, v_init, u[sample_indx])        
            Fu_sample = np.zeros((nx, nu))
            Fu_sample[nv:, :] = robot.data.Minv
            Fu_sample *= self.dt
            if self.NORMALIZE_INPUTS:
                Fu_sample *= (1/self.state_norm_arr[3])    

            Fu[sample_indx] = np.vstack((Fu_sample, np.zeros(self.nb_action)))    

            # Simulate control u
            self.simu.simulate(np.concatenate((q_init, v_init)), u[sample_indx], self.dt, 1)
            q_next, v_next, = np.copy(self.simu.q), np.copy(self.simu.v)
            q0_next_sample, q1_next_sample, q2_next_sample = q_next[0],q_next[1],q_next[2]
            v0_next_sample, v1_next_sample, v2_next_sample = v_next[0],v_next[1],v_next[2]
            q0_next[sample_indx] = q0_next_sample
            q1_next[sample_indx] = q1_next_sample
            q2_next[sample_indx] = q2_next_sample
            v0_next[sample_indx] = v0_next_sample
            v1_next[sample_indx] = v1_next_sample
            v2_next[sample_indx] = v2_next_sample    

        t_next = state[:,-1] + self.dt    
        
        if self.NORMALIZE_INPUTS:
            q0_next = q0_next / self.state_norm_arr[0]
            q1_next = q1_next / self.state_norm_arr[1]
            q2_next = q2_next / self.state_norm_arr[2]
            v0_next = v0_next / self.state_norm_arr[3]
            v1_next = v1_next / self.state_norm_arr[4]
            v2_next = v2_next / self.state_norm_arr[5]
            t_next = 2*t_next/self.state_norm_arr[-1] -1     

        Fu = tf.convert_to_tensor(Fu,dtype=tf.float32)    

        return tf.cast(tf.stack([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next],1),dtype=tf.float32), Fu
