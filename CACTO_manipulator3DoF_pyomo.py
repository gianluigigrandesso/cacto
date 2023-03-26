import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import random
from pyomo.environ import *
from pyomo.dae import *
import pinocchio as pin
from TO_manipulator3DoF_pyomo import TO_manipulator, plot_results_TO
import config_manipulator3DoF_pyomo as conf
from dynamics_manipulator3DoF import RobotWrapper, RobotSimulator, load_urdf
from replay_buffer import PrioritizedReplayBuffer
from inits import init_tau0,init_tau1,init_tau2,init_q0,init_q1,init_q2,init_v0,init_v1,init_v2,init_q0_ICS,init_q1_ICS,init_q2_ICS,init_v0_ICS,init_v1_ICS,init_v2_ICS,init_0

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Uncomment to run TF on CPU rather than GPU

# Class implementing the NNs update
class Training:
    def __init__(self, batch_size=conf.BATCH_SIZE):
        self.batch_size = batch_size
        return
    
    # Update both critic and actor                                                                     
    def update(self, episode, red_state_batch_norm, red_state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, weights_batch=None):

        with tf.GradientTape() as tape:

            # Update the critic
            if conf.TD_N:
                y = reward_batch                                                        # When using n-step TD, reward_batch is the batch of costs-to-go and not the batch of single step rewards
            else:    
                if conf.NORMALIZE_INPUTS:
                    target_values = target_critic(next_state_batch_norm, training=True) # Compute Value at next state after conf.nsteps_TD_N steps given by target critic 
                else:
                    target_values = target_critic(next_state_batch, training=True)
                y = reward_batch + (1-d_batch)*target_values                            # Compute batch of 1-step targets for the critic loss 
            
            if conf.NORMALIZE_INPUTS:
                critic_value = critic_model(state_batch_norm, training=True)            # Compute batch of Values associated to the sampled batch of states
            else:
                critic_value = critic_model(state_batch, training=True)

            if weights_batch is None:
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))     # Critic loss function (tf.math.reduce_mean computes the mean of elements across dimensions of a tensor, in this case across the batch)
            else:
                critic_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(weights_batch,(y - critic_value))))       

        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)    

        # Update the critic backpropagating the gradients
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        # Update the actor after the critic pre-training
        if episode >= conf.EPISODE_CRITIC_PRETRAINING:                                       
            if conf.NORMALIZE_INPUTS:
                actions = actor_model(red_state_batch_norm, training=True)
            else:
                actions = actor_model(red_state_batch, training=True)

            # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
            next_state_tf, ds_next_da = simulate_and_derivative_tf(red_state_batch,actions.numpy(),self.batch_size)  

            with tf.GradientTape() as tape:
                tape.watch(next_state_tf)
                critic_value_next = critic_model(next_state_tf, training=True)                                  # next_state_batch = next state after applying policy's action, already normalized if conf.NORMALIZE_INPUTS=1
            dV_ds_next = tape.gradient(critic_value_next, next_state_tf)                                        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)   

            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                        tape1.watch(actions)
                        tape2.watch(next_state_tf)
                        rewards_tf = reward_tf(next_state_tf, actions, self.batch_size, d_batch)
            dr_da = tape1.gradient(rewards_tf,actions)                                                          # dr_da = gradient of reward r w.r.t. policy's action a
            dr_ds_next = tape2.gradient(rewards_tf,next_state_tf)                                               # dr_ds' = gradient of reward r w.r.t. next state s' after performing policy's action a

            dr_ds_next_dV_ds_next_reshaped = tf.reshape(dr_ds_next+dV_ds_next,(self.batch_size,1,num_states))   # dr_ds' + dV_ds'
            dr_ds_next_dV_ds_next = tf.matmul(dr_ds_next_dV_ds_next_reshaped,ds_next_da)                        # (dr_ds' + dV_ds')*ds'_da
            dr_da_reshaped = tf.reshape(dr_da,(self.batch_size,1,num_actions))                               
            tf_sum = dr_ds_next_dV_ds_next+dr_da_reshaped                                                       # (dr_ds' + dV_ds')*ds'_da + dr_da

            # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
            with tf.GradientTape() as tape:
                tape.watch(actor_model.trainable_variables)
                if conf.NORMALIZE_INPUTS:
                    actions = actor_model(state_batch_norm, training=True)
                else:
                    actions = actor_model(state_batch, training=True)
                actions_reshaped = tf.reshape(actions,(self.batch_size,num_actions,1))
                tf_sum_reshaped = tf.reshape(tf_sum,(self.batch_size,1,num_actions))    
                Q_neg = tf.matmul(-tf_sum_reshaped,actions_reshaped) 
                
                mean_Qneg = tf.math.reduce_mean(Q_neg)                  # Also here we need a scalar so we compute the mean -Q across the batch

            # Gradients of the actor loss w.r.t. actor's parameters
            actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables) 
            
            # Update the actor backpropagating the gradients
            actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    def learn(self,episode, prioritized_buffer):
        # Sample batch of transitions from the buffer
        experience = prioritized_buffer.sample(self.batch_size, beta=conf.prioritized_replay_eps)            # Bias annealing not performed, that's why beta is equal to a very small number (0 not accepted by PrioritizedReplayBuffer)
        x_batch, a_next_batch, a_batch, r_batch, x2_batch, d_batch, weights, batch_idxes = experience   # Importance sampling weights (actually not used) should anneal the bias (see Prioritized Experience Replay paper) 
        
        # Convert batch of transitions into a tensor
        x_batch = x_batch.reshape(self.batch_size,num_states)
        a_next_batch = a_next_batch.reshape(self.batch_size,num_actions)
        a_batch = a_batch.reshape(self.batch_size,num_actions)
        r_batch = r_batch.reshape(self.batch_size,1)
        x2_batch = x2_batch.reshape(self.batch_size,num_states)
        d_batch = d_batch.reshape(self.batch_size,1)
        weights = weights.reshape(self.batch_size,1)
        state_batch = tf.convert_to_tensor(x_batch)
        state_batch = tf.cast(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(a_batch)
        action_batch = tf.cast(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(r_batch)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)                                     
        d_batch = tf.cast(d_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(x2_batch)
        next_state_batch = tf.cast(next_state_batch, dtype=tf.float32)                
        if conf.prioritized_replay_alpha == 0:
            weights_batch = tf.convert_to_tensor(np.ones_like(weights), dtype=tf.float32)
        else:    
            weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)
    
        if conf.NORMALIZE_INPUTS:
            x2_batch_norm = x2_batch / conf.state_norm_arr
            x2_batch_norm[:,-1] = 2*x2_batch_norm[:,-1] -1
            next_state_batch_norm = tf.convert_to_tensor(x2_batch_norm)
            next_state_batch_norm = tf.cast(next_state_batch_norm, dtype=tf.float32)
            x_batch_norm = x_batch / conf.state_norm_arr
            x_batch_norm[:,-1] = 2*x_batch_norm[:,-1] -1
            state_batch_norm = tf.convert_to_tensor(x_batch_norm)
            state_batch_norm = tf.cast(state_batch_norm, dtype=tf.float32)
          
        # Update priorities
        if conf.prioritized_replay_alpha != 0:
            if conf.NORMALIZE_INPUTS:
                v_batch = critic_model(state_batch_norm, training=True)                      # Compute batch of Values associated to the sampled batch ofstates
                v2_batch = target_critic(next_state_batch_norm, training=True)               # Compute batch of Values from target critic associated to sampled batch of next states
                vref_batch = reward_batch + (1-d_batch)*(v2_batch)                           # Compute the targets for the TD error         
            else:
                v_batch = critic_model(state_batch, training=True) 
                v2_batch = target_critic(next_state_batch, training=True)                     
                vref_batch = reward_batch + (1-d_batch)*(v2_batch)                          

            td_errors = tf.math.abs(tf.math.subtract(vref_batch,v_batch))                           
            new_priorities = td_errors.numpy() + conf.prioritized_replay_eps                      # Proportional prioritization where p_i = |TD_error_i| + conf.prioritized_replay_eps 
            prioritized_buffer.update_priorities(batch_idxes, new_priorities)                                               
        
        # Update NNs
        self.update(episode, state_batch_norm, state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, weights_batch=weights_batch)

# Update target critic NN
# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows TensorFlow to build a static graph out of the logic and computations in our function.
# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one. Working only with TensorFlow tensors (e.g. not working with Numpy arrays)
@tf.function                                 
def update_target(target_weights, weights, tau): 
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))    

# Create actor NN
def get_actor():

    inputs = layers.Input(shape=(conf.num_states,))
    lay1 = layers.Dense(conf.NH1,kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A))(inputs)                                        
    leakyrelu1 = layers.LeakyReLU()(lay1)
 
    lay2 = layers.Dense(conf.NH2, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A))(leakyrelu1)                                           
    leakyrelu2 = layers.LeakyReLU()(lay2)

    outputs = layers.Dense(conf.num_actions, activation="tanh", kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_A,conf.wreg_l2_A))(leakyrelu2) 

    outputs = outputs * conf.tau_upper_bound          # Bound actions
    model = tf.keras.Model(inputs, outputs)
    return model 

# Create critic NN 
def get_critic(): 
    state_input = layers.Input(shape=(num_states,))
    state_out1 = layers.Dense(16, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C))(state_input) 
    leakyrelu1 = layers.LeakyReLU()(state_out1)

    state_out2 = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C))(leakyrelu1) 
    leakyrelu2 = layers.LeakyReLU()(state_out2)
    
    out_lay1 = layers.Dense(conf.NH1, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C))(leakyrelu2)
    leakyrelu3 = layers.LeakyReLU()(out_lay1)
    
    out_lay2 = layers.Dense(conf.NH2, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C))(leakyrelu3)
    leakyrelu4 = layers.LeakyReLU()(out_lay2)

    outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(conf.wreg_l1_C,conf.wreg_l2_C))(leakyrelu4)

    model = tf.keras.Model([state_input], outputs)

    return model        

# Get optimal actions at timestep "step" in the current episode
def get_TO_actions(step, action0_TO=None, action1_TO=None, action2_TO=None):
       
    actions = np.array([action0_TO[step], action1_TO[step], action2_TO[step]])
    
    # Buond actions in case they are not already bounded in the TO problem
    if step < NSTEPS_SH:
        next_actions = np.array([np.clip(action0_TO[step+1], conf.tau_lower_bound, conf.tau_upper_bound), np.clip(action1_TO[step+1], conf.tau_lower_bound, conf.tau_upper_bound), np.clip(action2_TO[step+1], conf.tau_lower_bound, conf.tau_upper_bound)])
    else:
        next_actions = np.copy(actions) # Actions at last step of the episode are not performed 

    return actions, next_actions

# Define reward (-cost) function
def reward(x2,u=None):
    # End-effector coordinates
    x_ee = conf.x_base + conf.l*(math.cos(x2[0]) + math.cos(x2[0]+x2[1]) + math.cos(x2[0]+x2[1]+x2[2]))
    y_ee = conf.y_base + conf.l*(math.sin(x2[0]) + math.sin(x2[0]+x2[1]) + math.sin(x2[0]+x2[1]+x2[2]))   

    # Penalties for the ellipses representing the obstacle
    ell1_pen = math.log(math.exp(conf.alpha*-(((x_ee-conf.XC1)**2)/((conf.A1/2)**2) + ((y_ee-conf.YC1)**2)/((conf.B1/2)**2) - 1.0)) + 1)/conf.alpha
    ell2_pen = math.log(math.exp(conf.alpha*-(((x_ee-conf.XC2)**2)/((conf.A2/2)**2) + ((y_ee-conf.YC2)**2)/((conf.B2/2)**2) - 1.0)) + 1)/conf.alpha
    ell3_pen = math.log(math.exp(conf.alpha*-(((x_ee-conf.XC3)**2)/((conf.A3/2)**2) + ((y_ee-conf.YC3)**2)/((conf.B3/2)**2) - 1.0)) + 1)/conf.alpha

    # Term pushing the agent to stay in the neighborhood of target
    peak_reward = math.log(math.exp(conf.alpha2*-(math.sqrt((x_ee-conf.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1) - 0.1 + math.sqrt((y_ee-conf.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1) - 0.1)) + 1)/conf.alpha2

    # Term penalizing the FINAL joint velocity
    if x2[-1] == conf.dt*conf.NSTEPS:
        vel_joint = x2[3]**2 + x2[4]**2 + x2[5]**2 - 10000/conf.w_v
    else:    
        vel_joint = 0

    r = (conf.w_d*(-(x_ee-conf.TARGET_STATE[0])**2 -(y_ee-conf.TARGET_STATE[1])**2) + conf.w_peak*peak_reward - conf.w_v*vel_joint - conf.w_ob1*ell1_pen - conf.w_ob2*ell2_pen - conf.w_ob3*ell3_pen - conf.w_u*(u[0]**2 + u[1]**2 + u[2]**2))/100 
    
    return r

# Define reward function with TensorFlow tensors needed in the actor update (to compute its derivatives with TensorFlow autodifferentiation). Batch-wise computation
def reward_tf(x2,u,BATCH_SIZE,last_ts):

    # De-normalize x2 because it is normalized if conf.NORMALIZE_INPUTS=1. (Mask trick needed because TensorFlow's autodifferentiation doesn't work if tensors' elements are directly modified by accessing them)
    if conf.NORMALIZE_INPUTS:
        x2_time = tf.stack([np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), (x2[:,-1]+1)*conf.state_norm_arr[-1]/2],1)
        x2_no_time = x2 * conf.state_norm_arr
        mask = tf.cast(tf.stack([np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.ones(BATCH_SIZE), np.zeros(BATCH_SIZE)],1),tf.float32)
        x2_not_norm = x2_no_time * mask + x2_time * (1 - mask)
    else:
        x2_not_norm = x2 

    x_ee = conf.x_base + conf.l*(tf.math.cos(x2_not_norm[:,0]) + tf.math.cos(x2_not_norm[:,0]+x2_not_norm[:,1]) + tf.math.cos(x2_not_norm[:,0]+x2_not_norm[:,1]+x2_not_norm[:,2]))
    y_ee = conf.y_base + conf.l*(tf.math.sin(x2_not_norm[:,0]) + tf.math.sin(x2_not_norm[:,0]+x2_not_norm[:,1]) + tf.math.sin(x2_not_norm[:,0]+x2_not_norm[:,1]+x2_not_norm[:,2]))

    ell1_pen = tf.math.log(tf.math.exp(conf.alpha*-(((x_ee[:]-conf.XC1)**2)/((conf.A1/2)**2) + ((y_ee[:]-conf.YC1)**2)/((conf.B1/2)**2) - 1.0)) + 1)/conf.alpha
    ell2_pen = tf.math.log(tf.math.exp(conf.alpha*-(((x_ee[:]-conf.XC2)**2)/((conf.A2/2)**2) + ((y_ee[:]-conf.YC2)**2)/((conf.B2/2)**2) - 1.0)) + 1)/conf.alpha
    ell3_pen = tf.math.log(tf.math.exp(conf.alpha*-(((x_ee[:]-conf.XC3)**2)/((conf.A3/2)**2) + ((y_ee[:]-conf.YC3)**2)/((conf.B3/2)**2) - 1.0)) + 1)/conf.alpha

    peak_reward = tf.math.log(tf.math.exp(conf.alpha2*-(tf.math.sqrt((x_ee[:]-conf.TARGET_STATE[0])**2 +0.1) - tf.math.sqrt(0.1) - 0.1 + tf.math.sqrt((y_ee[:]-conf.TARGET_STATE[1])**2 +0.1) - tf.math.sqrt(0.1) - 0.1)) + 1)/conf.alpha2

    vel_joint_list = []
    for i in range(BATCH_SIZE):
        if last_ts[i][0] == 1.0:
            vel_joint_list.append(x2_not_norm[i,3]**2 + x2_not_norm[i,4]**2 + x2_not_norm[i,5]**2 - 10000/conf.w_v)
        else:    
            vel_joint_list.append(0)
    vel_joint = tf.cast(tf.stack(vel_joint_list),tf.float32)

    r = (conf.w_d*(-(x_ee[:]-conf.TARGET_STATE[0])**2 -(y_ee[:]-conf.TARGET_STATE[1])**2) + conf.w_peak*peak_reward -conf.w_v*vel_joint - conf.w_ob1*ell1_pen - conf.w_ob2*ell2_pen - conf.w_ob3*ell3_pen - conf.w_u*(u[:,0]**2 + u[:,1]**2 + u[:,2]**2))/100 

    return r 

# Simulate dynamics to get next state
def simulate(dt,state,u):

    # Create robot model in Pinocchio with q_init as initial configuration and v_init as initial velocities
    q_init = np.zeros(int((len(state)-1)/2))
    for state_index in range(int((len(state)-1)/2)):
        q_init[state_index] = state[state_index]
    q_init = q_init.T
    v_init = np.zeros(int((len(state)-1)/2))
    for state_index in range(int((len(state)-1)/2)):
        v_init[state_index] = state[int((len(state)-1)/2)+state_index]
    v_init = v_init.T    
    r = load_urdf(URDF_PATH)
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    simu = RobotSimulator(robot, q_init, v_init, conf.simulation_type, conf.tau_coulomb_max)

    # Simulate control u
    simu.simulate(u, conf.dt, 1)
    q_next, v_next, a_next = np.copy(simu.q), np.copy(simu.v), np.copy(simu.dv)
    q0_next, q1_next, q2_next = q_next[0],q_next[1],q_next[2]
    v0_next, v1_next, v2_next = v_next[0],v_next[1],v_next[2]

    t_next = state[-1] + conf.dt

    return np.array([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next])

# Simulate dynamics using tensors and compute its gradient w.r.t control. Batch-wise computation
def simulate_and_derivative_tf(state,u,BATCH_SIZE):
    
    q0_next, q1_next, q2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
    v0_next, v1_next, v2_next = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
    Fu = np.zeros((conf.BATCH_SIZE,num_states,num_actions))  

    for sample_indx in range(BATCH_SIZE):
        state_np = state[sample_indx]

        # Create robot model in Pinocchio with q_init as initial configuration
        q_init = np.zeros(int((len(state_np)-1)/2))
        for state_index in range(int((len(state_np)-1)/2)):
            q_init[state_index] = state_np[state_index]
        q_init = q_init.T    
        v_init = np.zeros(int((len(state_np)-1)/2))
        for state_index in range(int((len(state_np)-1)/2)):
            v_init[state_index] = state_np[int((len(state_np)-1)/2)+state_index]
        v_init = v_init.T
        r = load_urdf(URDF_PATH)
        robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
        simu = RobotSimulator(robot, q_init, v_init, conf.simulation_type, conf.tau_coulomb_max)

        # Dynamics gradient w.r.t control (1st order euler)
        nq = robot.nq
        nv = robot.nv
        nu = robot.na
        nx = nq+nv
        pin.computeABADerivatives(robot.model, robot.data, q_init, v_init, u[sample_indx])        
        Fu_sample = np.zeros((nx, nu))
        Fu_sample[nv:, :] = robot.data.Minv
        Fu_sample *= conf.dt
        if conf.NORMALIZE_INPUTS:
            Fu_sample *= (1/conf.state_norm_arr[3])

        Fu[sample_indx] = np.vstack((Fu_sample, np.zeros(num_actions)))

        # Simulate control u
        simu.simulate(u[sample_indx], conf.dt, 1)
        q_next, v_next, = np.copy(simu.q), np.copy(simu.v)
        q0_next_sample, q1_next_sample, q2_next_sample = q_next[0],q_next[1],q_next[2]
        v0_next_sample, v1_next_sample, v2_next_sample = v_next[0],v_next[1],v_next[2]
        q0_next[sample_indx] = q0_next_sample
        q1_next[sample_indx] = q1_next_sample
        q2_next[sample_indx] = q2_next_sample
        v0_next[sample_indx] = v0_next_sample
        v1_next[sample_indx] = v1_next_sample
        v2_next[sample_indx] = v2_next_sample

    t_next = state[:,-1] + conf.dt    
    
    if conf.NORMALIZE_INPUTS:
        q0_next = q0_next / conf.state_norm_arr[0]
        q1_next = q1_next / conf.state_norm_arr[1]
        q2_next = q2_next / conf.state_norm_arr[2]
        v0_next = v0_next / conf.state_norm_arr[3]
        v1_next = v1_next / conf.state_norm_arr[4]
        v2_next = v2_next / conf.state_norm_arr[5]
        t_next = 2*t_next/conf.state_norm_arr[-1] -1 

    Fu = tf.convert_to_tensor(Fu,dtype=tf.float32)

    return tf.cast(tf.stack([q0_next,q1_next,q2_next,v0_next,v1_next,v2_next,t_next],1),dtype=tf.float32), Fu

# Plot results from TO and episode to check consistency
def plot_results(tau0,tau1,tau2,x_TO,y_TO,x_RL,y_RL,steps,to=0):

    timesteps = conf.dt*np.arange(steps+1)
    timesteps2 = conf.dt*np.arange(steps)
    fig = plt.figure(figsize=(12,8))
    if to:
        plt.suptitle('TO EXPLORATION: N try = {}'.format(N_try), y=1, fontsize=20)
    else:  
        plt.suptitle('POLICY EXPLORATION: N try = {}'.format(N_try), y=1, fontsize=20)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timesteps, x_TO, 'ro', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y_TO, 'bo', linewidth=1, markersize=1)
    ax1.plot(timesteps, x_RL, 'go', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y_RL, 'ko', linewidth=1, markersize=1)
    ax1.set_title('End-Effector Position',fontsize=20)
    ax1.legend(["x_TO","y_TO","x_RL","y_RL"],fontsize=20)
    ax1.set_xlabel('Time [s]',fontsize=20)
    ax1.set_ylabel('[m]',fontsize=20)    

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(timesteps2, tau0, 'ro', linewidth=1, markersize=1) 
    ax2.plot(timesteps2, tau1, 'bo', linewidth=1, markersize=1) 
    ax2.plot(timesteps2, tau2, 'go', linewidth=1, markersize=1) 
    ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
    ax2.set_xlabel('Time [s]',fontsize=20)
    ax2.set_title('Controls',fontsize=20)

    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.plot(x_TO, y_TO, 'ro', linewidth=1, markersize=1)
    ax3.plot(x_RL, y_RL, 'bo', linewidth=1, markersize=1)
    ax3.legend(['TO','RL'],fontsize=20)
    ax3.plot([x_TO[0]],[y_TO[0]],'ko',markersize=5)
    ax3.add_artist(ell1)
    ax3.add_artist(ell2) 
    ax3.add_artist(ell3) 
    ax3.plot([conf.TARGET_STATE[0]],[conf.TARGET_STATE[1]],'bo',markersize=5)
    ax3.set_xlim(-41, 31)
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Plane',fontsize=20)
    ax3.set_xlabel('X [m]',fontsize=20)
    ax3.set_ylabel('Y [m]',fontsize=20)
    ax3.set_ylim(-35, 35)
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig.tight_layout()
    plt.show()

# Plot policy rollout from a single initial state as well as state and control trajectories
def plot_policy(tau0,tau1,tau2,x,y,steps,n_updates, diff_loc=0, PRETRAIN=0):

    timesteps = conf.dt*np.arange(steps)
    fig = plt.figure(figsize=(12,8))
    plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(N_try,n_updates), y=1, fontsize=20)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timesteps, x, 'ro', linewidth=1, markersize=1) 
    ax1.plot(timesteps, y, 'bo', linewidth=1, markersize=1)
    ax1.set_title('End-Effector Position',fontsize=20)
    ax1.legend(["x","y"],fontsize=20)
    ax1.set_xlabel('Time [s]',fontsize=20)
    ax1.set_ylabel('[m]',fontsize=20)    

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(timesteps, tau0, 'ro', linewidth=1, markersize=1) 
    ax2.plot(timesteps, tau1, 'bo', linewidth=1, markersize=1) 
    ax2.plot(timesteps, tau2, 'go', linewidth=1, markersize=1) 
    ax2.legend(['tau0','tau1','tau2'],fontsize=20) 
    ax2.set_xlabel('Time [s]',fontsize=20)
    ax2.set_title('Controls',fontsize=20)

    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.plot(x, y, 'ro', linewidth=1, markersize=1) 
    ax3.add_artist(ell1)
    ax3.add_artist(ell2) 
    ax3.add_artist(ell3) 
    ax3.plot([conf.x_base],[3*conf.l],'ko',markersize=5)   
    ax3.plot([conf.TARGET_STATE[0]],[conf.TARGET_STATE[1]],'b*',markersize=10) 
    ax3.set_xlim([-41, 31])
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Plane',fontsize=20)
    ax3.set_xlabel('X [m]',fontsize=20)
    ax3.set_ylabel('Y [m]',fontsize=20)
    ax3.set_ylim(-35, 35)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    fig.tight_layout()
    if PRETRAIN:
        plt.savefig(Fig_path+'/PolicyEvaluation_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    else:    
        if diff_loc==0:
            plt.savefig(Fig_path+'/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
        else:
            plt.savefig(Fig_path+'/Actor/PolicyEvaluation_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    plt.clf()
    plt.close(fig)

# Plot only policy rollouts from multiple initial states
def plot_policy_eval(x_list,y_list,n_updates, diff_loc=0, PRETRAIN=0):

    fig = plt.figure(figsize=(12,8))
    plt.suptitle('POLICY: Discrete model, N try = {} N updates = {}'.format(N_try,n_updates), y=1, fontsize=20)
    ell1 = Ellipse((conf.XC1, conf.YC1), conf.A1, conf.B1, 0.0)
    ell1.set_facecolor([30/255, 130/255, 76/255, 1])
    ell2 = Ellipse((conf.XC2, conf.YC2), conf.A2, conf.B2, 0.0)
    ell2.set_facecolor([30/255, 130/255, 76/255, 1])
    ell3 = Ellipse((conf.XC3, conf.YC3), conf.A3, conf.B3, 0.0)
    ell3.set_facecolor([30/255, 130/255, 76/255, 1])

    ax = fig.add_subplot(1, 1, 1)
    for idx in range(len(x_list)):
        if idx == 0:
            ax.plot(x_list[idx], y_list[idx], 'ro', linewidth=1, markersize=1)
        elif idx == 1:
            ax.plot(x_list[idx], y_list[idx], 'bo', linewidth=1, markersize=1)
        elif idx == 2:
            ax.plot(x_list[idx], y_list[idx], 'go', linewidth=1, markersize=1)
        elif idx == 3:
            ax.plot(x_list[idx], y_list[idx], 'co', linewidth=1, markersize=1)
        elif idx == 4:
            ax.plot(x_list[idx], y_list[idx], 'yo', linewidth=1, markersize=1)
        elif idx == 5:
            ax.plot(x_list[idx], y_list[idx], color='tab:orange', marker='o', markersize=1, linestyle='None')
        elif idx == 6:
            ax.plot(x_list[idx], y_list[idx], color='grey', marker='o', markersize=1, linestyle='None')
        elif idx == 7:
            ax.plot(x_list[idx], y_list[idx], color='tab:pink', marker='o', markersize=1, linestyle='None')
        elif idx == 8:
            ax.plot(x_list[idx], y_list[idx], color='lime', marker='o', markersize=1, linestyle='None')            
        ax.plot(x_list[idx][0],y_list[idx][0],'ko',markersize=5)
    ax.plot(conf.TARGET_STATE[0],conf.TARGET_STATE[1],'b*',markersize=10)
    ax.add_artist(ell1)
    ax.add_artist(ell2) 
    ax.add_artist(ell3)     
    ax.set_xlim([-41, 31])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]',fontsize=20)
    ax.set_ylabel('Y [m]',fontsize=20)
    ax.set_ylim(-35, 35)
    ax.grid(True)
    fig.tight_layout()
    if PRETRAIN:
        plt.savefig(Fig_path+'/PolicyEvaluationMultiInit_Pretrain_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    else:    
        if diff_loc==0:
            plt.savefig(Fig_path+'/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
        else:
            plt.savefig(Fig_path+'/Actor/PolicyEvaluationMultiInit_Manipulator3DoF_3OBS_{}_{}'.format(N_try,n_updates))
    plt.clf()
    plt.close(fig)

# Plot rollout of the actor from some initial states. It generates the results and then calls plot_policy() and plot_policy_eval()
def rollout(update_step_cntr, diff_loc=0, PRETRAIN=0):

    init_states_sim = [np.array([math.pi/4,-math.pi/8,-math.pi/8,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,math.pi/8,math.pi/8,0.0,0.0,0.0,0.0]),np.array([math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/2,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-3*math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([-math.pi/4,0.0,0.0,0.0,0.0,0.0,0.0]),np.array([math.pi,0.0,0.0,0.0,0.0,0.0,0.0])]
    tau0_all_sim,tau1_all_sim,tau2_all_sim = [],[],[]
    x_ee_all_sim, y_ee_all_sim = [], []

    for k in range(len(init_states_sim)):
        tau0_arr_sim,tau1_arr_sim,tau2_arr_sim = [],[],[]
        x_ee_arr_sim = [conf.x_base + conf.l*(math.cos(init_states_sim[k][0]) + math.cos(init_states_sim[k][0]+init_states_sim[k][1]) + math.cos(init_states_sim[k][0]+init_states_sim[k][1]+init_states_sim[k][2]))]
        y_ee_arr_sim = [conf.y_base + conf.l*(math.sin(init_states_sim[k][0]) + math.sin(init_states_sim[k][0]+init_states_sim[k][1]) + math.sin(init_states_sim[k][0]+init_states_sim[k][1]+init_states_sim[k][2]))]
        prev_state_sim = np.copy(init_states_sim[k])
        episodic_reward_sim = 0

        for i in range(conf.NSTEPS-1):
            if conf.NORMALIZE_INPUTS:
                prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
            else:
                tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)
            ctrl = actor_model(tf_x_sim)
            ctrl_sim = tf.squeeze(ctrl).numpy()
            next_state_sim = simulate(conf.dt,prev_state_sim,ctrl_sim)
            rwrd_sim = reward(next_state_sim,ctrl_sim)
            episodic_reward_sim += rwrd_sim
            tau0_arr_sim.append(ctrl_sim[0])
            tau1_arr_sim.append(ctrl_sim[1])
            tau2_arr_sim.append(ctrl_sim[2])
            x_ee_arr_sim.append(conf.x_base + conf.l*(math.cos(next_state_sim[0]) + math.cos(next_state_sim[0]+next_state_sim[1]) + math.cos(next_state_sim[0]+next_state_sim[1]+next_state_sim[2])))
            y_ee_arr_sim.append(conf.y_base + conf.l*(math.sin(next_state_sim[0]) + math.sin(next_state_sim[0]+next_state_sim[1]) + math.sin(next_state_sim[0]+next_state_sim[1]+next_state_sim[2])))
            prev_state_sim = np.copy(next_state_sim)

            if i==conf.NSTEPS-2:
                if conf.NORMALIZE_INPUTS:
                    prev_state_sim_norm = prev_state_sim / conf.state_norm_arr
                    prev_state_sim_norm[-1] = 2*prev_state_sim_norm[-1] -1
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim_norm), 0)
                else:
                    tf_x_sim = tf.expand_dims(tf.convert_to_tensor(prev_state_sim), 0)                    
                ctrl = actor_model(tf_x_sim)
                ctrl_sim = tf.squeeze(ctrl).numpy()
                tau0_arr_sim.append(ctrl_sim[0])
                tau1_arr_sim.append(ctrl_sim[1])
                tau2_arr_sim.append(ctrl_sim[2])
        if k==2:
            plot_policy(tau0_arr_sim,tau1_arr_sim,tau2_arr_sim,x_ee_arr_sim,y_ee_arr_sim,conf.NSTEPS,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)
            print("N try = {}: Simulation Return @ N updates = {} ==> {}".format(N_try,update_step_cntr,episodic_reward_sim))

        tau0_all_sim.append(np.copy(tau0_arr_sim))            
        tau1_all_sim.append(np.copy(tau1_arr_sim))            
        tau2_all_sim.append(np.copy(tau2_arr_sim))            
        x_ee_all_sim.append(np.copy(x_ee_arr_sim))            
        y_ee_all_sim.append(np.copy(y_ee_arr_sim))

    plot_policy_eval(x_ee_all_sim,y_ee_all_sim,update_step_cntr, diff_loc=diff_loc, PRETRAIN=PRETRAIN)

    return  tau0_all_sim, tau1_all_sim, tau2_all_sim, x_ee_all_sim, y_ee_all_sim

# Plot returns (not so meaningful given that the initial state, so also the time horizon, of each episode is randomized)
def plot_Return():
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(1, 1, 1)    
    ax.plot(ep_reward_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("N_try = {}".format(N_try))
    ax.grid(True)
    plt.savefig(Fig_path+'/EpReturn_Manipulator3DoF_{}'.format(N_try))
    plt.close()

# Plot average return considering 40 episodes 
def plot_AvgReturn():
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(avg_reward_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg. Return")
    ax.set_title("N_try = {}".format(N_try))
    ax.grid(True)
    plt.savefig(Fig_path+'/AvgReturn_Manipulator3DoF_{}'.format(N_try))
    plt.close()


# To run TF on CPU rather than GPU (seems faster since the NNs are small and some gradients are computed with Pinocchio on CPU --> bottleneck = communication CPU-GPU?)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"     
tf.config.experimental.list_physical_devices('GPU')

tf.random.set_seed(123)                                                     # Set seed for reproducibility

# Set the ticklabel font size globally
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22

URDF_PATH = "urdf/planar_manipulator_3dof.urdf"                             # Path to the urdf file

N_try = 1                                                                   # Number of test run
Fig_path = './Results/Figures/N_try_{}'.format(N_try)
NNs_path = './Results/NNs/N_try_{}'.format(N_try)
Config_path = './Results/Configs/'

# Create folders to store the results and the trained NNs
try:
    os.makedirs(Fig_path)                                                  
except:
    print("N try = {} Figures folder already existing".format(N_try))
    pass
try:
    os.makedirs(Fig_path+'/Actor')                                         
except:
    print("N try = {} Actor folder already existing".format(N_try))
    pass
try:
    os.makedirs(NNs_path)                                                 
except:
    print("N try = {} NNs folder already existing".format(N_try))
    pass
try:
    os.makedirs(Config_path)                                              
except:
    print("N try = {} Configs folder already existing".format(N_try))
    pass

num_states = 7      # Number of states
num_actions = 3     # Number of actions

if __name__ == "__main__":

    # Create actor, critic and target NNs
    actor_model = get_actor()
    critic_model = get_critic()
    target_critic = get_critic()

    # Set initial weights of targets equal to those of actor and critic. Comment if recovering from a stopped training
    target_critic.set_weights(critic_model.get_weights())

    # Uncomment if recovering from a stopped training
    # update_step_counter = 102400
    # NNs_path+"/Manipulator_{}.h5".format(update_step_counter))
    # actor_model.load_weights(NNs_path+"/Manipulator_{}.h5".format(update_step_counter))
    # critic_model.load_weights(NNs_path+"/Manipulator_critic{}.h5".format(update_step_counter))
    # target_critic.load_weights(NNs_path+"/Manipulator_target_critic{}.h5".format(update_step_counter))

    # Set optimizer specifying the learning rates
    if conf.LR_SCHEDULE:
        CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(conf.boundaries_schedule_LR_C, conf.values_schedule_LR_C)   # Piecewise constant decay schedule
        ACTOR_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(conf.boundaries_schedule_LR_A, conf.values_schedule_LR_A)
        critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR_SCHEDULE)
        actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR_SCHEDULE)
    else:
        critic_optimizer = tf.keras.optimizers.Adam(conf.CRITIC_LEARNING_RATE)
        actor_optimizer = tf.keras.optimizers.Adam(conf.ACTOR_LEARNING_RATE)

    # Save config file
    f=open(Config_path+'/Manipulator_config{}.txt'.format(N_try), 'w')
    f.write("conf.NEPISODES = {}, conf.NSTEPS = {}, conf.CRITIC_LEARNING_RATE = {}, conf.ACTOR_LEARNING_RATE = {}, conf.UPDATE_RATE = {}, conf.REPLAY_SIZE = {}, conf.BATCH_SIZE = {}, conf.NH1 = {}, conf.NH2 = {}, conf.dt = {}".format(conf.NEPISODES,conf.NSTEPS,conf.CRITIC_LEARNING_RATE,conf.ACTOR_LEARNING_RATE,conf.UPDATE_RATE,conf.REPLAY_SIZE,conf.BATCH_SIZE,conf.NH1,conf.NH2,conf.dt)+
            "\n"+str(conf.UPDATE_LOOPS)+" updates every {} episodes".format(conf.EP_UPDATE)+
            "\n\nReward = ({}*(-(x_ee-conf.TARGET_STATE[0])**2 -(y_ee-conf.TARGET_STATE[1])**2) + {}*peak_reward - {}*vel_joint - {}*ell1_pen - {}*ell2_pen - {}*ell3_pen - {}*(u[0]**2 + u[1]**2 + u[2]**2))/100, vel_joint = x2[3]**2 + x2[4]**2 + x2[5]**2 - 10000/{} if final step else 0, peak reward = math.log(math.exp({}*-(x_err-0.1 + y_err-0.1)) + 1)/{}, x_err = math.sqrt((x_ee-conf.TARGET_STATE[0])**2 +0.1) - math.sqrt(0.1), y_err = math.sqrt((y_ee-conf.TARGET_STATE[1])**2 +0.1) - math.sqrt(0.1), ell_pen = log(exp({}*-(((x_ee-XC)**2)/((a/2)**2) + ((y_ee-YC)**2)/((conf.B1/2)**2) - 1.0)) + 1)/{}".format(conf.w_d,conf.w_peak,conf.w_v,conf.w_ob1,conf.w_ob2,conf.w_ob3,conf.w_u,conf.w_v,conf.alpha2,conf.alpha2,conf.alpha,conf.alpha)+
            "\n\nBase = [{},{}]".format(conf.x_base,conf.y_base)+", target (conf.TARGET_STATE) = "+str(conf.TARGET_STATE)+
            "\nPrioritized_replay_alpha = "+str(conf.prioritized_replay_alpha)+", conf.prioritized_replay_beta0 = "+str(conf.prioritized_replay_beta0)+", conf.prioritized_replay_eps = "+str(conf.prioritized_replay_eps)+
            "\nActor: kernel_and_bias_regularizer = l1_l2({}), Critic:  kernel_and_bias_regularizer = l1_l2({}) (in each layer)".format(conf.wreg_l1_A,conf.wreg_l2_A,conf.wreg_l1_C,conf.wreg_l2_C)+
            "\nScheduled step LR decay = {}: critic values = {} and boundaries = {}, policy values = {} and boundaries = {}".format(conf.LR_SCHEDULE,conf.values_schedule_LR_C,conf.boundaries_schedule_LR_C,conf.values_schedule_LR_A,conf.boundaries_schedule_LR_A)+
            "\nRandom initial state -> [uniform(-pi,pi), uniform(-pi,pi), uniform(-pi,pi), uniform(-pi/4,pi/4), uniform(-pi/4,pi/4), uniform(-pi/4,pi/4),uniform(0,(NSTEPS_SH-1)*conf.dt)"+ 
            "\nNormalized inputs = {}, q by {} and qdot by {}".format(conf.NORMALIZE_INPUTS,conf.state_norm_arr[0],conf.state_norm_arr[3])+
            "\nEpisodes of critic pretraining = {}".format(conf.EPISODE_CRITIC_PRETRAINING)+
            "\nn-step TD = {} with {} lookahead steps".format(conf.TD_N, conf.nsteps_TD_N))
    f.close()

    # Create training instance and an empty (prioritized) replay buffer
    training = Training(conf.BATCH_SIZE)                                                            
    prioritized_buffer = PrioritizedReplayBuffer(conf.REPLAY_SIZE, alpha=conf.prioritized_replay_alpha)   

    # Lists to store the reward history of each episode and the average reward history of last few episodes
    ep_reward_list = []                                                                                     
    avg_reward_list = []

    # Initialize the counter of the updates. Comment if recovering from a stopped training
    update_step_counter = 0                          

    # START TRAINING
    # If recovering from a stopped training, starting episode must be ((update_step_counter/conf.UPDATE_LOOPS)*conf.EP_UPDATE)+1
    for ep in range(conf.NEPISODES): 
        
        TO = 0                                # Flag to indicate if the TO problem has been solved
        DONE = 0                              # Flag indicating if the episode has terminated
        ep_return = 0                         # Initialize the return
        step_counter = 0                      # Initialize the counter of episode steps

        # START TO PROBLEM 
        while TO==0:

            # Randomize initial state           
            rand_time = random.uniform(0,(conf.NSTEPS-1)*conf.dt)
            prev_state = np.array([random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), 
                        random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4), random.uniform(-math.pi/4,math.pi/4),
                        conf.dt*round(rand_time/conf.dt)])
            if conf.NORMALIZE_INPUTS:
                prev_state_norm = prev_state / conf.state_norm_arr
                prev_state_norm[-1] = 2*prev_state_norm[-1] - 1
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_norm), 0)   
            else:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)         

            # Set the horizon of TO problem / RL episode 
            NSTEPS_SH = conf.NSTEPS - int(round(rand_time/conf.dt))                   
            
            # Lists to store TO state and control trajectories
            tau0_arr,tau1_arr,tau2_arr = [],[],[]
            q0_arr, q1_arr, q2_arr = [prev_state[0]], [prev_state[1]], [prev_state[2]]
            v0_arr, v1_arr, v2_arr = [prev_state[3]], [prev_state[4]], [prev_state[5]]
            t_arr = [prev_state[-1]]
            x_ee_arr = [conf.x_base + conf.l*(math.cos(prev_state[0]) + math.cos(prev_state[0]+prev_state[1]) + math.cos(prev_state[0]+prev_state[1]+prev_state[2]))]
            y_ee_arr = [conf.y_base + conf.l*(math.sin(prev_state[0]) + math.sin(prev_state[0]+prev_state[1]) + math.sin(prev_state[0]+prev_state[1]+prev_state[2]))]
        
            # Actor rollout used to initialize TO state and control variables
            init_TO_states = np.zeros((num_states, NSTEPS_SH+1))
            init_TO_states[0][0] = prev_state[0]
            init_TO_states[1][0] = prev_state[1]
            init_TO_states[2][0] = prev_state[2]
            init_TO_states[3][0] = prev_state[3]                     
            init_TO_states[4][0] = prev_state[4]                     
            init_TO_states[5][0] = prev_state[5]                     
            init_TO_controls = np.zeros((num_actions, NSTEPS_SH+1))
            init_TO_controls[0][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[0]
            init_TO_controls[1][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[1] 
            init_TO_controls[2][0] = tf.squeeze(actor_model(tf_prev_state)).numpy()[2] 
            init_prev_state = np.copy(prev_state)

            # Simulate actor's actions to compute the state trajectory used to initialize TO state variables
            for i in range(1, NSTEPS_SH+1):                                                                                                                                 
                init_next_state =  simulate(conf.dt,init_prev_state,np.array([init_TO_controls[0][i-1],init_TO_controls[1][i-1],init_TO_controls[2][i-1]]))
                init_TO_states[0][i] = init_next_state[0]
                init_TO_states[1][i] = init_next_state[1]
                init_TO_states[2][i] = init_next_state[2]
                init_TO_states[3][i] = init_next_state[3] 
                init_TO_states[4][i] = init_next_state[4] 
                init_TO_states[5][i] = init_next_state[5] 
                if conf.NORMALIZE_INPUTS:
                    init_next_state_norm = init_next_state / conf.state_norm_arr
                    init_next_state_norm[-1] = 2*init_next_state_norm[-1] - 1
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state_norm), 0)        
                else:    
                    init_tf_next_state = tf.expand_dims(tf.convert_to_tensor(init_next_state), 0)        
                init_TO_controls[0][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[0]
                init_TO_controls[1][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[1] 
                init_TO_controls[2][i] = tf.squeeze(actor_model(init_tf_next_state)).numpy()[2] 
                init_prev_state = np.copy(init_next_state)

            # Create TO problem
            if ep < conf.EPISODE_CRITIC_PRETRAINING or ep < conf.EPISODE_ICS_INIT:
                TO_mdl = TO_manipulator(prev_state, init_q0_ICS, init_q1_ICS, init_q2_ICS, init_v0_ICS, init_v1_ICS, init_v2_ICS, init_0, init_0, init_0, init_0, init_0, init_0, NSTEPS_SH)
            else:
                if ep == conf.EPISODE_ICS_INIT and conf.LR_SCHEDULE:  
                    # Re-initialize Adam otherwise it keeps being affected by the estimates of first-order and second-order moments computed previously with ICS warm-starting
                    critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR_SCHEDULE)     
                    actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR_SCHEDULE)
                TO_mdl = TO_manipulator(prev_state, init_q0, init_q1, init_q2, init_v0, init_v1, init_v2, init_0, init_0, init_0, init_tau0, init_tau1, init_tau2, NSTEPS_SH, init_TO = [init_TO_controls, init_TO_states])
                
            # Indexes of TO variables       
            K = np.array([k for k in TO_mdl.k])               

            # Select solver
            solver = SolverFactory('ipopt')
            solver.options['linear_solver'] = "ma57"

            # Try to solve TO problem
            try:
                results = solver.solve(TO_mdl)                              
                if str(results.solver.termination_condition) == "optimal":    
                    #Retrieve control trajectory
                    tau0_TO = [TO_mdl.tau0[k]() for k in K]
                    tau1_TO = [TO_mdl.tau1[k]() for k in K]
                    tau2_TO = [TO_mdl.tau2[k]() for k in K]
                    TO = 1
                else:
                    print('TO solution not optimal')                   
                    raise Exception()         
            except:
                print("*** TO failed ***")                                 
                
            # Plot TO solution    
            # plot_results_TO(TO_mdl)                                                               

        ep_arr = []

        # START RL EPISODE
        while True:
            # Get current and next TO actions
            action, next_TO_action = get_TO_actions(step_counter, action0_TO=tau0_TO, action1_TO=tau1_TO, action2_TO=tau2_TO)              
                                                                                                                            
            # Simulate actions and retrieve next state
            next_state = simulate(conf.dt,prev_state,action)

            # Store performed action and next state
            tau0_arr.append(action[0])
            tau1_arr.append(action[1])
            tau2_arr.append(action[2])
            q0_arr.append(next_state[0])
            q1_arr.append(next_state[1])
            q2_arr.append(next_state[2])
            v0_arr.append(next_state[3])
            v1_arr.append(next_state[4])
            v2_arr.append(next_state[5])
            t_arr.append(next_state[-1])
            x_ee_arr.append(conf.x_base + conf.l*(math.cos(q0_arr[-1]) + math.cos(q0_arr[-1]+q1_arr[-1]) + math.cos(q0_arr[-1]+q1_arr[-1]+q2_arr[-1])))
            y_ee_arr.append(conf.y_base + conf.l*(math.sin(q0_arr[-1]) + math.sin(q0_arr[-1]+q1_arr[-1]) + math.sin(q0_arr[-1]+q1_arr[-1]+q2_arr[-1])))
            
            # Compute reward
            rwrd = reward(next_state, action)
            ep_arr.append(rwrd)

            if step_counter==NSTEPS_SH-1:
                DONE = 1

            # Store transition if you want to use 1-step TD
            if conf.TD_N==0:
                prioritized_buffer.add(prev_state, next_TO_action, action, rwrd, next_state, float(DONE))

            ep_return += rwrd                       # Increment the episodic return by the reward just recived
            step_counter += 1                       
            prev_state = np.copy(next_state)        # Next state becomes the prev state at the next episode step

            # End episode
            if DONE == 1:
                break
        
        # Plot the state and control trajectories of this episode 
        # plot_results(tau0_arr,tau1_arr,tau2_arr,x_ee,y_ee,x_ee_arr,y_ee_arr,NSTEPS_SH,to=TO)

        # Store transition after computing the (partial) cost-to go when using n-step TD (up to Monte Carlo)
        if conf.TD_N:
            DONE = 0
            cost_to_go_arr = []
            for i in range(len(ep_arr)):                
                final_i = min(i+conf.nsteps_TD_N,len(ep_arr))
                if final_i == len(ep_arr):
                    V_final = 0.0
                else:
                    if conf.NORMALIZE_INPUTS:
                        next_state_rollout = np.array([q0_arr[final_i],q1_arr[final_i],q2_arr[final_i],v0_arr[final_i],v1_arr[final_i],v2_arr[final_i],t_arr[final_i]]) / conf.state_norm_arr
                        next_state_rollout[-1] = 2*next_state_rollout[-1] - 1
                    else:
                        next_state_rollout = np.array([q0_arr[final_i],q1_arr[final_i],q2_arr[final_i],v0_arr[final_i],v1_arr[final_i],v2_arr[final_i],t_arr[final_i]]) / conf.state_norm_arr
                    tf_next_state_rollout = tf.expand_dims(tf.convert_to_tensor(next_state_rollout), 0)
                    V_final = target_critic(tf_next_state_rollout, training=False).numpy()[0][0]
                cost_to_go = sum(ep_arr[i:final_i+1]) + V_final
                cost_to_go_arr.append(np.float32(cost_to_go))
                if i == len(ep_arr)-1:
                    DONE = 1 
                prioritized_buffer.add(np.array([q0_arr[i],q1_arr[i],q2_arr[i],v0_arr[i],v1_arr[i],v2_arr[i],t_arr[i]]), next_TO_action, action, cost_to_go_arr[i], np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0]), float(DONE))

        # Update the NNs
        if ep % conf.EP_UPDATE == 0:
            for i in range(conf.UPDATE_LOOPS):
                training.learn(ep, prioritized_buffer)                                         # Update critic and actor
                update_target(target_critic.variables, critic_model.variables, conf.UPDATE_RATE)    # Update target critic
                update_step_counter += 1

        # Plot rollouts every 0.5% of the training (saved in a separate folder)
        if ep>=conf.ep_no_update and ep%int((conf.NEPISODES-conf.ep_no_update)/200)==0:
            _, _, _, _, _ = rollout(update_step_counter, diff_loc=1)            

        # Plot rollouts and save the NNs every 5% of the training
        if ep>=conf.ep_no_update and int((conf.NEPISODES-conf.ep_no_update)/20)!=1 and int((conf.NEPISODES-conf.ep_no_update)/20)!=0 and ep%int((conf.NEPISODES-conf.ep_no_update)/20)==0:          
            _, _, _, _, _ = rollout(update_step_counter)        
            actor_model.save_weights(NNs_path+"/Manipulator_{}.h5".format(update_step_counter))
            critic_model.save_weights(NNs_path+"/Manipulator_critic_{}.h5".format(update_step_counter))
            target_critic.save_weights(NNs_path+"/Manipulator_target_critic_{}.h5".format(update_step_counter))
                
        # ep_reward_list.append(ep_return)
        # avg_reward = np.mean(ep_reward_list[-40:])  # Mean of last 40 episodes
        # avg_reward_list.append(avg_reward)

        print("Episode  {}  --->   Return = {}".format(ep, ep_return))

    # Plot returns
    # plot_AvgReturn()
    # plot_Return()

    # Save networks at the end of the training
    actor_model.save_weights(NNs_path+"/Manipulator_actor_final.h5")
    critic_model.save_weights(NNs_path+"/Manipulator_critic_final.h5")
    target_critic.save_weights(NNs_path+"/Manipulator_target_critic_final.h5")

    # Simulate the final policy
    tau0_all_final_sim, tau1_all_final_sim, tau2_all_final_sim, x_ee_all_final_sim, y_ee_all_final_sim = rollout(update_step_counter)