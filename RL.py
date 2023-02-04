import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
from CACTO import CACTO
import numpy as np
import math
import time
#import manipulator_conf as conf #only for system parameters (M, l, Iz, ...)

class RL_AC(CACTO):
    def __init__(self, env, conf):
        super(RL_AC, self).__init__(env, conf, init_setup_model=False)

        self.TD_N = conf.TD_N
        self.nsteps_TD_N = conf.nsteps_TD_N
        self.prioritized_replay_eps = conf.prioritized_replay_eps
        self.prioritized_replay_alpha = conf.prioritized_replay_alpha
        self.state_norm_arr = conf.state_norm_arr
        self.UPDATE_LOOPS = conf.UPDATE_LOOPS
        self.UPDATE_RATE = conf.UPDATE_RATE

        self.SOBOLEV = 1

        return

    # Update both critic and actor                                                                     
    def update(self, episode, red_state_batch_norm, red_state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, action_batch, weights_batch=None):
        with tf.GradientTape() as tape:
            # Update the critic
            if self.SOBOLEV:
                with tf.GradientTape() as tape3:
                    tape3.watch(state_batch)
                    next_state_tf = self.env.simulate_tf(state_batch,action_batch)      
                    rewards_tf = self.env.reward_tf(next_state_tf, action_batch, self.batch_size, d_batch)    # We need also the batch of rewards associated to the batches of next states and actions
                    target_values = CACTO.target_critic(next_state_tf, training=True)
                    y = rewards_tf + (1-d_batch)*(target_values)
                    target_grad = tape3.gradient(y, state_batch)
                if self.NORMALIZE_INPUTS:
                    with tf.GradientTape() as tape2:
                        tape2.watch(state_batch_norm)
                        critic_value = CACTO.critic_model(state_batch_norm, training=True)                    # Compute batch of Values associated to the sampled batch of states
                        critic_state_grad = tape2.gradient(critic_value, state_batch_norm)
                else:
                    with tf.GradientTape() as tape2:
                        tape2.watch(state_batch)
                        critic_value = CACTO.critic_model(state_batch, training=True)                         # Compute batch of Values associated to the sampled batch of states
                        critic_state_grad = tape2.gradient(critic_value, state_batch)
            else:
                if self.TD_N: ### !!! ### 
                    y = reward_batch                                                                           # When using n-step TD, reward_batch is the batch of costs-to-go and not the batch of single step rewards
                else:    
                    if self.NORMALIZE_INPUTS:
                        target_values = CACTO.target_critic(next_state_batch_norm, training=True)              # Compute Value at next state after self.nsteps_TD_N steps given by target critic 
                        critic_value = CACTO.critic_model(state_batch_norm, training=True)                     # Compute batch of Values associated to the sampled batch of states
                    else:
                        target_values = CACTO.target_critic(next_state_batch, training=True)
                        critic_value = CACTO.critic_model(state_batch, training=True)
                    y = reward_batch + (1-d_batch)*target_values                                               # Compute batch of 1-step targets for the critic loss 

            if weights_batch is None:
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))                            # Critic loss function (tf.math.reduce_mean computes the mean of elements across dimensions of a tensor, in this case across the batch)
            else:
                if self.SOBOLEV:
                    der = tf.math.reduce_mean(tf.norm(critic_state_grad-target_grad, axis=1)) 
                    critic_loss = tf.math.reduce_mean(tf.math.square((y - critic_value))) + 5e-1*np.clip(der,0,1e5)      # tf.math.reduce_mean computes the mean of elements across dimensions of a tensor
                else:
                    critic_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(weights_batch,(y - critic_value))))   

        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, CACTO.critic_model.trainable_variables)   

        # Update the critic backpropagating the gradients
        CACTO.critic_optimizer.apply_gradients(zip(critic_grad, CACTO.critic_model.trainable_variables))

        # Update the actor after the critic pre-training
        if episode >= self.EPISODE_CRITIC_PRETRAINING:    
            if self.NORMALIZE_INPUTS:
                actions = CACTO.actor_model(red_state_batch_norm, training=True)
            else:
                actions = CACTO.actor_model(red_state_batch, training=True)
            # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
            next_state_tf, ds_next_da = self.env.simulate_and_derivative_tf(red_state_batch,actions.numpy(),self.batch_size)
            with tf.GradientTape() as tape:
                tape.watch(next_state_tf)
                critic_value_next = CACTO.critic_model(next_state_tf, training=True)                                    # next_state_batch = next state after applying policy's action, already normalized if self.NORMALIZE_INPUTS=1
            dV_ds_next = tape.gradient(critic_value_next, next_state_tf)                                                # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)   
            with tf.GradientTape() as tape1:    
                with tf.GradientTape() as tape2:
                        tape1.watch(actions)
                        tape2.watch(next_state_tf)
                        rewards_tf = self.env.reward_tf(next_state_tf, actions, self.batch_size, d_batch)
            dr_da = tape1.gradient(rewards_tf,actions)                                                                  # dr_da = gradient of reward r w.r.t. policy's action a
            dr_ds_next = tape2.gradient(rewards_tf,next_state_tf)                                                       # dr_ds' = gradient of reward r w.r.t. next state s' after performing policy's action a
            dr_ds_next_dV_ds_next_reshaped = tf.reshape(dr_ds_next+dV_ds_next,(self.batch_size,1,self.nb_state))        # dr_ds' + dV_ds'
            dr_ds_next_dV_ds_next = tf.matmul(dr_ds_next_dV_ds_next_reshaped,ds_next_da)                                # (dr_ds' + dV_ds')*ds'_da
            dr_da_reshaped = tf.reshape(dr_da,(self.batch_size,1,self.nb_action))                                       
            tf_sum = dr_ds_next_dV_ds_next + dr_da_reshaped                                                             # (dr_ds' + dV_ds')*ds'_da + dr_da
            # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a 
            # and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
            with tf.GradientTape() as tape:
                tape.watch(CACTO.actor_model.trainable_variables)
                if self.NORMALIZE_INPUTS:
                    actions = CACTO.actor_model(state_batch_norm, training=True)
                else:
                    actions = CACTO.actor_model(state_batch, training=True)
                actions_reshaped = tf.reshape(actions,(self.batch_size,self.nb_action,1))
                tf_sum_reshaped = tf.reshape(tf_sum,(self.batch_size,1,self.nb_action))    
                Q_neg = tf.matmul(-tf_sum_reshaped,actions_reshaped) 
                
                mean_Qneg = tf.math.reduce_mean(Q_neg)                                                           # Also here we need a scalar so we compute the mean -Q across the batch
            # Gradients of the actor loss w.r.t. actor's parameters
            actor_grad = tape.gradient(mean_Qneg, CACTO.actor_model.trainable_variables) 
            # Update the actor backpropagating the gradients
            CACTO.actor_optimizer.apply_gradients(zip(actor_grad, CACTO.actor_model.trainable_variables))

    @tf.function 
    def update_target(self,target_weights, weights, tau): 
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))  

    def learn(self, episode, prioritized_buffer):
        # Sample batch of transitions from the buffer
        experience = prioritized_buffer.sample(self.batch_size, beta=self.prioritized_replay_eps)            # Bias annealing not performed, that's why beta is equal to a very small number (0 not accepted by PrioritizedReplayBuffer)
        x_batch, a_next_batch, a_batch, r_batch, x2_batch, d_batch, weights, batch_idxes = experience        # Importance sampling weights (actually not used) should anneal the bias (see Prioritized Experience Replay paper) 
        
        # Convert batch of transitions into a tensor
        x_batch = x_batch.reshape(self.batch_size,self.nb_state)
        a_next_batch = a_next_batch.reshape(self.batch_size,self.nb_action)
        a_batch = a_batch.reshape(self.batch_size,self.nb_action)
        r_batch = r_batch.reshape(self.batch_size,1)
        x2_batch = x2_batch.reshape(self.batch_size,self.nb_state)
        d_batch = d_batch.reshape(self.batch_size,1)
        weights = weights.reshape(self.batch_size,1)

        state_batch = tf.convert_to_tensor(x_batch)
        action_batch = tf.convert_to_tensor(a_batch)
        reward_batch = tf.convert_to_tensor(r_batch)
        next_state_batch = tf.convert_to_tensor(x2_batch)

        state_batch = tf.cast(state_batch, dtype=tf.float32)
        action_batch = tf.cast(action_batch, dtype=tf.float32)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)                                     
        d_batch = tf.cast(d_batch, dtype=tf.float32)
        next_state_batch = tf.cast(next_state_batch, dtype=tf.float32)  

        if self.prioritized_replay_alpha == 0:
            weights_batch = tf.convert_to_tensor(np.ones_like(weights), dtype=tf.float32)
        else:    
            weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)
    
        if self.NORMALIZE_INPUTS:
            x_batch_norm = x_batch / self.state_norm_arr
            x_batch_norm[:,-1] = 2*x_batch_norm[:,-1] -1
            state_batch_norm = tf.convert_to_tensor(x_batch_norm)
            state_batch_norm = tf.cast(state_batch_norm, dtype=tf.float32)

            x2_batch_norm = x2_batch / self.state_norm_arr
            x2_batch_norm[:,-1] = 2*x2_batch_norm[:,-1] -1
            next_state_batch_norm = tf.convert_to_tensor(x2_batch_norm)
            next_state_batch_norm = tf.cast(next_state_batch_norm, dtype=tf.float32)
          
        # Update priorities
        if self.prioritized_replay_alpha != 0:
            if self.NORMALIZE_INPUTS:
                v_batch = CACTO.critic_model(state_batch_norm, training=True)                      # Compute batch of Values associated to the sampled batch ofstates
                v2_batch = CACTO.target_critic(next_state_batch_norm, training=True)               # Compute batch of Values from target critic associated to sampled batch of next states
                vref_batch = reward_batch + (1-d_batch)*(v2_batch)                           # Compute the targets for the TD error         
            else:
                v_batch = CACTO.critic_model(state_batch, training=True) 
                v2_batch = CACTO.target_critic(next_state_batch, training=True)                     
                vref_batch = reward_batch + (1-d_batch)*(v2_batch)                          

            td_errors = tf.math.abs(tf.math.subtract(vref_batch,v_batch))                           
            new_priorities = td_errors.numpy() + self.prioritized_replay_eps                      # Proportional prioritization where p_i = |TD_error_i| + self.prioritized_replay_eps 
            prioritized_buffer.update_priorities(batch_idxes, new_priorities)   

        # Update NNs
        self.update(episode, state_batch_norm, state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, action_batch, weights_batch=weights_batch)

    def RL_Solve(self, prev_state, ep, rand_time, env, tau_TO, prioritized_buffer):
        DONE = 0                              # Flag indicating if the episode has terminated
        ep_return = 0                         # Initialize the return
        step_counter = 0                      # Initialize the counter of episode steps
        ep_arr = []

        # START RL EPISODE
        while True:
            # Get current and next TO actions
            action, next_TO_action = self.get_TO_actions(step_counter, action_TO=tau_TO)           

            # Simulate actions and retrieve next state and compute reward
            next_state, rwrd = env.step(rand_time, prev_state, action)
            
            # Store performed action and next state and reward
            CACTO.control_arr = np.vstack([CACTO.control_arr, action.reshape(1,self.nb_action)])
            CACTO.state_arr = np.vstack([CACTO.state_arr, next_state.reshape(1,self.nb_state)])
            
            ###
            CACTO.x_ee_arr.append(self.env.get_end_effector_position(CACTO.state_arr[-1,:])[0])
            CACTO.y_ee_arr.append(self.env.get_end_effector_position(CACTO.state_arr[-1,:])[1])
            ###

            ep_arr.append(rwrd)

            if step_counter==CACTO.NSTEPS_SH-1:
                DONE = 1

            # Store transition if you want to use 1-step TD
            if self.TD_N==0:
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
        if self.TD_N:
            DONE = 0
            cost_to_go_arr = []
            for i in range(len(ep_arr)):                
                final_i = min(i+self.nsteps_TD_N,len(ep_arr))
                if final_i == len(ep_arr):
                    V_final = 0.0
                else:
                    if self.NORMALIZE_INPUTS:
                        next_state_rollout = CACTO.state_arr[final_i,:] / self.state_norm_arr
                        next_state_rollout[-1] = 2*next_state_rollout[-1] - 1
                    else:
                        next_state_rollout = CACTO.state_arr[final_i,:] / self.state_norm_arr
                    tf_next_state_rollout = tf.expand_dims(tf.convert_to_tensor(next_state_rollout), 0)
                    V_final = CACTO.target_critic(tf_next_state_rollout, training=False).numpy()[0][0]
                cost_to_go = sum(ep_arr[i:final_i+1]) + V_final
                cost_to_go_arr.append(np.float32(cost_to_go))
                if i == len(ep_arr)-1:
                    DONE = 1 
                prioritized_buffer.add(CACTO.state_arr[i,:], next_TO_action, action, cost_to_go_arr[i], np.zeros(self.nb_state), float(DONE)) 
        
        # Update the NNs
        for i in range(self.UPDATE_LOOPS):
            self.learn(ep, prioritized_buffer)                                                       # Update critic and actor
            self.update_target(CACTO.target_critic.variables, CACTO.critic_model.variables, self.UPDATE_RATE)    # Update target critic
            self.update_step_counter += 1
        return self.update_step_counter, ep_return

    # Get optimal actions at timestep "step" in the current episode
    def get_TO_actions(self, step, action_TO=None):

        actions = action_TO[step,:]
 
        # Buond actions in case they are not already bounded in the TO problem
        if step < CACTO.NSTEPS_SH:
            next_actions = np.clip(action_TO[step+1,:], self.tau_lower_bound, self.tau_upper_bound)
        else:
            next_actions = np.copy(actions) # Actions at last step of the episode are not performed 

        return actions, next_actions



    def sanity_check(self, state_batch_norm, grad, y, N_TESTS=1):
        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        eps = 1e-3
        y_eps = np.copy(np.array(state_batch_norm))
        grad_fd = np.zeros_like(state_batch_norm)
        for j in range(y_eps.shape[0]):
            for k in range(y_eps.shape[1]):
                y_eps[j,k] += eps

                cost_fd = CACTO.critic_model(tf.cast(y_eps, tf.float32), training=True)
                grad_fd[j,k] = (cost_fd[j] - y[j]) / eps

                y_eps[j,k] = state_batch_norm[j,k]

        tr_min = 0
        for jj in range(y_eps.shape[0]):
            for kk in range(y_eps.shape[1]):
                if np.abs(grad_fd[jj,kk]) < 1e-6:
                    grad_err = 1e-10
                    print('too small', grad[jj,kk], grad_fd[jj,kk])
                else:    
                    grad_err = np.abs((grad[jj,kk]-grad_fd[jj,kk])/(grad_fd[jj,kk]))
                if grad_err > tr_min:
                    tr_min=grad_err
                    print(np.array(grad[jj,kk]), grad_fd[jj,kk])
            
        if(tr_min>1e-2): 
            print('Gradient 1 computation is NOT correct. Relative error=', tr_min) #
            time.sleep(5)
        else:
            print('Everything 1 is fine ', tr_min)


    def sanity_check2(self, state_batch, action_batch, d_batch, grad, y, N_TESTS=1): 

        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        eps = 1e-2
        y_eps = np.copy(np.array(state_batch))

        grad_fd = np.zeros_like(state_batch)
        for j in range(y_eps.shape[0]):
            for k in range(y_eps.shape[1]):
                y_eps[j,k] += eps
  
                next_state_tf = self.env.simulate_tf(tf.cast(y_eps, tf.float32),action_batch)       
                rewards_tf = self.env.reward_tf(next_state_tf, action_batch, self.batch_size, d_batch)                           # We need also the batch of rewards associated to the batches of next states and actions
                target_values = CACTO.target_critic(next_state_tf, training=True)
                y_fd = rewards_tf + (1-d_batch)*(target_values)
                grad_fd[j,k] = (y_fd[j] - y[j]) / eps
                
                y_eps[j,k] = state_batch[j,k]

        tr_min = 0
        for jj in range(y_eps.shape[0]):
            for kk in range(y_eps.shape[1]):
                if np.abs(grad_fd[jj,kk]) < 1e-6:
                    grad_err = 1e-10
                    print('too small', grad[jj,kk], grad_fd[jj,kk])
                else:    
                    grad_err = np.abs((grad[jj,kk]-grad_fd[jj,kk])/(grad_fd[jj,kk]))
                if grad_err > tr_min:
                    tr_min=grad_err
                    print(np.array(grad[jj,kk]), grad_fd[jj,kk])
    
            
        if(tr_min>1e-2): 
            print('Gradient 2 computation is NOT correct. Relative error=', tr_min) #
        else:
            print('Everything 2 is fine ', tr_min)




'''
if self.SOBOLEV:
                with tf.GradientTape() as tape3:
                    tape3.watch(state_batch)
                    #x2_time = tf.stack([np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), np.zeros(self.batch_size), (state_batch_norm[:,-1]+1)*self.state_norm_arr[-1]/2],1)
                    #x2 = state_batch_norm * self.state_norm_arr
                    #mask = tf.cast(tf.stack([np.ones(self.batch_size), np.ones(self.batch_size), np.ones(self.batch_size), np.ones(self.batch_size), np.ones(self.batch_size), np.ones(self.batch_size), np.zeros(self.batch_size)],1),tf.float32)
                    #x2 = x2 * mask + x2_time * (1 - mask)
                    next_state_tf = self.env.simulate_tf(state_batch,action_batch)      
                    rewards_tf = self.env.reward_tf(next_state_tf, action_batch, self.batch_size, d_batch)                           # We need also the batch of rewards associated to the batches of next states and actions
                    target_values = CACTO.target_critic(next_state_tf, training=True)
                    y = rewards_tf + (1-d_batch)*(target_values)
                    target_grad = tape3.gradient(y, state_batch)
            else:
                if self.TD_N:
                    y = reward_batch                                                        # When using n-step TD, reward_batch is the batch of costs-to-go and not the batch of single step rewards
                else:    
                    if self.NORMALIZE_INPUTS:
                        target_values = CACTO.target_critic(next_state_batch_norm, training=True) # Compute Value at next state after self.nsteps_TD_N steps given by target critic 
                    else:
                        target_values = CACTO.target_critic(next_state_batch, training=True)
                    y = reward_batch + (1-d_batch)*target_values                            # Compute batch of 1-step targets for the critic loss 

            if self.NORMALIZE_INPUTS:
                if self.SOBOLEV:
                    with tf.GradientTape() as tape2:
                        tape2.watch(state_batch_norm)
                        critic_value = CACTO.critic_model(state_batch_norm, training=True)                                 # Compute batch of Values associated to the sampled batch of states
                        critic_state_grad = tape2.gradient(critic_value, state_batch_norm)
                else:
                    critic_value = CACTO.critic_model(state_batch_norm, training=True)            # Compute batch of Values associated to the sampled batch of states
            else:
                if self.SOBOLEV:
                    with tf.GradientTape() as tape2:
                        tape2.watch(state_batch)
                        critic_value = CACTO.critic_model(state_batch, training=True)                                 # Compute batch of Values associated to the sampled batch of states
                        grad = tape2.gradient(critic_value, state_batch)
                        critic_state_grad = tf.norm(grad,axis=1) ### ??? ###
                else:
                    critic_value = CACTO.critic_model(state_batch, training=True)
'''