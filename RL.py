import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
from CACTO import CACTO
import numpy as np
import math
import manipulator_conf as conf #only for system parameters (M, l, Iz, ...)

class RL_AC(CACTO):
    def __init__(self, env, NORMALIZE_INPUTS, EPISODE_CRITIC_PRETRAINING, tau_upper_bound, 
                 tau_lower_bound, dt, system_param, TO_method, system, nsteps_TD_N, soft_max_param, obs_param, weight, 
                 target, NSTEPS, EPISODE_ICS_INIT, TD_N, prioritized_replay_eps,prioritized_replay_alpha,
                 state_norm_arr, UPDATE_LOOPS, UPDATE_RATE, LR_SCHEDULE, update_step_counter, 
                 NH1, NH2, wreg_l1_A, wreg_l2_A, wreg_l1_C, wreg_l2_C, boundaries_schedule_LR_C,
                 values_schedule_LR_C, boundaries_schedule_LR_A, values_schedule_LR_A, 
                 CRITIC_LEARNING_RATE, ACTOR_LEARNING_RATE,nb_state, nb_action, robot,
                 recover_stopped_training, NNs_path, batch_size):
        super(RL_AC, self).__init__(env, NORMALIZE_INPUTS, EPISODE_CRITIC_PRETRAINING, tau_upper_bound, 
                 tau_lower_bound, dt, system_param, TO_method, system, nsteps_TD_N, soft_max_param, obs_param, weight, 
                 target, NSTEPS, EPISODE_ICS_INIT, TD_N, prioritized_replay_eps,prioritized_replay_alpha,
                 state_norm_arr, UPDATE_LOOPS, UPDATE_RATE, LR_SCHEDULE, update_step_counter, 
                 NH1, NH2, wreg_l1_A, wreg_l2_A, wreg_l1_C, wreg_l2_C, boundaries_schedule_LR_C,
                 values_schedule_LR_C, boundaries_schedule_LR_A, values_schedule_LR_A, 
                 CRITIC_LEARNING_RATE, ACTOR_LEARNING_RATE,nb_state, nb_action, robot,
                 recover_stopped_training, NNs_path, batch_size,init_setup_model=False)

        self.TD_N = TD_N
        self.nsteps_TD_N = nsteps_TD_N
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.state_norm_arr = state_norm_arr
        self.UPDATE_LOOPS = UPDATE_LOOPS
        self.UPDATE_RATE = UPDATE_RATE

        return

    # Update both critic and actor                                                                     
    def update(self, episode, red_state_batch_norm, red_state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, weights_batch=None):
        with tf.GradientTape() as tape:
            # Update the critic
            if self.TD_N:
                y = reward_batch                                                        # When using n-step TD, reward_batch is the batch of costs-to-go and not the batch of single step rewards
            else:    
                if self.NORMALIZE_INPUTS:
                    target_values = CACTO.target_critic(next_state_batch_norm, training=True) # Compute Value at next state after self.nsteps_TD_N steps given by target critic 
                else:
                    target_values = CACTO.target_critic(next_state_batch, training=True)
                y = reward_batch + (1-d_batch)*target_values                            # Compute batch of 1-step targets for the critic loss 
            
            if self.NORMALIZE_INPUTS:
                critic_value = CACTO.critic_model(state_batch_norm, training=True)            # Compute batch of Values associated to the sampled batch of states
            else:
                critic_value = CACTO.critic_model(state_batch, training=True)

            if weights_batch is None:
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))     # Critic loss function (tf.math.reduce_mean computes the mean of elements across dimensions of a tensor, in this case across the batch)
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
                critic_value_next = CACTO.critic_model(next_state_tf, training=True)                                 # next_state_batch = next state after applying policy's action, already normalized if self.NORMALIZE_INPUTS=1
            dV_ds_next = tape.gradient(critic_value_next, next_state_tf)                                       # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)   

            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                        tape1.watch(actions)
                        tape2.watch(next_state_tf)
                        rewards_tf = self.env.reward_tf(next_state_tf, actions, self.batch_size, d_batch)
            dr_da = tape1.gradient(rewards_tf,actions)                                                          # dr_da = gradient of reward r w.r.t. policy's action a
            dr_ds_next = tape2.gradient(rewards_tf,next_state_tf)                                               # dr_ds' = gradient of reward r w.r.t. next state s' after performing policy's action a

            dr_ds_next_dV_ds_next_reshaped = tf.reshape(dr_ds_next+dV_ds_next,(self.batch_size,1,self.nb_state))   # dr_ds' + dV_ds'
            dr_ds_next_dV_ds_next = tf.matmul(dr_ds_next_dV_ds_next_reshaped,ds_next_da)                        # (dr_ds' + dV_ds')*ds'_da
            dr_da_reshaped = tf.reshape(dr_da,(self.batch_size,1,self.nb_action))                               
            tf_sum = dr_ds_next_dV_ds_next + dr_da_reshaped                                                     # (dr_ds' + dV_ds')*ds'_da + dr_da

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
        self.update(episode, state_batch_norm, state_batch, state_batch_norm, state_batch, reward_batch, next_state_batch, next_state_batch_norm, d_batch, weights_batch=weights_batch)

    def RL_Manipulator_Solve(self, prev_state, ep, rand_time, env, tau0_TO, tau1_TO, tau2_TO, prioritized_buffer):
        DONE = 0                              # Flag indicating if the episode has terminated
        ep_return = 0                         # Initialize the return
        step_counter = 0                      # Initialize the counter of episode steps
        ep_arr = []

        # START RL EPISODE
        while True:
            # Get current and next TO actions
            action, next_TO_action = self.get_TO_actions(step_counter, action0_TO=tau0_TO, action1_TO=tau1_TO, action2_TO=tau2_TO)              

            # Simulate actions and retrieve next state and compute reward
            next_state, rwrd = env.step(rand_time, prev_state, action)
            
            # Store performed action and next state and reward
            CACTO.tau0_arr.append(action[0])
            CACTO.tau1_arr.append(action[1])
            CACTO.tau2_arr.append(action[2])
            CACTO.q0_arr.append(next_state[0])
            CACTO.q1_arr.append(next_state[1])
            CACTO.q2_arr.append(next_state[2])
            CACTO.v0_arr.append(next_state[3])
            CACTO.v1_arr.append(next_state[4])
            CACTO.v2_arr.append(next_state[5])
            CACTO.t_arr.append(next_state[-1])
            CACTO.x_ee_arr.append(conf.x_base + conf.l*(math.cos(CACTO.q0_arr[-1]) + math.cos(CACTO.q0_arr[-1]+CACTO.q1_arr[-1]) + math.cos(CACTO.q0_arr[-1]+CACTO.q1_arr[-1]+CACTO.q2_arr[-1])))
            CACTO.y_ee_arr.append(conf.y_base + conf.l*(math.sin(CACTO.q0_arr[-1]) + math.sin(CACTO.q0_arr[-1]+CACTO.q1_arr[-1]) + math.sin(CACTO.q0_arr[-1]+CACTO.q1_arr[-1]+CACTO.q2_arr[-1])))
            
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
                        next_state_rollout = np.array([CACTO.q0_arr[final_i],CACTO.q1_arr[final_i],CACTO.q2_arr[final_i],CACTO.v0_arr[final_i],CACTO.v1_arr[final_i],CACTO.v2_arr[final_i],CACTO.t_arr[final_i]]) / self.state_norm_arr
                        next_state_rollout[-1] = 2*next_state_rollout[-1] - 1
                    else:
                        next_state_rollout = np.array([CACTO.q0_arr[final_i],CACTO.q1_arr[final_i],CACTO.q2_arr[final_i],CACTO.v0_arr[final_i],CACTO.v1_arr[final_i],CACTO.v2_arr[final_i],CACTO.t_arr[final_i]]) / self.state_norm_arr
                    tf_next_state_rollout = tf.expand_dims(tf.convert_to_tensor(next_state_rollout), 0)
                    V_final = CACTO.target_critic(tf_next_state_rollout, training=False).numpy()[0][0]
                cost_to_go = sum(ep_arr[i:final_i+1]) + V_final
                cost_to_go_arr.append(np.float32(cost_to_go))
                if i == len(ep_arr)-1:
                    DONE = 1 
                prioritized_buffer.add(np.array([CACTO.q0_arr[i],CACTO.q1_arr[i],CACTO.q2_arr[i],CACTO.v0_arr[i],CACTO.v1_arr[i],CACTO.v2_arr[i],CACTO.t_arr[i]]), next_TO_action, action, cost_to_go_arr[i], np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0]), float(DONE))
        
        # Update the NNs
        for i in range(self.UPDATE_LOOPS):
            self.learn(ep, prioritized_buffer)                                                       # Update critic and actor
            self.update_target(CACTO.target_critic.variables, CACTO.critic_model.variables, self.UPDATE_RATE)    # Update target critic
            self.update_step_counter += 1
        return self.update_step_counter, ep_return

    # Get optimal actions at timestep "step" in the current episode
    def get_TO_actions(self,step, action0_TO=None, action1_TO=None, action2_TO=None):

        actions = np.array([action0_TO[step], action1_TO[step], action2_TO[step]])
 
        # Buond actions in case they are not already bounded in the TO problem
        if step < CACTO.NSTEPS_SH:
            next_actions = np.array([np.clip(action0_TO[step+1], self.tau_lower_bound, self.tau_upper_bound), np.clip(action1_TO[step+1], self.tau_lower_bound, self.tau_upper_bound), np.clip(action2_TO[step+1], self.tau_lower_bound, self.tau_upper_bound)])
        else:
            next_actions = np.copy(actions) # Actions at last step of the episode are not performed 

        return actions, next_actions
