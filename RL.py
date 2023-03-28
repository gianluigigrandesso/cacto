import numpy as np
import tensorflow as tf
from pyomo.dae import *
from pyomo.environ import *
from tensorflow.keras import layers, regularizers
from utils import *

class RL_AC:
    def __init__(self, env, conf):

        self.env = env
        self.conf = conf

        self.actor_model = None
        self.critic_model = None
        self.target_critic = None

        self.ACTOR_LR_SCHEDULE = None
        self.CRITIC_LR_SCHEDULE = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.init_rand_state = None
        self.NSTEPS_SH = None
        self.control_arr = None
        self.state_arr = None
        self.x_ee_arr = None
        self.y_ee_arr = None

        return

    def create_actor(self):
        ''' Create actor NN '''
        inputs = layers.Input(shape=(self.conf.nb_state,))
        
        lay1 = layers.Dense(self.conf.NH1,kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(inputs)                                        
        leakyrelu1 = layers.LeakyReLU()(lay1)
        
        lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(leakyrelu1)                                           
        leakyrelu2 = layers.LeakyReLU()(lay2)
        
        outputs = layers.Dense(self.conf.nb_action, activation="tanh", kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(leakyrelu2) 
        outputs = outputs * self.conf.u_max          # Bound actions
        
        model = tf.keras.Model(inputs, outputs)
        
        return model 
 
    def create_critic(self): 
        ''' Create critic NN '''
        state_input = layers.Input(shape=(self.conf.nb_state,))
        
        state_out1 = layers.Dense(16, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C))(state_input) 
        leakyrelu1 = layers.LeakyReLU()(state_out1)
        
        state_out2 = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C))(leakyrelu1) 
        leakyrelu2 = layers.LeakyReLU()(state_out2)
        
        out_lay1 = layers.Dense(self.conf.NH1, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C))(leakyrelu2)
        leakyrelu3 = layers.LeakyReLU()(out_lay1)
        
        out_lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C))(leakyrelu3)
        leakyrelu4 = layers.LeakyReLU()(out_lay2)
        
        outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_C,self.conf.wreg_l2_C))(leakyrelu4)
        
        model = tf.keras.Model([state_input], outputs)

        return model    
    
    def setup_model(self):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        self.actor_model = self.create_actor()
        self.critic_model = self.create_critic()
        self.target_critic = self.create_critic()

        # Set optimizer specifying the learning rates
        if self.conf.LR_SCHEDULE:
            # Piecewise constant decay schedule
            self.CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_C, self.conf.values_schedule_LR_C) 
            self.ACTOR_LR_SCHEDULE  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_A, self.conf.values_schedule_LR_A)
            self.critic_optimizer   = tf.keras.optimizers.Adam(self.CRITIC_LR_SCHEDULE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(self.ACTOR_LR_SCHEDULE)
        else:
            self.critic_optimizer   = tf.keras.optimizers.Adam(self.conf.CRITIC_LEARNING_RATE)
            self.actor_optimizer    = tf.keras.optimizers.Adam(self.conf.ACTOR_LEARNING_RATE)

        # Set initial weights of the NNs
        if self.conf.recover_stopped_training: 
            self.actor_model.load_weights(self.conf.NNs_path+"/actor_{}.h5".format(self.conf.update_step_counter))
            self.critic_model.load_weights(self.conf.NNs_path+"/critic_{}.h5".format(self.conf.update_step_counter))
            self.target_critic.load_weights(self.conf.NNs_path+"/target_critic_{}.h5".format(self.conf.update_step_counter))
        else:
            self.target_critic.set_weights(self.critic_model.get_weights())   

    def update(self, episode, state_batch_norm, state_batch, cost_to_go_batch, state_next_batch, state_next_batch_norm, d_batch, action_batch, weights_batch=None):
        ''' Update both critic and actor '''
        # Update the critic
        with tf.GradientTape() as tape:
            if self.conf.NORMALIZE_INPUTS:
                target_values = self.target_critic(state_next_batch_norm, training=True)             
                critic_value = self.critic_model(state_batch_norm, training=True)                    
            else:
                target_values = self.target_critic(state_next_batch, training=True)                  
                critic_value = self.critic_model(state_batch, training=True)       
            y = cost_to_go_batch + (1-min(1,self.conf.nsteps_TD_N))*(1-d_batch)*target_values                      

            if weights_batch is None:
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))                         # Critic loss function (tf.math.reduce_mean computes the mean of elements across dimensions of a tensor, in this case across the batch)
            else:
                critic_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(weights_batch,(y - critic_value))))   
        
        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)   

        # Update the critic backpropagating the gradients
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Update the actor after the critic pre-training
        if episode >= self.conf.EPISODE_CRITIC_PRETRAINING:
            if self.conf.NORMALIZE_INPUTS:
                actions = self.actor_model(state_batch_norm, training=True)
            else:
                actions = self.actor_model(state_batch, training=True)

            # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
            state_next_tf, ds_next_da = self.env.simulate_and_derivative_tf(state_batch, actions.numpy())

            if self.conf.NORMALIZE_INPUTS:
                for i in range(self.conf.BATCH_SIZE):
                    state_next_tf[i,:] = normalize(state_next_tf[i,:],self.conf.state_norm_arr)   
                    ds_next_da[i,:-1] *= (1/self.conf.state_norm_arr[:-1,None]) 
            
            state_next_tf = tf.convert_to_tensor(state_next_tf, dtype=tf.float32)
            ds_next_da = tf.convert_to_tensor(ds_next_da, dtype=tf.float32)  

            with tf.GradientTape() as tape:
                tape.watch(state_next_tf)
                critic_value_next = self.critic_model(state_next_tf, training=True)                                    # state_next_batch = next state after applying policy's action, already normalized if self.conf.NORMALIZE_INPUTS=1
            dV_ds_next = tape.gradient(critic_value_next, state_next_tf)                                               # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)   
            with tf.GradientTape() as tape1:
                with tf.GradientTape() as tape2:
                        tape1.watch(actions)
                        tape2.watch(state_next_tf)
                        rewards_tf = self.env.reward_tf(state_next_tf, actions, self.conf.BATCH_SIZE, d_batch)
            # dr_da = gradient of reward r w.r.t. policy's action a
            dr_da = tape1.gradient(rewards_tf, actions)
            dr_da_reshaped = tf.reshape(dr_da, (self.conf.BATCH_SIZE, 1, self.conf.nb_action))

            # dr_ds' = gradient of reward r w.r.t. next state s' after performing policy's action a
            dr_ds_next = tape2.gradient(rewards_tf, state_next_tf)
            
            # dr_ds' + dV_ds'
            dr_ds_next_dV_ds_next_reshaped = tf.reshape(dr_ds_next+dV_ds_next, (self.conf.BATCH_SIZE, 1, self.conf.nb_state))        

            # (dr_ds' + dV_ds')*ds'_da
            dr_ds_next_dV_ds_next = tf.matmul(dr_ds_next_dV_ds_next_reshaped, ds_next_da)
            
            # (dr_ds' + dV_ds')*ds'_da + dr_da
            sum_tf = dr_ds_next_dV_ds_next + dr_da_reshaped

            # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a 
            # and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
            with tf.GradientTape() as tape:
                tape.watch(self.actor_model.trainable_variables)
                if self.conf.NORMALIZE_INPUTS:
                    actions = self.actor_model(state_batch_norm, training=True)
                else:
                    actions = self.actor_model(state_batch, training=True)
                
                actions_reshaped = tf.reshape(actions,(self.conf.BATCH_SIZE,self.conf.nb_action,1))
                sum_tf_reshaped = tf.reshape(sum_tf,(self.conf.BATCH_SIZE,1,self.conf.nb_action))    
                Q_neg = tf.matmul(-sum_tf_reshaped,actions_reshaped) 
                
                # Also here we need a scalar so we compute the mean -Q across the batch
                mean_Qneg = tf.math.reduce_mean(Q_neg)

            # Gradients of the actor loss w.r.t. actor's parameters
            actor_grad = tape.gradient(mean_Qneg, self.actor_model.trainable_variables)

            # Update the actor backpropagating the gradients
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    @tf.function
    def update_target(self,target_weights, weights, tau):
        ''' Update target critic NN '''
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn(self, episode, prioritized_buffer):
        # Sample batch of transitions from the buffer
        experience = prioritized_buffer.sample(self.conf.BATCH_SIZE, beta=self.conf.prioritized_replay_eps)            # Bias annealing not performed, that's why beta is equal to a very small number (0 not accepted by PrioritizedReplayBuffer)
        s_batch, a_batch, ctg_batch, s_next_batch, d_batch, weights, batch_idxes = experience                          # Importance sampling weights (actually not used) should anneal the bias (see Prioritized Experience Replay paper) 
        
        # Convert batch of transitions into a tensor
        state_batch      = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        action_batch     = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        cost_to_go_batch = tf.reshape(tf.convert_to_tensor(ctg_batch, dtype=tf.float32), [self.conf.BATCH_SIZE, 1])                                     
        d_batch          = tf.reshape(tf.convert_to_tensor(d_batch, dtype=tf.float32), [self.conf.BATCH_SIZE, 1])
        state_next_batch = tf.convert_to_tensor(s_next_batch, dtype=tf.float32)

        if self.conf.prioritized_replay_alpha == 0:
            weights_batch = tf.convert_to_tensor(np.ones_like(weights), dtype=tf.float32)
        else:    
            weights_batch = tf.convert_to_tensor(weights, dtype=tf.float32)

        if self.conf.NORMALIZE_INPUTS:
            state_batch_norm = normalize_tensor(state_batch, self.conf.state_norm_arr)
            state_next_batch_norm = normalize_tensor(state_next_batch, self.conf.state_norm_arr)
        else:
            state_batch_norm = None
            state_next_batch_norm = None
        
        # Update priorities
        if self.conf.prioritized_replay_alpha != 0:
            if self.conf.NORMALIZE_INPUTS:
                v_batch = self.critic_model(state_batch_norm, training=True)                          # Compute batch of Values associated to the sampled batch ofstates
                v_next_batch = self.target_critic(state_next_batch_norm, training=True)               # Compute batch of Values from target critic associated to sampled batch of next states
            else:
                v_batch = self.critic_model(state_batch, training=True) 
                v_next_batch = self.target_critic(state_next_batch, training=True)                     
            vref_batch = cost_to_go_batch + (1-d_batch)*(v_next_batch)                          

            td_errors = tf.math.abs(tf.math.subtract(vref_batch,v_batch))                              # Compute the targets for the TD error                       
            new_priorities = td_errors.numpy() + self.conf.prioritized_replay_eps                      # Proportional prioritization where p_i = |TD_error_i| + self.conf.prioritized_replay_eps 
            prioritized_buffer.update_priorities(batch_idxes, new_priorities)   

        # Update NNs
        self.update(episode, state_batch_norm, state_batch, cost_to_go_batch, state_next_batch, state_next_batch_norm, d_batch, action_batch, weights_batch=weights_batch)

    def RL_Solve(self, ep, tau_TO, prioritized_buffer):
        ''' Solve RL problem '''
        DONE = 0                              # Flag indicating if the episode has terminated
        ep_return = 0                         # Initialize the return
        ep_arr = np.empty(self.NSTEPS_SH)     # Reward array
        
        state_prev = self.init_rand_state

        if ep == self.conf.EPISODE_ICS_INIT and self.conf.LR_SCHEDULE:
            # Re-initialize Adam otherwise it keeps being affected by the estimates of first-order and second-order moments computed previously with ICS warm-starting
            self.critic_optimizer = tf.keras.optimizers.Adam(self.CRITIC_LR_SCHEDULE)
            self.actor_optimizer = tf.keras.optimizers.Adam(self.ACTOR_LR_SCHEDULE)
        
        # START RL EPISODE
        for step_counter in range(self.NSTEPS_SH):
            # Get current TO action
            action = tau_TO[step_counter, :] # action clipped in TO

            # Simulate actions and retrieve next state and compute reward
            state_next, rwrd = self.env.step(state_prev, action)
            # Store performed action and next state and reward
            self.control_arr[step_counter,:] = action.reshape(1,self.conf.nb_action)
            self.state_arr[step_counter+1,:] = state_next.reshape(1,self.conf.nb_state)
            self.x_ee_arr[step_counter+1], self.y_ee_arr[step_counter+1] = [self.env.get_end_effector_position(self.state_arr[-1,:])[i] for i in range(2)]

            ep_arr[step_counter] = rwrd

            # Increment the episodic return by the reward just recived
            ep_return += rwrd

            # Next state becomes the prev state at the next episode step
            state_prev = np.copy(state_next)   
        DONE = 1
        
        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        DONE = 0
        cost_to_go_arr = np.empty(len(ep_arr))

        for i in range(len(ep_arr)):
            final_lookahead_step = min(i+self.conf.nsteps_TD_N, len(ep_arr))
            if final_lookahead_step == len(ep_arr):
                V_final = 0.0
            else:
                if self.conf.NORMALIZE_INPUTS:
                    state_next_rollout = normalize(self.state_arr[final_lookahead_step,:], self.conf.state_norm_arr)
                else:
                    state_next_rollout = self.state_arr[final_lookahead_step,:]
                state_next_rollout_tf = tf.convert_to_tensor(np.array([state_next_rollout]), dtype=tf.float32)
                V_final = self.target_critic(state_next_rollout_tf, training=False).numpy()[0][0]
            cost_to_go = sum(ep_arr[i:final_lookahead_step+1]) + min(1,self.conf.nsteps_TD_N)*V_final 
            cost_to_go_arr[i] = np.float32(cost_to_go)
            
            if i == len(ep_arr)-1:
                DONE = 1
            prioritized_buffer.add(self.state_arr[i, :], action, cost_to_go_arr[i], np.zeros(self.conf.nb_state), float(DONE))
            

        # Update the NNs
        if ep >= self.conf.ep_no_update and ep % self.conf.EP_UPDATE == 0:
            for i in range(self.conf.UPDATE_LOOPS):
                # Learn and update critic
                self.learn(ep, prioritized_buffer)

                # Update target critic
                self.update_target(self.target_critic.variables, self.critic_model.variables, self.conf.UPDATE_RATE)

                self.conf.update_step_counter += 1

        return ep_return, self.conf.update_step_counter, self.x_ee_arr, self.y_ee_arr

    def create_TO_init(self):
        ''' Create initial state and initial controls for TO '''
        # Select an initial state at random
        init_rand_time, self.init_rand_state = self.env.reset()      

        # Set the horizon of TO problem / RL episode
        self.NSTEPS_SH = self.conf.NSTEPS - int(round(init_rand_time/self.conf.dt))

        # Lists to store TO state and control trajectories
        self.control_arr = np.empty((self.NSTEPS_SH, self.conf.nb_action))
        self.state_arr = np.empty((self.NSTEPS_SH+1, self.conf.nb_state))
        self.x_ee_arr = np.empty(self.NSTEPS_SH+1)
        self.y_ee_arr = np.empty(self.NSTEPS_SH+1)

        self.state_arr[0,:] = self.init_rand_state
        self.x_ee_arr[0], self.y_ee_arr[0] = [self.env.get_end_effector_position(self.state_arr[-1, :])[i] for i in range(2)]

        # Actor rollout used to initialize TO state and control variables
        init_TO_controls = np.zeros((self.conf.nb_action, self.NSTEPS_SH+1))
        init_TO_states = np.zeros((self.conf.nb_state, self.NSTEPS_SH+1))

        if self.conf.NORMALIZE_INPUTS:
            state_prev_tf = tf.convert_to_tensor(np.array([normalize(self.init_rand_state, self.conf.state_norm_arr)]), dtype=tf.float32)
        else:
            state_prev_tf = tf.convert_to_tensor(np.array([self.init_rand_state]), dtype=tf.float32)  
        init_TO_controls[:,0] = tf.squeeze(self.actor_model(state_prev_tf)).numpy()

        init_TO_states[:,0] = self.init_rand_state
        
        init_state_prev = np.copy(self.init_rand_state)
        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables
        for i in range(1, self.NSTEPS_SH+1):
            init_TO_controls_sim = init_TO_controls[:,i-1]                                              
            init_next_state, _ =  self.env.step(init_state_prev,init_TO_controls_sim)
            init_TO_states[:,i] = init_next_state

            if self.conf.NORMALIZE_INPUTS:
                init_next_state_tf = tf.convert_to_tensor(np.array([normalize(init_next_state, self.conf.state_norm_arr)]), dtype=tf.float32)      
            else:    
                init_next_state_tf = tf.convert_to_tensor(np.array([init_next_state]), dtype=tf.float32)
            init_TO_controls[:,i] = tf.squeeze(self.actor_model(init_next_state_tf)).numpy()
            
            init_state_prev = np.copy(init_next_state)

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH