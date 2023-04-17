import numpy as np
import tensorflow as tf

class RL_AC:
    def __init__(self, env, NN, conf):

        self.env = env
        self.NN = NN
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
    
    def setup_model(self):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        self.actor_model = self.NN.create_actor()
        self.critic_model = self.NN.create_critic()
        self.target_critic = self.NN.create_critic()

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

    def update(self, episode, state_batch, state_next_rollout_batch, partial_cost_to_go_batch, d_batch, weights_batch=None):
        ''' Update both critic and actor '''
        # Update the critic backpropagating the gradients
        critic_grad = self.NN.compute_critic_grad(self.critic_model, self.target_critic, state_batch, state_next_rollout_batch, partial_cost_to_go_batch, d_batch, weights_batch)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Update the actor backpropagating the gradients
        actor_grad = self.NN.compute_actor_grad(self.actor_model, self.critic_model, state_batch, d_batch)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    @tf.function
    def update_target(self,target_weights, weights, tau):
        ''' Update target critic NN '''
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_priorities(self,state_batch,state_next_rollout_batch, partial_cost_to_go_batch, d_batch, batch_idxes, buffer):
        ''' Update buffer priorities '''
        v_batch = self.NN.NN_model(self.critic_model, state_batch)                           # Compute batch of Values associated to the sampled batch ofstates
        if self.conf.MC:
            vref_batch = partial_cost_to_go_batch
        else:
            v_next_batch = self.NN.NN_model(self.target_critic, state_next_rollout_batch)    # Compute batch of Values from target critic associated to sampled batch of next rollout states                
            vref_batch = partial_cost_to_go_batch + (1-d_batch)*(v_next_batch)                          

        td_errors = tf.math.abs(tf.math.subtract(vref_batch,v_batch))                        # Compute the targets for the TD error                       
        new_priorities = td_errors.numpy() + self.conf.prioritized_replay_eps                # Proportional prioritization where p_i = |TD_error_i| + self.conf.prioritized_replay_eps 
        buffer.update_priorities(batch_idxes, new_priorities)  

    def learn_and_update(self, ep, update_step_counter, buffer):
        ''' Sample experience and update buffer priorities and NNs '''
        for i in range(self.conf.UPDATE_LOOPS):
            # Sample batch of transitions from the buffer
            experience = buffer.sample(self.conf.BATCH_SIZE)                                                                                # Bias annealing not performed, that's why beta is equal to a very small number (0 not accepted by PrioritizedReplayBuffer)
            state_batch, partial_cost_to_go_batch, state_next_rollout_batch, d_batch, weights_batch, batch_idxes = experience                          # Importance sampling weights (actually not used) should anneal the bias (see Prioritized Experience Replay paper) 

            # Update priorities
            if self.conf.prioritized_replay_alpha != 0:
                self.update_priorities(state_batch,state_next_rollout_batch, partial_cost_to_go_batch, d_batch, batch_idxes,buffer)

            # Update both critic and actor
            self.update(ep, state_batch, state_next_rollout_batch, partial_cost_to_go_batch, d_batch, weights_batch=weights_batch)

            # Update target critic
            self.update_target(self.target_critic.variables, self.critic_model.variables, self.conf.UPDATE_RATE)

            update_step_counter += 1
        
        return update_step_counter
    
    def RL_Solve(self, ep, TO_controls):
        ''' Solve RL problem '''
        ep_return = 0                          # Initialize the return
        rwrd_arr = np.empty(self.NSTEPS_SH)    # Reward array
        state_next_rollout_arr = np.empty((self.NSTEPS_SH, self.conf.nb_state))    # Next state array
        partial_cost_to_go_arr = np.empty(len(rwrd_arr))                           # Partial cost-to-go array
        done_arr = np.zeros(len(rwrd_arr))                                         # Episode-termination flag array

        if ep == self.conf.EPISODE_ICS_INIT and self.conf.LR_SCHEDULE:
            # Re-initialize Adam otherwise it keeps being affected by the estimates of first-order and second-order moments computed previously with ICS warm-starting
            self.critic_optimizer = tf.keras.optimizers.Adam(self.CRITIC_LR_SCHEDULE)
            self.actor_optimizer = tf.keras.optimizers.Adam(self.ACTOR_LR_SCHEDULE)
        
        # START RL EPISODE
        for step_counter in range(self.NSTEPS_SH):
            # Get current TO action
            self.control_arr[step_counter,:] = TO_controls[step_counter, :] # action clipped in TO

            # Simulate actions and retrieve next state and compute reward
            self.state_arr[step_counter+1,:], rwrd_arr[step_counter] = self.env.step(self.state_arr[step_counter,:], self.control_arr[step_counter,:])

            # Store performed action and next state and reward
            self.x_ee_arr[step_counter+1], self.y_ee_arr[step_counter+1] = [self.env.get_end_effector_position(self.state_arr[step_counter+1,:])[i] for i in range(2)]

            # Increment the episodic return by the reward just recived
            ep_return += rwrd_arr[step_counter]
        
        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        for i in range(len(rwrd_arr)):
            # set final lookahead step depending on whether Monte Cartlo or TD(n) is used
            if self.conf.MC:
                final_lookahead_step = len(rwrd_arr)
                state_next_rollout_arr[i,:] = np.empty_like(self.state_arr[-1,:])
            else:
                final_lookahead_step = min(i+self.conf.nsteps_TD_N, len(rwrd_arr))
                state_next_rollout_arr[i,:] = self.state_arr[final_lookahead_step,:]
            
            # Compute the partial cost to go
            partial_cost_to_go = sum(rwrd_arr[i:final_lookahead_step+1])
            partial_cost_to_go_arr[i] = np.float32(partial_cost_to_go)
            
            # Set done_arr[i] to 1 if the episode is terminated
            if i == len(rwrd_arr)-1:
                done_arr[i] = 1 

        return self.state_arr, partial_cost_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, ep_return, self.x_ee_arr, self.y_ee_arr

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
        self.x_ee_arr[0], self.y_ee_arr[0] = [self.env.get_end_effector_position(self.state_arr[0, :])[i] for i in range(2)]

        # Actor rollout used to initialize TO state and control variables
        init_TO_controls = np.zeros((self.conf.nb_action, self.NSTEPS_SH))
        init_TO_states = np.zeros((self.conf.nb_state, self.NSTEPS_SH+1))

        init_TO_states[:,0] = self.init_rand_state

        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables
        for i in range(self.NSTEPS_SH):   
            init_TO_controls[:,i] = tf.squeeze(self.NN.NN_model(self.actor_model, np.array([init_TO_states[:,i]]))).numpy()
            init_TO_states[:,i+1], _ = self.env.step(init_TO_states[:,i],init_TO_controls[:,i])

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH