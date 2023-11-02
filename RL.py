import uuid
import math
import numpy as np
import tensorflow as tf

class RL_AC:
    def __init__(self, env, NN, conf, N_try):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :parma critic_type :                (str) Activation function to use for the critic NN
            :param LR_SCHEDULE :                (bool) Flag to use a scheduler for the learning rates
            :param boundaries_schedule_LR_C :   (list) Boudaries of critic LR
            :param values_schedule_LR_C :       (list) Values of critic LR
            :param boundaries_schedule_LR_A :   (list) Boudaries of actor LR
            :param values_schedule_LR_A :       (list) Values of actor LR
            :param CRITIC_LEARNING_RATE :       (float) Learning rate for the critic network
            :param ACTOR_LEARNING_RATE :        (float) Learning rate for the policy network
            :param fresh_factor :               (float) Refresh factor
            :param prioritized_replay_alpha :   (float) α determines how much prioritization is used
            :param prioritized_replay_eps :     (float) It's a small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
            :param UPDATE_LOOPS :               (int array) Number of updates of both critic and actor performed every EP_UPDATE episodes
            :param save_interval :              (int) save NNs interval
            :param env_RL :                     (bool) Flag RL environment
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param MC :                         (bool) Flag to use MC or TD(n)
            :param nsteps_TD_N :                (int) Number of lookahed steps if TD(n) is used
            :param UPDATE_RATE :                (float) Homotopy rate to update the target critic network if TD(n) is used
            :param cost_weights_terminal :      (float array) Running cost weights vector
            :param cost_weights_running :       (float array) Terminal cost weights vector 
            :param dt :                         (float) Timestep
            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param NNs_path :                   (str) NNs path
            :param NSTEPS :                     (int) Max episode length

    '''
        self.env = env
        self.NN = NN
        self.conf = conf

        self.N_try = N_try

        self.actor_model = None
        self.critic_model = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.init_rand_state = None
        self.NSTEPS_SH = 0
        self.control_arr = None
        self.state_arr = None
        self.ee_pos_arr = None
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

        return
    
    def setup_model(self, recover_training=None):
        ''' Setup RL model '''
        # Create actor, critic and target NNs
        self.actor_model = self.NN.create_actor()

        if self.conf.critic_type == 'elu':
            self.critic_model = self.NN.create_critic_elu()
            self.target_critic = self.NN.create_critic_elu()
        elif self.conf.critic_type == 'sine':
            self.critic_model = self.NN.create_critic_sine()
            self.target_critic = self.NN.create_critic_sine()
        elif self.conf.critic_type == 'sine-elu':
            self.critic_model = self.NN.create_critic_sine_elu()
            self.target_critic = self.NN.create_critic_sine_elu()
        else:
            self.critic_model = self.NN.create_critic_relu()
            self.target_critic = self.NN.create_critic_relu()

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
        if recover_training is not None: 
            NNs_path_rec = str(recover_training[0])
            N_try = recover_training[1]
            update_step_counter = recover_training[2]
            self.actor_model.load_weights("{}/N_try_{}/actor_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
            self.critic_model.load_weights("{}/N_try_{}/critic_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
            self.target_critic.load_weights("{}/N_try_{}/target_critic_{}.h5".format(NNs_path_rec,N_try,update_step_counter))
        else:
            self.target_critic.set_weights(self.critic_model.get_weights())   

    def update(self, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_size=None):
        ''' Update both critic and actor '''
        # Update the critic backpropagating the gradients
        critic_grad, reward_to_go_batch, critic_value, target_critic_value = self.NN.compute_critic_grad(self.critic_model, self.target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, weights_batch)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Update the actor backpropagating the gradients
        actor_grad = self.NN.compute_actor_grad(self.actor_model, self.critic_model, state_batch, term_batch, batch_size)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        return reward_to_go_batch, critic_value, target_critic_value
    
    @tf.function
    def update_target(self, target_weights, weights):
        ''' Update target critic NN '''
        tau = self.conf.UPDATE_RATE
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def learn_and_update(self, update_step_counter, buffer, ep):
        ''' Sample experience and update buffer priorities and NNs '''
        for i in range(int(self.conf.UPDATE_LOOPS[ep])):
            # Sample batch of transitions from the buffer
            state_batch, partial_reward_to_go_batch, state_next_rollout_batch, dVdx_batch, d_batch, term_batch, weights_batch, batch_idxes = buffer.sample()

            # Update both critic and actor
            reward_to_go_batch, critic_value, target_critic_value = self.update(state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch, term_batch, weights_batch)

            # Update buffer priorities
            if self.conf.prioritized_replay_alpha != 0:                                
                buffer.update_priorities(batch_idxes, reward_to_go_batch, critic_value, target_critic_value)  

            # Update target critic
            if not self.conf.MC:
                self.update_target(self.target_critic.variables, self.critic_model.variables)

            update_step_counter += 1

            # Plot rollouts and save the NNs every conf.log_rollout_interval-training episodes
            if update_step_counter%self.conf.save_interval == 0:
                self.RL_save_weights(update_step_counter)

        return update_step_counter
    
    def RL_Solve(self, TO_controls, TO_states, TO_step_cost):
        ''' Solve RL problem '''
        ep_return = 0                                                                 # Initialize the return
        rwrd_arr = np.empty(self.NSTEPS_SH+1)                                         # Reward array
        state_next_rollout_arr = np.zeros((self.NSTEPS_SH+1, self.conf.nb_state))     # Next state array
        partial_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                         # Partial cost-to-go array
        total_reward_to_go_arr = np.empty(self.NSTEPS_SH+1)                           # Total cost-to-go array
        term_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-termination flag array
        term_arr[-1] = 1
        done_arr = np.zeros(self.NSTEPS_SH+1)                                         # Episode-MC-termination flag array

        # START RL EPISODE
        self.control_arr = TO_controls # action clipped in TO
        
        if self.conf.env_RL:
            for step_counter in range(self.NSTEPS_SH):
                # Simulate actions and retrieve next state and compute reward
                self.state_arr[step_counter+1,:], rwrd_arr[step_counter] = self.env.step(self.conf.cost_weights_running, self.state_arr[step_counter,:], self.control_arr[step_counter,:])

                # Compute end-effector position
                self.ee_pos_arr[step_counter+1,:] = self.env.get_end_effector_position(self.state_arr[step_counter+1, :])
            rwrd_arr[-1] = self.env.reward(self.conf.cost_weights_terminal, self.state_arr[-1,:])
        else:
            self.state_arr, rwrd_arr = TO_states, -TO_step_cost

        ep_return = sum(rwrd_arr)

        # Store transition after computing the (partial) cost-to go when using n-step TD (from 0 to Monte Carlo)
        for i in range(self.NSTEPS_SH+1):
            # set final lookahead step depending on whether Monte Cartlo or TD(n) is used
            if self.conf.MC:
                final_lookahead_step = self.NSTEPS_SH
                done_arr[i] = 1 
            else:
                final_lookahead_step = min(i+self.conf.nsteps_TD_N, self.NSTEPS_SH)
                if final_lookahead_step == self.NSTEPS_SH:
                    done_arr[i] = 1 
                else:
                    state_next_rollout_arr[i,:] = self.state_arr[final_lookahead_step+1,:]
            
            # Compute the partial and total cost to go
            partial_reward_to_go_arr[i] = np.float32(sum(rwrd_arr[i:final_lookahead_step+1]))
            total_reward_to_go_arr[i] = np.float32(sum(rwrd_arr[i:self.NSTEPS_SH+1]))

        return self.state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, self.ee_pos_arr
    
    def RL_save_weights(self, update_step_counter='final'):
        ''' Save NN weights '''
        self.actor_model.save_weights(self.conf.NNs_path+"/N_try_{}/actor_{}.h5".format(self.N_try,update_step_counter))
        self.critic_model.save_weights(self.conf.NNs_path+"/N_try_{}/critic_{}.h5".format(self.N_try,update_step_counter))
        self.target_critic.save_weights(self.conf.NNs_path+"/N_try_{}/target_critic_{}.h5".format(self.N_try,update_step_counter))

    def create_TO_init(self, ep, ICS):
        ''' Create initial state and initial controls for TO '''
        self.init_rand_state = ICS    
        
        self.NSTEPS_SH = self.conf.NSTEPS - int(self.init_rand_state[-1]/self.conf.dt)
        if self.NSTEPS_SH == 0:
            return None, None, None, None, 0

        # Initialize array to store RL state, control, and end-effector trajectories
        self.control_arr = np.empty((self.NSTEPS_SH, self.conf.nb_action))
        self.state_arr = np.empty((self.NSTEPS_SH+1, self.conf.nb_state))
        self.ee_pos_arr = np.empty((self.NSTEPS_SH+1,3))

        # Set initial state and end-effector position
        self.state_arr[0,:] = self.init_rand_state
        self.ee_pos_arr[0,:] = self.env.get_end_effector_position(self.state_arr[0, :])

        # Initialize array to initialize TO state and control variables
        init_TO_controls = np.zeros((self.NSTEPS_SH, self.conf.nb_action))
        init_TO_states = np.zeros(( self.NSTEPS_SH+1, self.conf.nb_state))

        # Set initial state 
        init_TO_states[0,:] = self.init_rand_state

        # Simulate actor's actions to compute the state trajectory used to initialize TO state variables (use ICS for state and 0 for control if it is the first episode otherwise use policy rollout)
        success_init_flag = 1
        for i in range(self.NSTEPS_SH):   
            if ep == 0:
                init_TO_controls[i,:] = np.zeros(self.conf.nb_action)
            else:
                init_TO_controls[i,:] = tf.squeeze(self.NN.eval(self.actor_model, np.array([init_TO_states[i,:]]))).numpy()
            init_TO_states[i+1,:] = self.env.simulate(init_TO_states[i,:],init_TO_controls[i,:])
            if np.isnan(init_TO_states[i+1,:]).any():
                success_init_flag = 0
                return None, None, None, None, success_init_flag

        return self.init_rand_state, init_TO_states, init_TO_controls, self.NSTEPS_SH, success_init_flag