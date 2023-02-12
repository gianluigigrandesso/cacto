import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
 
class CACTO():
    ''' 
    ### CACTO and NNs parameters ###
    :param batch_size:                  (int) Size of the mini-batch
    :param NORMALIZE_INPUTS:            (bool) Flag to normalize inputs (state and action)
    :param EPISODE_CRITIC_PRETRAINING:  (int) Episodes of critic pretraining
    :param TD_N:                        (bool) Flag to use n-step TD rather than 1-step TD
    :param nsteps_TD_N:                 (int) Number of lookahed steps
    :param prioritized_replay_eps:      (foat) Small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
    :param prioritized_replay_alpha:    (float) Determines how much prioritization is used, set to 0 to use a normal buffer
    :param UPDATE_LOOPS:                (int) Number of updates of both critic and actor performed every EP_UPDATE episodes 
    :param SOBOLEV:                     (bool) Flag to use Sobolev training
    :param wd:                          (float) Derivative-related loss weight
    :param NSTEPS:                      (int) Max episode length
    :param EPISODE_ICS_INIT:            (int) Episodes where ICS warm-starting is used instead of actor rollout
 
    :param LR_SCHEDULE:                 (bool) Flag to use a scheduler for the learning rates
    :param NH1:                         (int) 1st hidden layer size
    :param NH2:                         (int) 2st hidden layer size 
    :param wreg_l1_A:                   (float) Weight of L1 regularization in actor's network
    :param wreg_l2_A:                   (float) Weight of L2 regularization in actor's network
    :param wreg_l1_C:                   (float) Weight of L1 regularization in critic's network
    :param wreg_l2_C:                   (float) Weight of L2 regularization in critic's network
    :param boundaries_schedule_LR_C:    (float list) 
    :param values_schedule_LR_C:        (float list) Values of critic LR 
    :param boundaries_schedule_LR_A:    (float list) 
    :param values_schedule_LR_A:        (float list) Values of actor LR 
    :param UPDATE_RATE:                 (float) Homotopy rate to update the target critic network
    :param CRITIC_LEARNING_RATE:        (float) Learning rate for the critic network
    :param ACTOR_LEARNING_RATE:         (float) Learning rate for the policy network

    ### Recover training parameters ###
    :param recover_stopped_training:    (bool) Flag to recover training
    :param update_step_counter:         (int) Recover training step number

    ### Save path ###
    :param NNs_path:                    (str) NNs save path

    ### Cost function parameters ###
    :param soft_max_param:              (float array) Soft parameters array
    :param obs_param:                   (float array) Obtacle parameters array
    :param weight:                      (float array) Weights array
    :param TARGET_STATE:                (float array) Target position

    ### Robot parameters ###
    :param dt:                          (float) Timestep
    :param robot:                       (RobotWrapper instance) 
    :param nb_state:                    (int) State size (robot state size + 1)
    :param nb_action:                   (int) Action size (robot action size)
    :param u_min:                       (float array) Action lower bound array
    :param u_max:                       (float array) Action upper bound array
    :param state_norm_arr:              (float array) Array used to normalize states
    '''

    # Initialize variables used both in TO and RL
    NSTEPS_SH = None
    control_arr = None
    state_arr = None
    x_ee_arr = []
    y_ee_arr = []
    
    actor_model = None
    critic_model = None
    target_critic = None
    actor_optimizer = None
    critic_optimizer = None
    ACTOR_LR_SCHEDULE = None
    CRITIC_LR_SCHEDULE = None
    

    def __init__(self, env, conf, init_setup_model=True):
        self.env = env
        self.conf = conf

        return

    # Update target critic NN
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one. Working only with TensorFlow tensors (e.g. not working with Numpy arrays)
    #@tf.function  
    def update_target(self,target_weights, weights, tau): 
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))  
            
    # Create actor NN 
    def get_actor(self):
        inputs = layers.Input(shape=(self.conf.nb_state,))
        
        lay1 = layers.Dense(self.conf.NH1,kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(inputs)                                        
        leakyrelu1 = layers.LeakyReLU()(lay1)
        
        lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(leakyrelu1)                                           
        leakyrelu2 = layers.LeakyReLU()(lay2)
        
        outputs = layers.Dense(self.conf.nb_action, activation="tanh", kernel_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.wreg_l1_A,self.conf.wreg_l2_A))(leakyrelu2) 
        outputs = outputs * self.conf.u_max          # Bound actions
        
        model = tf.keras.Model(inputs, outputs)
        return model 

    # Create critic NN 
    def get_critic(self): 
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
    
    # Setup RL model #
    def setup_model(self):
        # Create actor, critic and target NNs
        CACTO.actor_model = self.get_actor()
        CACTO.critic_model = self.get_critic()
        CACTO.target_critic = self.get_critic()

        # Set optimizer specifying the learning rates
        if self.conf.LR_SCHEDULE:
            # Piecewise constant decay schedule
            CACTO.CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_C, self.conf.values_schedule_LR_C) 
            CACTO.ACTOR_LR_SCHEDULE  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.conf.boundaries_schedule_LR_A, self.conf.values_schedule_LR_A)
            CACTO.critic_optimizer   = tf.keras.optimizers.Adam(CACTO.CRITIC_LR_SCHEDULE)
            CACTO.actor_optimizer    = tf.keras.optimizers.Adam(CACTO.ACTOR_LR_SCHEDULE)
        else:
            CACTO.critic_optimizer   = tf.keras.optimizers.Adam(self.conf.CRITIC_LEARNING_RATE)
            CACTO.actor_optimizer    = tf.keras.optimizers.Adam(self.conf.ACTOR_LEARNING_RATE)

        # Set initial weights of the NNs
        if self.conf.recover_stopped_training: 
            CACTO.actor_model.load_weights(self.conf.NNs_path+"/actor_{}.h5".format(self.conf.update_step_counter))
            CACTO.critic_model.load_weights(self.conf.NNs_path+"/critic_{}.h5".format(self.conf.update_step_counter))
            CACTO.target_critic.load_weights(self.conf.NNs_path+"/target_critic_{}.h5".format(self.conf.update_step_counter))
        else:
            CACTO.target_critic.set_weights(CACTO.critic_model.get_weights())   
