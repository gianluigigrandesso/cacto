import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
import numpy as np

class CACTO():
    # Initialize variables used both in TO and RL

    NSTEPS_SH = None
    control_arr = np.empty((0, 3))
    state_arr = np.empty((0, 7)) 
    x_ee_arr = []
    y_ee_arr = []
    
    actor_model = None
    critic_model = None
    target_critic = None
    actor_optimizer = None
    critic_optimizer = None
    ACTOR_LR_SCHEDULE = None
    CRITIC_LR_SCHEDULE = None
    

    def __init__(self, env, NORMALIZE_INPUTS, EPISODE_CRITIC_PRETRAINING, tau_upper_bound, 
                 tau_lower_bound, dt, system_param, TO_method, system, nsteps_TD_N, soft_max_param, obs_param, weight, 
                 target, NSTEPS, EPISODE_ICS_INIT, TD_N, prioritized_replay_eps,prioritized_replay_alpha,
                 state_norm_arr, UPDATE_LOOPS, UPDATE_RATE, LR_SCHEDULE, update_step_counter, 
                 NH1, NH2, wreg_l1_A, wreg_l2_A, wreg_l1_C, wreg_l2_C, boundaries_schedule_LR_C,
                 values_schedule_LR_C, boundaries_schedule_LR_A, values_schedule_LR_A, 
                 CRITIC_LEARNING_RATE, ACTOR_LEARNING_RATE,nb_state, nb_action, robot,
                 recover_stopped_training, NNs_path, batch_size,init_setup_model=True):

        self.batch_size = batch_size
        self.env = env
 
        self.NORMALIZE_INPUTS = NORMALIZE_INPUTS
        self.EPISODE_CRITIC_PRETRAINING = EPISODE_CRITIC_PRETRAINING
        self.tau_lower_bound = tau_lower_bound
        self.tau_upper_bound = tau_upper_bound
        self.dt = dt
        self.system_param = system_param
        self.LR_SCHEDULE = LR_SCHEDULE
        self.update_step_counter = update_step_counter
        self.NH1 = NH1
        self.NH2 = NH2
        self.wreg_l1_A = wreg_l1_A
        self.wreg_l2_A = wreg_l2_A
        self.wreg_l1_C = wreg_l1_C
        self.wreg_l2_C = wreg_l2_C
        self.boundaries_schedule_LR_C = boundaries_schedule_LR_C
        self.values_schedule_LR_C = values_schedule_LR_C
        self.boundaries_schedule_LR_A = boundaries_schedule_LR_A
        self.values_schedule_LR_A = values_schedule_LR_A
        self.CRITIC_LEARNING_RATE = CRITIC_LEARNING_RATE
        self.ACTOR_LEARNING_RATE = ACTOR_LEARNING_RATE
        self.nb_state = nb_state
        self.nb_action = nb_action
        self.robot = robot
        self.recover_stopped_training = recover_stopped_training
        self.NNs_path = NNs_path

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

        inputs = layers.Input(shape=(self.nb_state,))

        lay1 = layers.Dense(self.NH1,kernel_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A))(inputs)                                        
        leakyrelu1 = layers.LeakyReLU()(lay1)
    
        lay2 = layers.Dense(self.NH2, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A))(leakyrelu1)                                           
        leakyrelu2 = layers.LeakyReLU()(lay2)

        outputs = layers.Dense(self.nb_action, activation="tanh", kernel_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A),bias_regularizer=regularizers.l1_l2(self.wreg_l1_A,self.wreg_l2_A))(leakyrelu2) 

        outputs = outputs * self.tau_upper_bound          # Bound actions
        model = tf.keras.Model(inputs, outputs)
        return model 

    # Create critic NN 
    def get_critic(self): 

        state_input = layers.Input(shape=(self.nb_state,))
        state_out1 = layers.Dense(16, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C))(state_input) 
        leakyrelu1 = layers.LeakyReLU()(state_out1)

        state_out2 = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C))(leakyrelu1) 
        leakyrelu2 = layers.LeakyReLU()(state_out2)

        out_lay1 = layers.Dense(self.NH1, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C))(leakyrelu2)
        leakyrelu3 = layers.LeakyReLU()(out_lay1)

        out_lay2 = layers.Dense(self.NH2, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C))(leakyrelu3)
        leakyrelu4 = layers.LeakyReLU()(out_lay2)

        outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C),bias_regularizer=regularizers.l1_l2(self.wreg_l1_C,self.wreg_l2_C))(leakyrelu4)

        model = tf.keras.Model([state_input], outputs)

        return model    
    
    # Setup RL model #
    def setup_model(self):
        # Create actor, critic and target NNs
        CACTO.actor_model = self.get_actor()
        CACTO.critic_model = self.get_critic()
        CACTO.target_critic = self.get_critic()

        # Set optimizer specifying the learning rates
        if self.LR_SCHEDULE:
            # Piecewise constant decay schedule
            CACTO.CRITIC_LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries_schedule_LR_C, self.values_schedule_LR_C) 
            CACTO.ACTOR_LR_SCHEDULE  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries_schedule_LR_A, self.values_schedule_LR_A)
            CACTO.critic_optimizer   = tf.keras.optimizers.Adam(CACTO.CRITIC_LR_SCHEDULE)
            CACTO.actor_optimizer    = tf.keras.optimizers.Adam(CACTO.ACTOR_LR_SCHEDULE)
        else:
            CACTO.critic_optimizer   = tf.keras.optimizers.Adam(CACTO.CRITIC_LEARNING_RATE)
            CACTO.actor_optimizer    = tf.keras.optimizers.Adam(CACTO.ACTOR_LEARNING_RATE)

        # Set initial weights of the NNs
        if self.recover_stopped_training: 
            CACTO.actor_model.load_weights(self.NNs_path+"/Manipulator_{}.h5".format(self.update_step_counter))
            CACTO.critic_model.load_weights(self.NNs_path+"/Manipulator_critic{}.h5".format(self.update_step_counter))
            CACTO.target_critic.load_weights(self.NNs_path+"/Manipulator_target_critic{}.h5".format(self.update_step_counter))
        else:
            CACTO.target_critic.set_weights(CACTO.critic_model.get_weights())   
