import sys
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from pyomo.environ import *
from pyomo.dae import *
import numpy as np

class CACTO():
    # Initialize variables used both in TO and RL

    NSTEPS_SH = None
    control_arr = None
    state_arr = None
    x_ee_arr = []
    y_ee_arr = []

    critic_loss_tot = []
    
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

        self.batch_size = conf.BATCH_SIZE
        self.NORMALIZE_INPUTS = conf.NORMALIZE_INPUTS
        self.EPISODE_CRITIC_PRETRAINING = conf.EPISODE_CRITIC_PRETRAINING
        self.tau_lower_bound = conf.tau_lower_bound
        self.tau_upper_bound = conf.tau_upper_bound
        self.dt = conf.dt
        self.LR_SCHEDULE = conf.LR_SCHEDULE
        self.update_step_counter = conf.update_step_counter
        self.NH1 = conf.NH1
        self.NH2 = conf.NH2
        self.wreg_l1_A = conf.wreg_l1_A
        self.wreg_l2_A = conf.wreg_l2_A
        self.wreg_l1_C = conf.wreg_l1_C
        self.wreg_l2_C = conf.wreg_l2_C
        self.boundaries_schedule_LR_C = conf.boundaries_schedule_LR_C
        self.values_schedule_LR_C = conf.values_schedule_LR_C
        self.boundaries_schedule_LR_A = conf.boundaries_schedule_LR_A
        self.values_schedule_LR_A = conf.values_schedule_LR_A
        self.CRITIC_LEARNING_RATE = conf.CRITIC_LEARNING_RATE
        self.ACTOR_LEARNING_RATE = conf.ACTOR_LEARNING_RATE
        self.nb_state = conf.nb_state
        self.nb_action = conf.nb_action
        self.robot = conf.robot
        self.recover_stopped_training = conf.recover_stopped_training
        self.NNs_path = conf.NNs_path

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
            CACTO.actor_model.load_weights(self.NNs_path+"/actor_{}.h5".format(self.update_step_counter))
            CACTO.critic_model.load_weights(self.NNs_path+"/critic_{}.h5".format(self.update_step_counter))
            CACTO.target_critic.load_weights(self.NNs_path+"/target_critic_{}.h5".format(self.update_step_counter))
        else:
            CACTO.target_critic.set_weights(CACTO.critic_model.get_weights())   
