from utils import normalize_tensor
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
from plot import PLOT

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse

class NN:

    def __init__(self, env, conf, w_S=0):
        '''    
        :input env :                            (Environment instance)

        :input conf :                           (Configuration file)

            :param NH1:                         (int) 1st hidden layer size
            :param NH2:                         (int) 2nd hidden layer size
            :param kreg_l1_A :                  (float) Weight of L1 regularization in actor's network - kernel  
            :param kreg_l2_A :                  (float) Weight of L2 regularization in actor's network - kernel  
            :param breg_l1_A :                  (float) Weight of L2 regularization in actor's network - bias  
            :param breg_l2_A :                  (float) Weight of L2 regularization in actor's network - bias  
            :param kreg_l1_C :                  (float) Weight of L1 regularization in critic's network - kernel  
            :param kreg_l2_C :                  (float) Weight of L2 regularization in critic's network - kernel  
            :param breg_l1_C :                  (float) Weight of L1 regularization in critic's network - bias  
            :param breg_l2_C :                  (float) Weight of L2 regularization in critic's network - bias  
            :param u_max :                      (float array) Action upper bound array
            :param nb_state :                   (int) State size (robot state size + 1)
            :param nb_action :                  (int) Action size (robot action size)
            :param NORMALIZE_INPUTS :           (bool) Flag to normalize inputs (state)
            :param state_norm_array :           (float array) Array used to normalize states
            :param MC :                         (bool) Flag to use MC or TD(n)
            :param cost_weights_terminal :      (float array) Running cost weights vector
            :param cost_weights_running :       (float array) Terminal cost weights vector 
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param dt :                         (float) Timestep

        :input w_S :                            (float) Sobolev-training weight
    '''

        self.env = env
        self.conf = conf

        self.w_S = w_S

        return
    
    def create_actor(self):
        ''' Create actor NN '''
        inputs = layers.Input(shape=(self.conf.nb_state,))
        
        lay1 = layers.Dense(self.conf.NH1,kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(inputs)                                        
        leakyrelu1 = layers.LeakyReLU()(lay1)
        lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(leakyrelu1)                                           
        leakyrelu2 = layers.LeakyReLU()(lay2)  
        
        outputs = layers.Dense(self.conf.nb_action, activation="tanh", kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_A,self.conf.kreg_l2_A),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_A,self.conf.breg_l2_A))(leakyrelu2) 
        outputs = outputs * self.conf.u_max          # Bound actions
        
        model = tf.keras.Model(inputs, outputs)
        
        return model 
 
    def create_critic_tanh(self): 
        ''' Create critic NN - tanh'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = layers.Dense(16,activation='tanh')(state_input) 
        state_out2 = layers.Dense(32,activation='tanh')(state_out1) 
        out_lay1 = layers.Dense(self.conf.NH1,activation='tanh')(state_out2)
        out_lay2 = layers.Dense(self.conf.NH2,activation='tanh')(out_lay1)
        
        outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l1_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.breg_l1_C,self.conf.breg_l2_C))(out_lay2)
        model = tf.keras.Model([state_input], outputs)

        return model    
    
    def create_critic_elu(self): 
        ''' Create critic NN - elu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = layers.Dense(16, activation='elu')(state_input) 
        state_out2 = layers.Dense(32, activation='elu')(state_out1) 
        out_lay1 = layers.Dense(self.conf.NH1, activation='elu')(state_out2)
        out_lay2 = layers.Dense(self.conf.NH2, activation='elu')(out_lay1)
        
        outputs = layers.Dense(1)(out_lay2)

        model = tf.keras.Model([state_input], outputs)

        return model   
    
    def create_critic_relu(self): 
        ''' Create critic NN - relu'''
        state_input = layers.Input(shape=(self.conf.nb_state,))

        state_out1 = layers.Dense(16, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(state_input) 
        leakyrelu1 = layers.LeakyReLU()(state_out1)
        
        state_out2 = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu1) 
        leakyrelu2 = layers.LeakyReLU()(state_out2)
        out_lay1 = layers.Dense(self.conf.NH1, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu2)
        leakyrelu3 = layers.LeakyReLU()(out_lay1)
        out_lay2 = layers.Dense(self.conf.NH2, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu3)
        leakyrelu4 = layers.LeakyReLU()(out_lay2)
        
        outputs = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C),bias_regularizer=regularizers.l1_l2(self.conf.kreg_l2_C,self.conf.kreg_l2_C))(leakyrelu4)

        model = tf.keras.Model([state_input], outputs)

        return model      

    def eval(self, NN, input):
        ''' Compute the output of a NN given an input '''
        if not tf.is_tensor(input):
            input = tf.convert_to_tensor(input, dtype=tf.float32)
            
        if self.conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, self.conf.state_norm_arr)
    
        return NN(input, training=True)
    
    def compute_critic_grad(self, critic_model, target_critic, state_batch, state_next_rollout_batch, partial_reward_to_go_batch, dVdx_batch, d_batch):
        ''' Compute the gradient of the critic NN '''
        with tf.GradientTape() as tape: 
            tape.watch(critic_model.trainable_variables)
            if self.conf.MC:
                reward_to_go_batch = partial_reward_to_go_batch
            else:     
                target_values = self.eval(target_critic, state_next_rollout_batch)                         # Compute Value at next state after conf.nsteps_TD_N steps given by target critic                 
                reward_to_go_batch = partial_reward_to_go_batch + (1-d_batch)*target_values                        # Compute batch of 1-step targets for the critic loss                    

            if self.w_S != 0:
                with tf.GradientTape() as tape2:
                    tape2.watch(state_batch)                  
                    critic_value = self.eval(critic_model, state_batch)   
                der_critic_value = tape2.gradient(critic_value, state_batch)

                critic_loss_v = tf.keras.losses.mean_squared_error(reward_to_go_batch, critic_value)
                critic_loss_der = tf.keras.losses.mean_squared_error(dVdx_batch, der_critic_value)
                critic_loss = self.w_S*critic_loss_der + critic_loss_v
                        
            else:
                critic_value = self.eval(critic_model, state_batch)

                critic_loss = tf.keras.losses.mean_squared_error(reward_to_go_batch, critic_value)

        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)   

        return critic_grad

    def compute_actor_grad(self, actor_model, critic_model, state_batch, term_batch):
        ''' Compute the gradient of the actor NN '''
        actions = self.eval(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        state_next_tf, ds_next_da = self.env.simulate_batch(state_batch.numpy(), actions.numpy()) , self.env.derivative_batch(state_batch.numpy(), actions.numpy())

        with tf.GradientTape() as tape:
            tape.watch(state_next_tf)
            critic_value_next = self.eval(critic_model,state_next_tf) 
        # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)                                           
        dV_ds_next = tape.gradient(critic_value_next, state_next_tf)

        with tf.GradientTape() as tape1:
            tape1.watch(actions)
            cost_weights_terminal_reshaped = np.reshape(self.conf.cost_weights_terminal,[1,len(self.conf.cost_weights_terminal)])
            cost_weights_running_reshaped = np.reshape(self.conf.cost_weights_running,[1,len(self.conf.cost_weights_running)])
            rewards_tf = self.env.reward_batch(term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped), state_batch.numpy(), actions) #term_batch.dot(cost_weights_terminal_reshaped) + (1-term_batch).dot(cost_weights_running_reshaped)

        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
        dr_da = tape1.gradient(rewards_tf, actions, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        dr_da_reshaped = tf.reshape(dr_da, (self.conf.BATCH_SIZE, 1, self.conf.nb_action))
        
        # dr_ds' + dV_ds' (note: dr_ds' = 0)
        dQ_ds_next = tf.reshape(dV_ds_next, (self.conf.BATCH_SIZE, 1, self.conf.nb_state))        
        
        # (dr_ds' + dV_ds')*ds'_da
        dQ_ds_next_da = tf.matmul(dQ_ds_next, ds_next_da)
        
        # (dr_ds' + dV_ds')*ds'_da + dr_da
        dQ_da = dQ_ds_next_da + dr_da_reshaped

        # Now let's multiply -[(dr_ds' + dV_ds')*ds'_da + dr_da] by the actions a 
        # and then let's autodifferentiate w.r.t theta_A (actor NN's parameters) to finally get -dQ/dtheta_A 
        with tf.GradientTape() as tape:
            tape.watch(actor_model.trainable_variables)
            actions = self.eval(actor_model, state_batch)
            
            actions_reshaped = tf.reshape(actions,(self.conf.BATCH_SIZE,self.conf.nb_action,1))
            dQ_da_reshaped = tf.reshape(dQ_da,(self.conf.BATCH_SIZE,1,self.conf.nb_action))    
            Q_neg = tf.matmul(-dQ_da_reshaped,actions_reshaped) 
            
            # Also here we need a scalar so we compute the mean -Q across the batch
            mean_Qneg = tf.math.reduce_mean(Q_neg)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables)
        
        return actor_grad