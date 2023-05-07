from utils import normalize_tensor
import tensorflow as tf
from tensorflow.keras import layers, regularizers

class NN:

    def __init__(self, env, conf):

        self.env = env
        self.conf = conf

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
    
    def NN_model(self, NN, input):
        ''' Compute the output of a NN given an input '''
        if not tf.is_tensor(input):
            input = tf.convert_to_tensor(input, dtype=tf.float32)

        if self.conf.NORMALIZE_INPUTS:
            input = normalize_tensor(input, self.conf.state_norm_arr)
            
        return NN(input, training=True)
    
    def compute_critic_grad(self, critic_model, target_critic, state_batch, state_next_rollout_batch, partial_cost_to_go_batch, d_batch, weights_batch):
        ''' Compute the gradient of the critic NN '''
        with tf.GradientTape() as tape:         
            critic_value = self.NN_model(critic_model, state_batch)
            if self.conf.MC:
                cost_to_go_batch = partial_cost_to_go_batch
            else:     
                target_values = self.NN_model(target_critic, state_next_rollout_batch)                         # Compute Value at next state after conf.nsteps_TD_N steps given by target critic                 
                cost_to_go_batch = partial_cost_to_go_batch + (1-d_batch)*target_values                        # Compute batch of 1-step targets for the critic loss                    

            if weights_batch is None:
                critic_loss = tf.math.reduce_mean(tf.math.square(cost_to_go_batch - critic_value))                         # Critic loss function (tf.math.reduce_mean computes the mean of elements across dimensions of a tensor, in this case across the batch)
            else:
                critic_loss = tf.math.reduce_mean(tf.math.square(tf.math.multiply(weights_batch,(cost_to_go_batch - critic_value))))   
        
        # Compute the gradients of the critic loss w.r.t. critic's parameters
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)   

        return critic_grad

    def compute_actor_grad(self, actor_model, critic_model, state_batch, d_batch):
        ''' Compute the gradient of the actor NN '''
        actions = self.NN_model(actor_model, state_batch)

        # Both take into account normalization, ds_next_da is the gradient of the dynamics w.r.t. policy actions (ds'_da)
        state_next_tf, ds_next_da = self.env.simulate_and_derivative_tf(state_batch, actions.numpy())

        if self.conf.NORMALIZE_INPUTS:
            for i in range(self.conf.BATCH_SIZE):  
                ds_next_da[i,:-1] *= (1/self.conf.state_norm_arr[:-1,None]) 
        ds_next_da = tf.convert_to_tensor(ds_next_da, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(state_next_tf)
            critic_value_next = self.NN_model(critic_model,state_next_tf)                                           # state_next_batch = next state after applying policy's action, already normalized if self.conf.NORMALIZE_INPUTS=1
        dV_ds_next = tape.gradient(critic_value_next, state_next_tf)                                                # dV_ds' = gradient of V w.r.t. s', where s'=f(s,a) a=policy(s)   
        
        with tf.GradientTape() as tape1:
            tape1.watch(actions)
            rewards_tf = self.env.reward_tf(state_batch, actions, self.conf.BATCH_SIZE, d_batch)
        
        # dr_da = gradient of reward r(s,a) w.r.t. policy's action a
        dr_da = tape1.gradient(rewards_tf, actions)
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
            actions = self.NN_model(actor_model, state_batch)
            
            actions_reshaped = tf.reshape(actions,(self.conf.BATCH_SIZE,self.conf.nb_action,1))
            dQ_da_reshaped = tf.reshape(dQ_da,(self.conf.BATCH_SIZE,1,self.conf.nb_action))    
            Q_neg = tf.matmul(-dQ_da_reshaped,actions_reshaped) 
            
            # Also here we need a scalar so we compute the mean -Q across the batch
            mean_Qneg = tf.math.reduce_mean(Q_neg)

        # Gradients of the actor loss w.r.t. actor's parameters
        actor_grad = tape.gradient(mean_Qneg, actor_model.trainable_variables)
        
        return actor_grad