import math
import random
import numpy as np
import tensorflow as tf
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + conf.nb_state + 1 + 1))
        self.next_idx = 0
        self.full = 0
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

    def add(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Add transitions to the buffer '''
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dVdxs, dones, terms)

        if len(data) + self.next_idx > self.conf.REPLAY_SIZE:
            self.storage_mat[self.next_idx:,:] = data[:self.conf.REPLAY_SIZE-self.next_idx,:]
            self.storage_mat[:self.next_idx+len(data)-self.conf.REPLAY_SIZE,:] = data[self.conf.REPLAY_SIZE-self.next_idx:,:]
            self.full = 1
        else:
            self.storage_mat[self.next_idx:self.next_idx+len(data),:] = data

        self.next_idx = (self.next_idx + len(data)) % self.conf.REPLAY_SIZE

    def sample(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx
        idxes = np.random.randint(0, max_idx, size=self.conf.BATCH_SIZE) 

        obses_t = self.storage_mat[idxes, :self.conf.nb_state]
        rewards = self.storage_mat[idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dVdxs = self.storage_mat[idxes, self.conf.nb_state*2+1:self.conf.nb_state*3+1]
        dones = self.storage_mat[idxes, self.conf.nb_state*3+1:self.conf.nb_state*3+2]
        terms = self.storage_mat[idxes, self.conf.nb_state*3+2:self.conf.nb_state*3+3]

        # Priorities not used
        weights = np.ones((self.conf.BATCH_SIZE,1))
        batch_idxes = None

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes

    def concatenate_sample(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Convert batch of transitions into a tensor '''
        obses_t = np.concatenate(obses_t, axis=0)
        rewards = np.concatenate(rewards, axis=0)                                 
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dVdxs = np.concatenate(dVdxs, axis=0)
        dones = np.concatenate(dones, axis=0)
        terms = np.concatenate(terms, axis=0)
        
        return np.concatenate((obses_t, rewards.reshape(-1,1), obses_t1, dVdxs, dones.reshape(-1,1), terms.reshape(-1,1)),axis=1)
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                                  
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights



class PrioritizedReplayBuffer:
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)
        
            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
            :param prioritized_replay_alpha :   (float) Determines how much prioritization is used, set to 0 to use a normal buffer
            :param prioritized_replay_beta :    (float) Small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
        '''

        self.conf = conf

        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + conf.nb_state + 1 + 1))
        self.full = 0
        self.next_idx = 0
        self.exp_counter = np.zeros(self.conf.REPLAY_SIZE)
        self.priorities = np.empty(self.conf.REPLAY_SIZE)

        assert conf.prioritized_replay_alpha >= 0
        assert conf.prioritized_replay_beta > 0

        it_capacity = 1
        while it_capacity < self.conf.REPLAY_SIZE:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        #self.RB_type = 'ReLO'

        self.MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    
    def add(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Add transitions to the buffer '''
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dVdxs, dones, terms)

        if len(data) + self.next_idx > self.conf.REPLAY_SIZE:
            self.storage_mat[self.next_idx:,:] = data[:self.conf.REPLAY_SIZE-self.next_idx,:]
            self.storage_mat[:self.next_idx+len(data)-self.conf.REPLAY_SIZE,:] = data[self.conf.REPLAY_SIZE-self.next_idx:,:]
            self.full = 1
        else:
            self.storage_mat[self.next_idx:self.next_idx+len(data),:] = data
        
        for i in range(len(data)):
            self._it_sum[(self.next_idx+i) % self.conf.REPLAY_SIZE] = self._max_priority ** self.conf.prioritized_replay_alpha 
            self._it_min[(self.next_idx+i) % self.conf.REPLAY_SIZE] = self._max_priority ** self.conf.prioritized_replay_alpha
        
        self.next_idx = (self.next_idx + len(data)) % self.conf.REPLAY_SIZE

    def _sample_proportional(self):
        ''' Sample a batch of transitions '''
        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx

        idx_arr = np.zeros(self.conf.BATCH_SIZE)

        p_total = self._it_sum.sum(0, max_idx - 1)       
        
        segment = p_total / self.conf.BATCH_SIZE
        
        for i in range(self.conf.BATCH_SIZE):
            p = random.random() * segment + i * segment #random.random(segment*i, segment*(i+1))
            idx = self._it_sum.find_prefixsum_idx(p)
            idx_arr[i] = idx
            
        return idx_arr
    
    def sample(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx

        idxes = self._sample_proportional()
        batch_idxes = idxes.astype(int)

        # Compute weights normalization
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * max_idx) ** (-self.conf.prioritized_replay_beta)

        self.exp_counter[batch_idxes] += 1
        self.priorities[batch_idxes] = self._it_sum[batch_idxes] / self._it_sum.sum()
        weights = (self.priorities[batch_idxes] * max_idx) ** (-self.conf.prioritized_replay_beta) / max_weight

        obses_t = self.storage_mat[batch_idxes, :self.conf.nb_state]
        rewards = self.storage_mat[batch_idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[batch_idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dVdxs = self.storage_mat[batch_idxes, self.conf.nb_state*2+1:self.conf.nb_state*3+1]
        dones = self.storage_mat[batch_idxes, self.conf.nb_state*3+1:self.conf.nb_state*3+2]
        terms = self.storage_mat[batch_idxes, self.conf.nb_state*3+2:self.conf.nb_state*3+3]

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes

    def update_priorities(self, idxes, reward_to_go_batch, critic_value, target_critic_value=None):
        '''Update priorities of sampled transitions '''
        # Create td_errors
        if self.RB_type == 'ReLO':
            # MSE(Delta V_c) - MSE(Delta V_t)
            td_errors = self.MSE(reward_to_go_batch, critic_value).numpy()-self.MSE(reward_to_go_batch, target_critic_value).numpy()  
            td_errors_norm = np.clip(td_errors, 0, np.max(td_errors))
        else: # 'PER' is default self.RB_type
            # |TD_error_i|
            td_errors_norm = tf.math.abs(tf.math.subtract(reward_to_go_batch, critic_value))
            td_errors_norm = td_errors_norm[:,0]

        # Compute the freshness discount factor
        fresh_disc_factor = self.conf.fresh_factor**self.exp_counter[idxes]

        # Compute new priorities: p_i = mu**C_i * td_error + self.conf.prioritized_replay_eps
        new_priorities = fresh_disc_factor * td_errors_norm + self.conf.prioritized_replay_eps

        # Sets priority of transition at index idxes[i] in buffer to priorities[i]
        assert len(idxes) == len(new_priorities)
        for idx, priority in zip(idxes, new_priorities):
            assert priority > 0  

            idx = int(idx)
                                           
            self._it_sum[idx] = priority ** self.conf.prioritized_replay_alpha
            self._it_min[idx] = priority ** self.conf.prioritized_replay_alpha

            self._max_priority = max(self._max_priority, priority)

    def concatenate_sample(self, obses_t, rewards, obses_t1, dVdxs, dones, terms):
        ''' Convert batch of transitions into a tensor '''
        obses_t = np.concatenate(obses_t, axis=0)
        rewards = np.concatenate(rewards, axis=0)                                 
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dVdxs = np.concatenate(dVdxs, axis=0)
        dones = np.concatenate(dones, axis=0)
        terms = np.concatenate(terms, axis=0)
        
        return np.concatenate((obses_t, rewards.reshape(-1,1), obses_t1, dVdxs, dones.reshape(-1,1), terms.reshape(-1,1)),axis=1)
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                             
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights