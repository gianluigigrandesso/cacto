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
        self.storage = []
        self.next_idx = 0
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

    def __len__(self):
        return len(self.storage)

    def add(self, obs_t, reward, obs_t1, dVdx, done, term):
        ''' Add transitions to the buffer '''
        data = (obs_t, reward, obs_t1, dVdx, done, term)
    
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.conf.REPLAY_SIZE

    def sample(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(self.conf.BATCH_SIZE)]

        obses_t = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        rewards = np.zeros((self.conf.BATCH_SIZE,1))
        obses_t1 = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        dVdxs = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        dones = np.zeros((self.conf.BATCH_SIZE,1))
        terms = np.zeros((self.conf.BATCH_SIZE,1))

        for i in range(self.conf.BATCH_SIZE):
            data = self.storage[int(idxes[i])]
            obs_t, reward, obs_t1, dVdx, done, term = data
            obses_t[i,:] = obs_t
            rewards[i,:] = reward
            obses_t1[i,:] = obs_t1
            dVdxs[i,:] = dVdx
            dones[i,0] = done
            terms[i,0] = term
   
        # Priorities not used
        weights = np.ones(len(idxes))
        batch_idxes = None

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, batch_idxes

    def sample_all(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elementsù
        idxes = [i for i in range(len(self.storage))]

        obses_t, rewards = [], []
        for i in idxes:
            data = self.storage[int(i)]
            obs_t, reward, obs_t1, dVdx, done, terms = data
            obses_t.append(np.array(obs_t, copy=False))
            rewards.append(reward)
                
        return np.array(obses_t), np.array(rewards)
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                                  
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights

class PrioritizedReplayBuffer(ReplayBuffer):
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

        self.storage = []
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

    def add(self, obs_t, reward, obs_t1, dVdx, done, term):
        ''' Add transitions to the buffer '''
        data = (obs_t, reward, obs_t1, dVdx, done, term)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
            self.exp_counter[self.next_idx] = 0

        self.next_idx = (self.next_idx + 1) % self.conf.REPLAY_SIZE

        self._it_sum[self.next_idx] = self._max_priority ** self.conf.prioritized_replay_alpha 
        self._it_min[self.next_idx] = self._max_priority ** self.conf.prioritized_replay_alpha

    def _sample_proportional(self):
        ''' Sample a batch of transitions '''
        res = []

        p_total = self._it_sum.sum(0, len(self.storage) - 1)       
        
        every_range_len = p_total / self.conf.BATCH_SIZE
        
        for i in range(self.conf.BATCH_SIZE):
            mass = random.random() * every_range_len + i * every_range_len 
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
            
        return res

    def sample(self):
        ''' Sample a batch of experiences '''
        # Compared to ReplayBuffer.sample it also returns importance weights and idxes of sampled experiences.
        idxes = self._sample_proportional()

        weights = []

        # Compute weights normalization
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.storage)) ** (-self.conf.prioritized_replay_beta)
        
        # Compute and normalize weights
        for idx in idxes:
            self.exp_counter[idx] += 1
            self.priorities[idx] = self._it_sum[idx] / self._it_sum.sum()
            weight = (self.priorities[idx] * len(self.storage)) ** (-self.conf.prioritized_replay_beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        obses_t = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        rewards = np.zeros((self.conf.BATCH_SIZE,1))
        obses_t1 = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        dVdxs = np.zeros((self.conf.BATCH_SIZE,self.conf.nb_state))
        dones = np.zeros((self.conf.BATCH_SIZE,1))
        terms = np.zeros((self.conf.BATCH_SIZE,1))

        for i in range(self.conf.BATCH_SIZE):
            data = self.storage[int(idxes[i])]
            obs_t, reward, obs_t1, dVdx, done, term = data
            obses_t[i,:] = obs_t
            rewards[i,:] = reward
            obses_t1[i,:] = obs_t1
            dVdxs[i,:] = dVdx
            dones[i,0] = done
            terms[i,0] = term
        
        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dVdxs, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dVdxs, dones, weights)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, terms, weights, [idxes]

    def update_priorities(self, idxes, priorities):
        '''Update priorities of sampled transitions '''
        # Sets priority of transition at index idxes[i] in buffer to priorities[i]
        idxes = np.asarray(idxes[0])
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            if  math.isnan(priority):
                print("\n ######################################       PRIORITY IS NAN     #######################################\n")
                priority = self._max_priority
            assert priority > 0                                                          
            assert 0 <= idx < len(self.storage)
            self._it_sum[idx] = priority ** self.conf.prioritized_replay_alpha
            self._it_min[idx] = priority ** self.conf.prioritized_replay_alpha

            self._max_priority = max(self._max_priority, priority)

    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dVdxs, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)                             
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dVdxs = tf.convert_to_tensor(dVdxs, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dVdxs, dones, weights