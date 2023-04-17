import math
import random
import numpy as np
import tensorflow as tf
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size, alpha=None, beta=None):
        '''
        :size:                       (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        '''
        self._maxsize = size
        self._storage = []
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, reward, obs_t1, done):
        ''' Add transitions to the buffer '''
        data = (obs_t, reward, obs_t1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

        obses_t, rewards, obses_t1, dones = [], [], [], []
        for i in idxes:
            data = self._storage[int(i)]
            obs_t, reward, obs_t1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            rewards.append(reward)
            obses_t1.append(np.array(obs_t1, copy=False))
            dones.append(done)
            
        # Priorities not used
        weights = np.ones(len(idxes))
        batch_idxes = None
        
        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dones, weights)
        
        return obses_t, rewards, obses_t1, dones, weights, batch_idxes
    
    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        size = len(rewards)
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), [size, 1])                                     
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dones = tf.reshape(tf.convert_to_tensor(dones, dtype=tf.float32), [size, 1])
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dones, weights

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, beta):
        '''
        :size:                       (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        :alpha:                      (float) Determines how much prioritization is used, set to 0 to use a normal buffer
        :beta:                       (float) Small positive constant that prevents the edge-case of transitions not being revisited once their error is zero
        '''
        
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        assert beta > 0
        self._alpha = alpha
        self._beta = beta
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        ''' Add transitions to the buffer '''
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha   # Call to __setitem__
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        ''' Sample a batch of transitions '''
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)       
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len 
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        ''' Sample a batch of experiences '''
        # Compared to ReplayBuffer.sample it also returns importance weights and idxes of sampled experiences.
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        obses_t, rewards, obses_t1, dones = [], [], [], []
        for i in idxes:
            data = self._storage[int(i)]
            obs_t, reward, obs_t1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            rewards.append(reward)
            obses_t1.append(np.array(obs_t1, copy=False))
            dones.append(done)
        
        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dones, weights)
        
        return obses_t, rewards, obses_t1, dones, weights, [idxes]

    def update_priorities(self, idxes, priorities):
        '''Update priorities of sampled transitions '''
        # Sets priority of transition at index idxes[i] in buffer to priorities[i]
        
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            if  math.isnan(priority):
                print("\n ######################################       PRIORITY IS NAN     #######################################\n")
                priority = self._max_priority
            assert priority > 0                                                          
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dones, weights):
        ''' Convert batch of transitions into a tensor '''
        size = len(rewards)
        obses_t = tf.convert_to_tensor(obses_t, dtype=tf.float32)
        rewards = tf.reshape(tf.convert_to_tensor(rewards, dtype=tf.float32), [size, 1])                                     
        obses_t1 = tf.convert_to_tensor(obses_t1, dtype=tf.float32)
        dones = tf.reshape(tf.convert_to_tensor(dones, dtype=tf.float32), [size, 1])
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        
        return obses_t, rewards, obses_t1, dones, weights