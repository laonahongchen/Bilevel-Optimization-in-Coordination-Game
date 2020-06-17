from __future__ import print_function
import os
import time
import random
from random import randint
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from fifo_shdqn import FIFO_SHDQN
from perm import PERM
from cnn2d import CNN2D
import config
from ops import clipped_error
from dqn import DQN
from time import sleep


class SHDQN(DQN):

    def __init__(self, agentID):
        self.hysteretic_beta = config.HYSTERETIC_BETA
        super(SHDQN, self).__init__(agentID)
        self.replay_memory = FIFO_SHDQN()
 

    def train_dqn(self, t):
        s_t, s_tp1, action, reward, terminal, beta = self.replay_memory.get_mini_batch()

        q_t_plus_1_with_pred_action = self.getQTarget(s_tp1)

        target_q_t = (1. - terminal) * config.DISCOUNT * q_t_plus_1_with_pred_action + reward

        target_q_t[target_q_t > 1.0] = 1.0 
        _, q_t, loss, delta = self.sess.run([self.optim, 
                                             self.current_network.outputs, 
                                             self.loss,
                                             self.delta], {
                                             self.targets: target_q_t,
                                             self.actions: action,
                                             self.current_network.inputs: s_t,
                                             self.beta: beta
                                             })
   

        if t%config.SYNC_TIME == 0:
            print("SYNC HAS OCCURED!!!!!!!!")
            self.target_network.run_copy(self.sess)


    def growHystereticBeta(self):
        self.hysteretic_beta *= config.HYSTERETIC_BETA_GROWTH_RATE
        if self.hysteretic_beta > 1.0:
            self.hysteretic_beta = 1.0 



    def setOptimizer(self, name):
	# optimize
        with tf.variable_scope('optimizer_' + name):
            self.targets = tf.placeholder('float32', [None], name='target_q_t')
            self.actions = tf.placeholder('int64', [None], name='action')
            self.beta = tf.placeholder('float32', [None], name='beta')
            action_one_hot = tf.one_hot(self.actions, config.NUMBER_OF_ACTIONS, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.current_network.outputs * action_one_hot, reduction_indices=1, name='q_acted')
            self.delta = self.targets - q_acted
            self.delta =  tf.where(tf.greater(self.delta, tf.constant(0.0)), self.delta, self.delta*self.beta)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.optim = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.loss)


