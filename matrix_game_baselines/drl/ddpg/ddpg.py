from exploration.OrnsteinUhlenbeckActionNoise import OrnsteinUhlenbeckActionNoise 
from critic import CRITIC
from actor import ACTOR
import tensorflow as tf
from drl.drl import DRL
import numpy as np


class DDPG(DRL):
    """ DDPG Implementation """

    def __init__(self, config):
        '''
        :param dict config: Dictionary containing hyperparameters
        '''
        self.networks = [("cActor_", "tActor_"), ("cCritic_", "tCritic_")]
        super(DDPG, self).__init__(config)
	self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.c.outputs))

    def actorNoiseReset(self):
	'''
	Resets actor noise:
	'''
        self.actor_noise.reset()

    def getAction(self, o_t, explore=True):
	'''
        Load action using observation as input
        :param tensor o_t: state
	'''
        if not self.c.use_conv:
            self.current = np.copy(o_t).squeeze()
        outputs = self.cActor.fetch('scaled_outputs', self.sess, [self.current])
	self.action = (outputs + self.actor_noise())[0] if explore else outputs[0]
        return self.action.clip(max=self.c.ddpg.upper_bound, min=self.c.ddpg.lower_bound)


    def calcTargets(self, terminal, o_tp1, reward):
        '''
        Returns target used for updating the network.
        :param terminal: 1 if final state transition of an episode, 0 otherwise
        :param tensor o_tp1: observation at time t plus 1
        :param float reward: reward received at time t plus 1
        :return tensor containing target valus
        '''
        # Target Q-Values for o_tp1 are obtained
        a = self.tActor.fetch('scaled_outputs', self.sess, o_tp1)
        q_tp1 = self.tCritic.fetch('outputs', self.sess, o_tp1, a)
        # Tragets are calculated
        targets = (1.0 - terminal) * self.c.gamma * q_tp1 + reward
	return targets

    def optimise(self):
        '''
        Optimises DDPG
        '''

        # Optimise Critic:
        o_t, o_tp1, action, reward, terminal = self.getUnzippedSamples()
        if not self.c.use_conv:
            o_t = np.copy(o_t).squeeze()
            o_tp1 = np.copy(o_tp1).squeeze()
        terminal = np.array(terminal).reshape((self.c.erm.batch_size, 1))
        reward = np.array(reward).reshape((self.c.erm.batch_size,1))
        _, loss, outs = self.sess.run([self.criticOpt,
                                       self.loss, 
                                       self.cCritic.outputs],
                                       {self.targets: self.calcTargets(terminal, o_tp1, reward),
                                        self.cCritic.actions: action,
                                        self.cCritic.inputs: o_t})

        # Optimise Actor:
        outputs = self.cActor.fetch('scaled_outputs', self.sess, o_t)
        gradients = self.cCritic.fetch('action_grads', self.sess, o_t, outputs)
        _ = self.sess.run([self.actorOpt],
                          {self.action_gradient: gradients,
                           self.cActor.inputs: o_t})
     
    def setOptimiser(self):
        # Critic Optimiser:
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.delta = self.targets - self.cCritic.outputs
        self.deltaAfter = self.deltaProcessing()
        self.loss = tf.reduce_mean(tf.square(self.deltaAfter))
        self.criticOpt = tf.train.AdamOptimizer(self.c.ddpg.clr).minimize(self.loss)

        # Actor Optimiser:
        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.c.outputs])
        self.unnormalized_actor_gradients = tf.gradients(self.cActor.scaled_outputs, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.c.erm.batch_size), self.unnormalized_actor_gradients))
        self.actorOpt = tf.train.AdamOptimizer(self.c.ddpg.alr).apply_gradients(zip(self.actor_gradients, self.network_params))

    def addNetworks(self):
        '''
        Instantiates current and target networks.
        '''
        # Build current actor network
        self.cActor = ACTOR("cActor_" + self._name, self.c)
        self.network_params = tf.trainable_variables()

        # Build target critic network
        self.tActor = ACTOR("tActor_" + self._name, self.c)

        # Build current critic network
        self.cCritic = CRITIC("cCritic_" + self._name, self.c)

        # Build target critic network
        self.tCritic = CRITIC("tCritic_" + self._name, self.c)

