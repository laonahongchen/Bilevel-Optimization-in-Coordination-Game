import tensorflow as tf
from drl.dqn.dqn import DQN

def HQL_FACTORY(parent=DQN):
    class HQL(parent):
        """ Hysteretic-Q-Learning """
        def deltaProcessing(self):
            '''
            Method applies hysteretic Q-learning principle to the losses.
            If a loss is less than 0, then it is downsized using, the 
            hysteretic beta value supplied in the config file.
            :return vector: delta with downsized negative losses
            '''
            return tf.where(tf.greater(self.delta, tf.constant(0.0)), 
			    self.delta, 
			    self.delta*self.c.hysteretic.beta)
    return HQL
