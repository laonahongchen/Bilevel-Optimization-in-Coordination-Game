from collections import deque
import random as random
import numpy as np

class FIFO(object):
    '''Episodic fifo stores whole episodes innside a 
       first-in first-out queue data structure.'''

    def __init__(self, c):
        '''
        Class is instantiated with a queue data
        structure for storing the episodes, and
        a list for storing the transitions of 
        the current episode. 
        :param config: Replay Memory Hyperparameters
        '''
        self.c = c
        self._transitions = deque([], c.erm.size)
        self._num_transitions_stored = 0
        
    def getSize(self):
        ''' 
        Returns the number of transitions currently
        stored inside the list. 
        '''
        return len(self._transitions)

    def add_experience(self, transition, meta_strategy=None):
        ''' 
        Method used to add state transitions to the replay memory. 
        :param transition: Tuple containing state transition tuple
        '''
        # Add transition to list:
        self._transitions.append(transition)

    def isFull(self):
        '''
        :return bool: True if RM is full, false otherwise
        '''
        return True if len(self._transitions) == self.c.erm.size else False
            

    def get_mini_batch(self):
        '''
        Method returns a mini-batch of sample traces.
        :return list traces: List of traces
        '''
        # Episodes are randomly choosen for sequence sampling:
        if len(self.c.dim) == 3:
 	    # From each of the episodes a sequence is selected:
            return random.sample(self._transitions, self.c.erm.batch_size)	
        else:
            samples = [] # List used to store n traces used for sampling
            for i in range(self.c.erm.batch_size):
                transition = random.randint(self.c.erm.sequence_len, len(self._transitions))
                # State-trajectories are stored in lists.
                # Storing these each time provides memory to 
                # run multiple training runs in paralle at the 
                # cost of efficiency.
                o_t = []  
                o_tp1 = []
                for j in range(transition-self.c.erm.sequence_len, transition):
                    o_t.append(self._transitions[j][0])
                    o_tp1.append(self._transitions[j][1])
                transitionTuple = np.copy(self._transitions[transition-1])
                transitionTuple[0] = o_t
                transitionTuple[1] = o_tp1
                samples.append(transitionTuple)
        return samples

    def getUnzippedSamples(self):
        '''
        :return unzipped samples
        '''
        # Samples are obtained from the replay memory and unzipped
        samples = self.get_mini_batch()
        return zip(*samples)
