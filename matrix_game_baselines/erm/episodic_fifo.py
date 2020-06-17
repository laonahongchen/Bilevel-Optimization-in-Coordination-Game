from collections import deque
import random as random
import numpy as np

class EPISODIC_FIFO(object):
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
        self._episodes = deque([])
        self._episode = []
        self._num_transitions_stored = 0
        self.c = c

    def getSize(self):
        ''' 
        Returns the number of transitions currently
        stored inside the list. 
        '''
        return self._num_transitions_stored

    def addStateTransition(self, transition):
        '''
        Adds state transition to self._episodes
        :param tuple transition: transition to be added
        '''
        self._episode.append(transition)

    def add_experience(self, transition, meta_strategy=None):
        ''' 
        Method used to add state transitions to the replay memory. 
        :param transition: Tuple containing state transition tuple
        '''
        # Add transition to episode list:
        self.addStateTransition(transition)
        if transition[4] == 1: # If the transition is terminal
            self._num_transitions_stored += len(self._episode)
            while self.isFull():
                deletedEpisode = self._episodes.popleft() # Pop first entry if RM is full
                self._num_transitions_stored -= len(deletedEpisode)
            self._episodes.append(self._episode) # Store episode
            self._episode = [] # Reset for next episode

    def isFull(self):
        '''
        :return bool: True if RM is full, false otherwise
        '''
        return True if self._num_transitions_stored >= self.c.erm.size else False
            

    def get_mini_batch(self):
        '''
        Method returns a mini-batch of sample traces.
        :return list traces: List of traces
        '''
        samples = [] # List used to store n traces used for sampling
        
        # Episodes are randomly choosen for sequence sampling:

        indexes = [random.randrange(len(self._episodes)) for i in range(self.c.erm.batch_size)]
        # From each of the episodes a sequence is selected:
        if len(self.c.dim) == 3:
 	    # From each of the episodes a sequence is selected:
            for i in indexes:
                samples.append(self._episodes[i][random.randint(0, len(self._episodes[i])-1)])
        else:
            for i in indexes:
                transition = random.randint(self.c.erm.sequence_len, len(self._episodes[i]))
                # State-trajectories are stored in lists.
                # Storing these each time provides memory to 
                # run multiple training runs in paralle at the 
                # cost of efficiency.
                o_t = []  
                o_tp1 = []
                for j in range(transition-self.c.erm.sequence_len, transition):
                    o_t.append(self._episodes[i][j][0])
                    o_tp1.append(self._episodes[i][j][1])
                    #print(self._episodes[i][j][0])
                #print(len(o_t))
                transitionTuple = np.copy(self._episodes[i][transition-1])
                #print(transitionTuple.shape)
                #print(o_t)
                #print(transitionTuple)
                #print(o_t)
                transitionTuple[0] = 0.
                transitionTuple[1] = 0.
                #print(o_t, o_tp1)
                samples.append(transitionTuple)
                #print(transitionTuple)
        return samples
   
    def getAllUnzippedOutcomes(self):
        '''
        :return unzipped samples
        '''
        samples = [] # List used to store n traces used for sampling
        for i in range(len(self._episodes)):
            samples.append(self._episodes[i][(len(self._episodes[i])-1)])
        return zip(*samples)

    def getUnzippedSamples(self):
        '''
        :return unzipped samples
        '''
        # Samples are obtained from the replay memory and unzipped
        samples = self.get_mini_batch()
        return zip(*samples)
