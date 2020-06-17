from __future__ import print_function
from scipy.misc import imsave 
#from store import writeTo
from math import exp
import numpy as np
import inspect
import xxhash
import time
try:
    set
except NameError:
    from sets import Set as set

class Temperature:

    def __init__(self, config, nActions, net, sess, replay_memory):
        '''
        :param dict config: Supplies hyperparameters
        :param int nActions: Number of actions used for indexing
        '''
        self.c = config
        self._nActions = nActions 
        if self.c.leniency.hashing is not 'xxhash':
            self._replay_memory = replay_memory
            self._time = 0
        self._net = net
        self._sess = sess
        self.temperatures = [dict() for i in range(nActions)]
        self._maxTemperature = self.c.leniency.max
        self._maxTemperatureDecay = self.c.leniency.max_temperature_decay
        if self.c.leniency.method.name == 'TDS':
            self.initTempDecayTrace()
        self.tGraph = set()
        self.terminalHashkeys = []
        self.eps = 0
        self.mask = np.zeros((13,13,3))
        for i in range(4,9): 
            for j in range(4,9): 
                self.mask[i][j][:] = 1.0

    def getMaxTemperature(self):
        '''
        :return float: Max temperature 
        '''
        return self._maxTemperature        

    def initTempDecayTrace(self):
        '''
        Temperatuer Decay Schedule (TDS) initialisaion. Creates 
        a schedule which can be used to retroactively decay temperature
        values, decaying temperatures near terminal near terminal states
        at a faster rate commpared to earlier transition.
        '''
        self._betas = []
        beta = self.c.leniency.method.exp_rho  
        for i in range(self.c.leniency.method.trace_len):
            t = exp(beta)
            if t > self.c.leniency.method.max_decay:
                t = self.c.leniency.method.max_decay
            self._betas.append(t)
            beta *= self.c.leniency.method.tdf

    def getDiffusion(self):
        '''
        :return float: current diffusion value
        '''
        return self.c.leniency.method.diffusion
       

    def trainNet(self):
        '''
        Called to train hash key network
        '''
        o_t, _, _, _, _, _, _, _, _ = self._replay_memory.getUnzippedSamples()
        if self.c.cnn.format == "NHWC" and len(self.c.dim) == 2:
            o_t = np.moveaxis(o_t, 1, -1)
        optDict = {self._net.inputs: o_t}
        _, outputs, loss = self._sess.run([self._net.optim, self._net.outputs, self._net.ae_loss], optDict)

    def saveStateImage(self, img, name='outfile.jpg', r=8):
        '''
        Method can be used to save a np array as a padded image.
        :param np.array img: contains image to be saved
        :param string name: name of file to be saved
        :param padding: number of times pixels are to be repeated
        '''
        imsave(name, np.repeat(np.repeat(img, r, axis=0), r, axis=1))
 
    def getHash(self, s_t):
        '''
        Get hash key for state s_t. For smaller
        discrete environments xxhash can be used,
        while larger environments require the 
        grouping of semantically simmilar states.
        :param tensor s_t: state for which key needs to be obtained
        :return int: Hash key for state
        '''

        if self.c.cnn.format == "NHWC":
            s_t = np.moveaxis([s_t], 0, -1)
        if self.c.leniency.hashing == 'AutoEncoder':
            self._time += 1
            if self._time%4 == 0 and self._time > self.c.leniency.aestart and self._time < self.c.leniency.aeend:
                self.trainNet()
        if self.c.leniency.hashing is not 'xxhash':
            index_array =  self._net.fetch('simHash', self._sess, [s_t])
            index = 0
            x = 1
            for i in range(len(index_array)):
                if index_array[i] > 0:
                    index += x              
                x *= 10
            return index
        else:
            return hash(s_t.tostring())

    def getTemperature(self, o_t, action):
        """
        Get temperature for state action pair.
        :param tensor o_t: observation
        :param int action: action 
        """ 
        index = self.getHash(o_t)
        return self.getTemperatureUsingIndex(index, action), index

    def getTemperatureUsingIndex(self, index, action):
        """
        Get temperature for state action pair using index key.
        :param int index: Hash key for a state 
        :param int action: action 
        """ 
        # If the index key already exists in the hash table:
        if index in self.temperatures[action]:
            # If temporature is less than the decaying max temperature:
            if self._maxTemperature > self.temperatures[action][index]:
                return self.temperatures[action][index]
            # Else return the max temperature:
            else:
                return self._maxTemperature
        else:
            # If undefined set temperature to current max temperature
            self.temperatures[action][index] = self._maxTemperature
            return self._maxTemperature 
    
    def getAvgTempUsingIndex(self, index):
        '''
        Calculates average temperature for a state based on index.
        :param int index: Hash key for a state.
        :return float: average temperature for the state belongin to index.
        '''
        temperatures = []
        for i in range(self._nActions):
            temperatures.append(self.getTemperatureUsingIndex(index, i))

        return (sum(temperatures) / float(self._nActions))

    def getAvgTemp(self, observation):
        '''
        Returns average temperature for a state.
        :param tensor observation: Observation for which the avg temperature is calcuted 
        :return float: Avg temperature value for state       
        '''
        index = self.getHash(observation)
        return self.getAvgTempUsingIndex(index)

    def decayMaxTemperature(self):
        '''
        Decays the max (global) temperature
        '''
        self._maxTemperature *= self._maxTemperatureDecay  

    def applyATF(self, index, index_tp1, action, terminal):
        '''
        Decays temperature for a observation action pair using
        averge temperature folding (ATF):
        :param int index: Hash key of observation
        :param int index_tp1: Hask key of state a time + 1
        :param int action: Action used at time t
        :param int terminal: 1 if transition was terminal, 0 otherwise 
        '''
        if index in self.temperatures[action]:
            if terminal == 1:
                self.temperatures[action][index] *= self.c.leniency.method.beta
            else:
                self.temperatures[action][index] = self.c.leniency.method.beta *\
                                                   (((1-self.c.leniency.method.diffusion)*\
                                                   self.temperatures[action][index]) +\
                                                   (self.c.leniency.method.diffusion *\
                                                    self.getAvgTempUsingIndex(index_tp1)))
        else:
            self.temperatures[action][index] = self._maxTemperature 
     

    def updateTemperatures(self, episode):
        '''
        Decays temperatures for state actions pairs in episode.
        '''
        decIndex = 0
        self.decayMaxTemperature()
        for transition in reversed(episode):
            s_t, s_tp1, action, _, terminal, idx, idx_tp1, _, _ = transition
            if terminal == 1: 
                self.terminalHashkeys.append(idx_tp1)
                print("key2: " +str(transition[5]))

            if self.c.leniency.method.name == 'TDS':
                self.applyTDS(idx, action, decIndex)
                decIndex += 1
            elif  self.c.leniency.method.name == 'ATF':
                self.applyATF(idx, idx_tp1, action, terminal)
        tempDict = {}
   
    def setRunDir(self, folder):
        '''
        Used to set save dir
        '''
        self.folder = folder
 
    def incEps(self):
        '''
        Increments episode counter.
        '''
        self.eps += 1

    def applyTDS(self, index, action, decIndex):
        '''
         Decay temperature using TDS (Temperature Decay Schedule)
        :param int index: Hash key of state
        :param int action: Action used at time t
        :param int decIndex: Index of beta value to use for decay
        '''
        #for action in range(self._nActions):
        if index in self.temperatures[action] and decIndex < self.c.leniency.method.trace_len:
            self.temperatures[action][index] *= self._betas[decIndex]
        else:
            self.temperatures[action][index] = self._maxTemperature

        if self._maxTemperature < self.temperatures[action][index]:
            self.temperatures[action][index] = self._maxTemperature

