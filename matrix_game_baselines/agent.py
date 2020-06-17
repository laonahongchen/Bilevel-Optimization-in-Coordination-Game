import itertools

class Agent(object):
    # Itertools used to create an unique id for each agent:
    mkid = next(itertools.count())

    """ Agent """
    def __init__(self, config):
        '''
        :param int agentID: Agent's ID
        :param dict config: Dictionary containing hyperparameters
        '''
        self.c = config
        self.c.id = Agent.mkid
        DRL = self.getDRL(self.getBase()) 
        self.drl = DRL(config)

    def getDRL(self, base):
        '''
        Requested drlis imported.
        :param drl: Base drl upon which the MA-DRL is built
        :param dict config: Dictionary containing hyperparameters
        '''
        if self.c.madrl == 'hysteretic':
           import madrl.hysteretic.hql
           DRL = madrl.hysteretic.hql.HQL_FACTORY(parent=base) 
        elif self.c.madrl == 'leniency': 
           if self.c.drl != 'dqn':
               raise ValueError('Leniency can currently only use dqn as base.')
           else:
               from madrl.leniency.leniency import Leniency as DRL
        elif self.c.madrl == 'nui': 
           if self.c.drl != 'dqn':
               raise ValueError('NUI can currently only use dqn as base.')
           else:
               from madrl.nui.nui_dqn import NUI_DQN as DRL
        elif self.c.madrl == None: # If base drl is to be used without modification: 
            DRL = base
        else:
            raise ValueError('Invalid madrl specified.')

        return DRL

    def saveModel(self, folder, step):
         '''
         Save the model.
         '''
         self.drl.saveModel(folder, step)
    
    def restoreModel(self, folder, model): 
        '''
        Restore Model
        '''
        self.drl.restoreModel(folder, model)

    def opt(self):
        '''
        Called to optimise the agent.
        '''
        self.drl.opt()

    def getSaliencyCoordinates(self, obs, coordinates, location, hw):
        '''
        Used to get saliency coordinates for agent
        :param tensor: Observation
        :param vector: coordinates for which saliency is to be loaded
        :param vector: Agent location
        :param tuple: Height and width
        '''
        return self.drl.getSaliencyCoordinates(obs, coordinates, location, hw) 

    def getSaliency(self, obs):
        '''
        Used to save saliency for agent
        :param tensor: Observation
        '''
        return self.drl.getSaliency(obs) 

    def getBase(self):
        '''
        Requested base drl is imported.
        :param dict config: Dictionary containing hyperparameters
        '''
        if self.c.drl == 'dqn':
            from drl.dqn.dqn import DQN as base
        elif  self.c.drl == 'ddpg':
            from drl.ddpg.ddpg import DDPG as base
        return base

    def move(self, state, explore=True):
        '''
        Returns an action from the deep rl agent instance.
        :param tensor: State/Observation
        :param bool: Explore 
        '''
        return self.drl.move(state, explore)



    def feedback(self, reward, terminal, state, meta_action=None, reduced_observation=None):
        '''
        Feedback is passed to the deep rl agent instance.
        :param float: Reward received during transition
        :param boolean: Indicates if the transition is terminal
        :param tensor: State/Observation
        '''
        if meta_action is None and reduced_observation is None:
            self.drl.feedback(reward, terminal, state)
        elif meta_action is not None:
            self.drl.feedback(reward, terminal, state, meta_action)
        elif reduced_observation is not None:
            self.drl.feedback(reward, terminal, state, reduced_observation)

