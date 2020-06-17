import tflearn

class Config(object):
        
    def __init__(self,\
        dim,\
        out,\
        drl='dqn',\
        madrl=None,\
        gpu='/gpu:0',\
        conv=False,\
        meta_actions=None,\
        gamma=0,
        optimise=True,
        format="NHWC",
        matrix_game=False):

        # Experience Replay Memory Config
        self.erm = self.Experience_Replay_Memory_Config() 

        # If conv, then convolutional layers are used for feature extraction
        self.cnn = self.CNN_Feature_Extractors(format)
        self.fcfe = self.Fully_Connected_Feature_Extractors()

        # Load config for the required base algorithm
        self.dqn = self.DQN_Config()
        self.ddpg = self.DDPG_Config()

        # If recurrent add memory cell parameters
        self.recurrent = self.Recurrent()
       
        # Exploration
        self.epsgreedy = self.Eps_Greedy()

        # MA-DRL Add-ons:
        self.leniency = self.Leniency_Config()
        self.hysteretic = self.Hysteretic_Config()
        self.nui = self.NUI_Config()

        self.matrix_game = matrix_game
        # Hyperparamters 
        self.__drl          = drl          # DRL algorithm. Can serve as base for MADRL
        self.__optimise     = optimise     # Disables optimiser
        self.__madrl        = madrl        # MA-DRL algorithm
        self.__gpu          = gpu          # CPU, GPU device to be used for training
        self.__outputs      = out          # Number of outputs
        self.__meta_actions = meta_actions # Number of meta actions
        self.__gamma        = gamma        # Discount rate
        self.__use_conv     = conv         # True=ConvNet, False=Fully Connected Net
        self.__dim          = dim          # Set input dimensions
        self.__id           = None
        self.__inc_sync     = False        # Used for incremental sync 
        self.__tau          = 0.01        # Incremental sync rate
        self.__sync_time    = 5000         # Steps between sync for non-inc approach

    def __repr__(self):
        return str(vars(self))

    @property
    def drl(self):
        return self.__drl

    @drl.setter
    def drl(self, value):
        if self.__drl == None:
            self.__drl = value
        else:
            raise Exception("Can't modify drl.")

    @property
    def optimise(self):
        return self.__optimise

    @property
    def madrl(self):
        return self.__madrl

    @madrl.setter
    def madrl(self, value):
        if self.__madrl == None:
            self.__madrl = value
        else:
            raise Exception("Can't modify drl.")

    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, value):
        self.__gpu = value

    @property
    def outputs(self):
        return self.__outputs

    @property
    def meta_actions(self):
        return self.__meta_actions

    @property
    def gamma(self):
        return self.__gamma

    @property
    def use_conv(self):
        return self.__use_conv
     
    @use_conv.setter
    def use_conv(self, value):
        self.__use_conv = value

    @property
    def dim(self):
        return self.__dim

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        self.__id = value

    @property
    def inc_sync(self):
        return self.__inc_sync

    @inc_sync.setter
    def inc_sync(self, value):
        self.__inc_sync = value

    @property
    def tau(self):
        return self.__tau

    @tau.setter
    def tau(self, value):
        self.__tau = value

    @property
    def sync_time(self):
        return self.__sync_time


    class Experience_Replay_Memory_Config(object):

        """ ERM Config """
        def __init__(self):
            self.__type          = 'FIFO'  # Options: FIFO, EPISODIC_FIFO (Ingored for Leniency and NUI)
            self.__size          = 10000           # Transitions for FIFO, Episodes for Episodic_FIFO
            self.__threshold     = 512           # Learning threshold transitions
            self.__batch_size    = 512            # Training batch size
            self.__train_steps   = 1                # Time between replay memory sampling
            self.__sequence_len  = 1                # Sampled transition trajectory length

        def __repr__(self):
            return str(vars(self))

        @property
        def type(self):
            return self.__type

        @type.setter
        def type(self, value):
            self.__type = value

        @property
        def size(self):
            return self.__size

        @size.setter
        def size(self, value):
            self.__size = value

        @property
        def threshold(self):
            return self.__threshold

        @threshold.setter
        def threshold(self, value):
            self.__threshold = value

        @property
        def batch_size(self):
            return self.__batch_size

        @property
        def train_steps(self):
            return self.__train_steps

        @train_steps.setter
        def train_steps(self, value):
            self.__train_steps = value

        @property
        def sequence_len(self):
            return self.__sequence_len

        @sequence_len.setter
        def sequence_len(self, value):
            self.__sequence_len = value

    class Fully_Connected_Feature_Extractors(object):

        """ Used for fully connected networks """
        def __init__(self):
            self.__layers   = [20, 20] # List specifying size of each layer
            self.__init_min = -0.0003   # Weight init min
            self.__init_max = 0.0003    # Weight init max
            self.__normalise = False    # Normalise Inputs
            self.__w_init = tflearn.initializations.truncated_normal()

        def __repr__(self):
            return str(vars(self))

        @property
        def layers(self):
            return self.__layers

        @layers.setter
        def layers(self, value):
            if self.__layers == None:
                self.__layers= value

        @property
        def init_min(self):
            return self.__init_min

        @property
        def init_max(self):
            return self.__init_max

        @property
        def normalise(self):
            return self.__normalise

        @normalise.setter
        def normalise(self, value):
            self.__normalise = value

        @property
        def w_init(self):
            return self.__w_init

        @w_init.setter
        def w_init(self, value):
            self.__w_init = value

    class CNN_Feature_Extractors(object):

        """ ConvNet Hyperparameters """
        def __init__(self, format):
            self.__p        = 'VALID' #Padding
            self.__format   = format
            self.__outdim   = [32,64]
            self.__kernels  = [4,2]
            self.__stride   = [2,1]
            self.__fc       = 1024
            self.__max_in   = 255.0

        def __repr__(self):
            return str(vars(self))

        @property
        def h(self):
            return self.__h

        @property
        def w(self):
            return self.__w

        @property
        def c(self):
            return self.__c

        @property
        def p(self):
            return self.__p

        @property
        def format(self):
            return self.__format

        @property
        def outdim(self):
            return self.__outdim

        @property
        def kernels(self):
            return self.__kernels

        @property
        def stride(self):
            return self.__stride

        @property
        def fc(self):
            return self.__fc

        @property
        def max_in(self):
            return self.__max_in


    class DQN_Config(object):

        """ DQN Hyperparameters """
        def __init__(self):
            self.__double = True             # Double Q-Learning
            self.__max    = 1.0              # max val that output can approximate
            self.__exploration = 'epsGreedy' # Options: CepsGreedy, tBarGreedy
            self.__alpha = 0.0001            # Learning rate

        def __repr__(self):
            return str(vars(self))

        @property
        def double(self):
            return self.__double

        @property
        def max(self):
            return self.__max

        @property
        def exploration(self):
            return self.__exploration

        @property
        def alpha(self):
            return self.__alpha

    class DDPG_Config(object):

        """ DDPG Hyperparameters """
        def __init__(self):
            self.__clr               = 0.001        # Critic Leraning Rate
            self.__alr               = 0.001       # Actor Learning Rate
            self.__w_init            = 0.003        # Weight initialisation
            self.__t1_nodes          = 300          # Critic nodes for term 1
            self.__t2_nodes          = 300          # Critic nodes for term 2
            self.__prev_action_nodes = 300          # Critic nodes for term 2
            self.__upper_bound       = 1.0          # Upper Action bound
            self.__lower_bound       = -1.0         # Lower Action bound

        def __repr__(self):
            return str(vars(self))

        @property
        def clr(self):
            return self.__clr

        @property
        def alr(self):
            return self.__alr

        @property
        def w_init(self):
            return self.__w_init

        @w_init.setter
        def w_init(self, value):
            self.__w_init = value

        @property
        def t1_nodes(self):
            return self.__t1_nodes

        @property
        def t2_nodes(self):
            return self.__t2_nodes
    
        @property
        def upper_bound(self):
            return self.__upper_bound

        @upper_bound.setter
        def upper_bound(self, value):
            self.__upper_bound = value

        @property
        def lower_bound(self):
            return self.__lower_bound

        @lower_bound.setter
        def lower_bound(self, value):
            self.__lower_bound = value


    class Recurrent(object):

        """ Recurrent Config """
        def __init__(self):
            self.__h_size = 512

        def __repr__(self):
            return str(vars(self))

        @property
        def h_size(self):
            return self.__h_size

        @h_size.setter
        def h_size(self, value):
            self.__h_size = value

    class Eps_Greedy(object):

        """ Epsilon-Greedy Exploration """
        def __init__(self):
            self.__initial  = 0.1
            self.__min      = 0
            self.__update   = 1      # Update eps every n episodes
            self.__discount = 0.999999  # Epsilong discount factor

        def __repr__(self):
            return str(vars(self))

        @property
        def initial(self):
            return self.__initial

        @property
        def min(self):
            return self.__min

        @property
        def update(self):
            return self.__update

        @property
        def discount(self):
            return self.__discount

    class Hysteretic_Config(object):

        """ Hysteretic Hyperparameters """
        def __init__(self):
            self.__beta = 0.5 # Percentage of alpha

        def __repr__(self):
            return str(vars(self))

        @property
        def beta(self):
            return self.__beta

        @beta.setter
        def beta(self, value):
            self.__beta = value


    class NUI_Config(object):

        """ NUI (Negative Update Intervals) hyperparameters """
        def __init__(self):
            self.__max_episodes    = 100   # Max episodes per replay memory
            self.__decay_threshold = 50    # Intervals are expanded after specified number of episodes
            self.__decay           = 0.995 # Decay rate applied to lower bound
            self.__eps             = 0.0   # Small value subtracted from max r
            self.__threshold       = 50    # Learning threshold (Number of episodes)

        def __repr__(self):
            return str(vars(self))

        @property
        def threshold(self):
            return self.__threshold
        
        @property
        def max_episodes(self):
            return self.__max_episodes

        @property
        def decay_threshold(self):
            return self.__decay_threshold

        @property
        def decay(self):
            return self.__decay

        @property
        def eps(self):
            return self.__eps

    class Leniency_Config(object):

        """ Leniency related hyperparameters """
        def __init__(self, method='TDS'):
            self.__ase                   = 0.25         # Action Selection Exponent
            self.__max                   = 1.0          # Initial temperature
            self.__min                   = 0.0          # Min temperature value
            self.__tmc                   = 1.0          # Leniency moderatoin coefficient
            self.__hashing               = 'xxhash'     # Options: xxhash, AutoEncoder, L2 (Layer2)
            self.__max_temperature_decay = 0.9998       # Used for global temperature decay
            self.__threshold             = 200000       # Temperature update threshold
            if method == 'TDS':
                self.__method = self.Temperature_Decay_Schedule()
            elif method == 'ATF':
                self.__method = self.Average_Temperature_Folding()
            # AutoEncoder related parameters
            self.__aestart      = 50000                 # Steps after which optimsiation of the ae starts
            self.__aeend        = 250000                # Steps after which the ae is no longer trained

        def __repr__(self):
            return str(vars(self))

        @property
        def aestart(self):
            return self.__aestart

        @property
        def aeend(self):
            return self.__aeend

        @property
        def ase(self):
            return self.__ase

        @property
        def max(self):
            return self.__max

        @property
        def min(self):
            return self.__min

        @property
        def tmc(self):
            return self.__tmc

        @property
        def hashing(self):
            return self.__hashing

        @property
        def max_temperature_decay(self):
            return self.__max_temperature_decay

        @property
        def method(self):
            return self.__method

        @property
        def threshold(self):
            return self.__threshold
        
        @property
        def ase(self):
            return self.__ase

        class Average_Temperature_Folding(object):

            """ ATF Hyperparameters """
            def __init__(self):
                  self.__name            = 'ATF'  # Method name
                  self.__diffusion       = 0.99   # Diffusion used with ATF
                  self.__diffusion_decay = 0.9999 
                  self.__beta            = 0.99   # Temperature Discount Rate

            def __repr__(self):
                return str(vars(self))

            @property
            def name(self):
                return self.__name

            @property
            def diffusion(self):
                return self.__diffusion

            @property
            def diffusion_decay(self):
                return self.__diffusion_decay

            @property
            def beta(self):
                return self.__beta

        class Temperature_Decay_Schedule(object):

            """ TDS Hyperparameters """
            def __init__(self):
                self.__name      = 'TDS'  # Method name
                self.__exp_rho   = -0.01
                self.__tdf       = 0.9    # Trace decay factor
                self.__max_decay = 1.0    # Max decay factor
                self.__trace_len = 10000  # Trace Length

            def __repr__(self):
                return str(vars(self))

            @property
            def name(self):
                return self.__name

            @property
            def exp_rho(self):
                return self.__exp_rho

            @property
            def tdf(self):
                return self.__tdf

            @property
            def max_decay(self):
                return self.__max_decay

            @property
            def trace_len(self):
                return self.__trace_len
