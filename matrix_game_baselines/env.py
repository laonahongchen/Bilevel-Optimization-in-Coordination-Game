from env.openai_gym.load import load

class Environment(object):

    """ Environment """
    def __init__(self, name):
        self.__env, self.__dim, self.__out, self.__bound, self.__max = gym.load(name) 

    @property
    def env(self):
        return self.__env

    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    @property
    def bound(self):
        return self.__bound

    @property    
    def maxin(self):
        return self.__maxin

