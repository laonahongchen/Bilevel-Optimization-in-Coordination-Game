"""
The trainer for multi-agent training.
"""
import pickle
# from malib.trainers.utils import *
from bilevel_pg.bilevelpg.trainer.utils_highway_maddpg import *
import time


class Tester:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, env, agents, train_agents, sampler,
            batch_size=128,
            steps=10000,
            exploration_steps=100,
            training_interval=1,
            extra_experiences=['target_actions'],
            save_path=None,
    ):
        self.env = env
        self.agents = agents
        self.train_agents = train_agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.exploration_steps = exploration_steps
        self.training_interval = training_interval
        # print(training_interval)
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path
        self.epsilon = 0.1

    def setup(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler

    def sample_batches(self):
        assert len(self.agents) > 1
        batches = []
        indices = self.agents[0].replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        return batches

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    def run(self):
        #print('trainer_start')
        while(True):
            #print("step ", step)
            '''
            if (np.random.uniform(0, 1) < self.epsilon):
                self.sampler.sample(explore=True)
            else:
                self.sampler.sample()
            '''
            self.sampler.sample()


            if self.env.merge_count > 100:
                break
            # print('train finish')
            # print(int(round(time.time() * 1000)))

    def save(self):
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, restore_path):
        with open(restore_path, 'rb') as f:
            self.train_agents = pickle.load(f)

    def resume(self):
        pass

    def log_diagnostics(self):
        pass