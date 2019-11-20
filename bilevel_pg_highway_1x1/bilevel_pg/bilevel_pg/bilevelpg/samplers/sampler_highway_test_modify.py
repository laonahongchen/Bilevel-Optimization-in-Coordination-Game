import numpy as np

# from malib.logger import logger, tabular
from bilevel_pg.bilevelpg.logger import logger, tabular
import tensorflow as tf

num_sample = 10

class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class MASampler_Test(Sampler):
    def __init__(self, agent_num, leader_num, follower_num, leader_action_num = 2, max_path_length=20, min_pool_size=10e4, batch_size=64, global_reward=False, **kwargs):
        # super(MASampler, self).__init__(**kwargs)
        self.agent_num = agent_num
        self.leader_num = leader_num
        self.leader_action_num = leader_action_num
        self.follower_num = follower_num
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._global_reward = global_reward
        self._path_length = 0
        self._path_return = np.zeros(self.agent_num)
        self._last_path_return = np.zeros(self.agent_num)
        self._max_path_return = np.array([-np.inf] * self.agent_num, dtype=np.float32)
        self._n_episodes = 0
        self._total_samples = 0
        # self.episode_rewards = [0]  # sum of rewards for all agents
        # self.agent_rewards = [[0] for _ in range(self.agent_num)] # individual agent reward
        self.step = 0
        self._current_observation_n = None
        self.env = None
        self.agents = None
        self.count = 0
        self.level_agent_num = 2
        self.leader_idx = 0
        self.follower_idx = 1
        self.correct_merge_per_array = []

    def set_policy(self, policies):
        for agent, policy in zip(self.agents, policies):
            agent.policy = policy

    def batch_ready(self):
        enough_samples = max(agent.replay_buffer.size for agent in self.agents) >= self._min_pool_size
        return enough_samples

    def random_batch(self, i):
        return self.agents[i].pool.random_batch(self._batch_size)

    def initialize(self, env, agents, train_agents):
        self._current_observation_n = None
        self.env = env
        self.agents = agents
        self.train_agents = train_agents

    def sample(self, explore=False):
        self.step += 1
        # print(self._current_observation_n)
        
        if self._current_observation_n is None:
            # print('now updating')
            self._current_observation_n = self.env.reset()
            # print(self._current_observation_n)
        
        action_n = []

        supplied_observation = []
        #print(self._current_observation_n.shape)
        # print(self._current_observation_n)
        # print(self._current_observation_n.shape)
        #mix_observe_0 = tf.one_hot(self._current_observation_n[0], self.env.num_state)

        if explore:
            for i in range(self.agent_num):
                if self.env.is_vehicles_valid[i]:
                    action_n.append([np.random.randint(0, self.env.action_num)])
                else:
                    action_n.append([0])  # idle action 
            for i in range(self.leader_num):
                supplied_observation.append(self._current_observation_n[i])
            for i in range(self.leader_num, self.env.agent_num):      
                v = self.env.road.vehicles[i]
                #print(v)
                closest_leaders = self.env.road.closest_leader_vehicles_to(v, self.level_agent_num)
                # ordered leader idx according to distance to leader vehicle
                
                #print(closest_leaders)
                leader_actions_concat = []
                for leader in closest_leaders:
                    leader_actions_concat = np.append(leader_actions_concat, tf.one_hot(action_n[leader.index], self.env.action_num))
                    #print(leader_actions_concat)
                #print(leader_actions_concat.reshape(1, -1))
                #print(len(self._current_observation_n[0]))
                mix_obs = np.hstack((self._current_observation_n[i], leader_actions_concat)).reshape(1, -1)
                #print(mix_obs.shape[1])
                supplied_observation.append(mix_obs)
        else:
            for i in range(self.leader_num):
                supplied_observation.append(self._current_observation_n[i])
                if self.env.is_vehicles_valid[i]:
                    action_n.append(self.train_agents[self.leader_idx].act(self._current_observation_n[i].reshape(1, -1)))
                else:
                    action_n.append([0])  #idle action
            
            for i in range(self.leader_num, self.env.agent_num):
                
                v = self.env.road.vehicles[i]
                close_leaders = self.env.road.closest_leader_vehicles_to(v, self.level_agent_num)
                # ordered leader idx according to distance to leader vehicle
                leader_actions_concat = []
                for leader in close_leaders:
                    leader_actions_concat = np.append(leader_actions_concat, tf.one_hot(action_n[leader.index], self.env.action_num))
                mix_obs = np.hstack((self._current_observation_n[i], leader_actions_concat)).reshape(1, -1)
                supplied_observation.append(mix_obs)
                if self.env.is_vehicles_valid[i]:
                    follower_action = self.train_agents[self.follower_idx].act(mix_obs.reshape(1, -1))
                    action_n.append(follower_action)
                else:
                    action_n.append([0]) # idle action
            # print('action shape:')
            # print(action_0.shape, action_1.shape, np.array(action_n).shape)
        #supplied_observation.append(mix_observe_0)
        #supplied_observation.append(mix_observe_1)

        action_n = np.asarray(action_n)
        #print(supplied_observation)
        next_observation_n, reward_n, done_n, info = self.env.step(action_n)
        
        self.env.render()
        # print('done:')
        # print(type(done_n[0]))
        if self._global_reward:
            reward_n = np.array([np.sum(reward_n)] * self.agent_num)

        # if action_n[0] == 0:
        #     print('explore up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     print(action_n)
        #     print(reward_n)

        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32)
        self._total_samples += 1

    
        self._current_observation_n = next_observation_n
        # for i, rew in enumerate(reward_n):
        #     self.episode_rewards[-1] += rew
        #     self.agent_rewards[i][-1] += rew
        
        if self.step % (25 * 1000) == 0:
            print("steps: {}, episodes: {}, mean episode reward: {}".format(
                        self.step, len(reward_n), np.mean(reward_n[-1000:])))

        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()
            #self.env.merge_count += 1
            self.correct_merge_per_array.append(self.env.correct_merge_count / self.env.merge_count)
            self._max_path_return = np.maximum(self._max_path_return, self._path_return)
            self._mean_path_return = self._path_return / self._path_length
            self._last_path_return = self._path_return
            self._path_length = 0

            self._path_return = np.zeros(self.agent_num)
            self._n_episodes += 1
            self.log_diagnostics()
            logger.log(tabular)
            logger.dump_all()
        else:
            self._current_observation_n = next_observation_n

    def log_diagnostics(self):
        for i in range(self.agent_num):
            tabular.record('max-path-return_agent_{}'.format(i), self._max_path_return[i])
            tabular.record('mean-path-return_agent_{}'.format(i), self._mean_path_return[i])
            tabular.record('last-path-return_agent_{}'.format(i), self._last_path_return[i])
        tabular.record('episodes', self._n_episodes)
        tabular.record('episode_reward', self._n_episodes)
        tabular.record('total-samples', self._total_samples)