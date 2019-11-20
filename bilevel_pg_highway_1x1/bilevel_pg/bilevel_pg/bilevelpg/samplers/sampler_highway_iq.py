import numpy as np

# from malib.logger import logger, tabular
from bilevel_pg.bilevelpg.logger import logger, tabular
import tensorflow as tf
from highway_env import utils
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


class MASampler(Sampler):
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
        self.step = 0
        self._current_observation_n = None
        self.env = None
        self.agents = None
        self.count = 0
        self.level_agent_num = 2
        self.leader_idx = 0
        self.follower_idx = 1
        self.correct_merge = 0
        self.idle_action = 0
        self.rewards_record = []

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
        observations = np.zeros((2, self.env.num_state))
        next_observations = np.zeros((2, self.env.num_state))
        if self.env.sim_step >= self.env.num_state - 3:
            print('wrong')
        observations[0][self.env.sim_step] = 1
        observations[1][self.env.sim_step] = 1
        next_observations[0][self.env.sim_step + 1] = 1
        next_observations[1][self.env.sim_step + 1] = 1
        relative_info = np.zeros((2, 2))
        speed_max = 70
        velocity_range = 2 * 70
        x_position_range = speed_max
        delta_dx = self.env.road.vehicles[1].position[0] - self.env.road.vehicles[0].position[0]
        delta_vx = self.env.road.vehicles[1].velocity - self.env.road.vehicles[0].velocity 
        relative_info[0][0] = utils.remap(delta_dx, [-x_position_range, x_position_range], [-1, 1])
        relative_info[0][1] = utils.remap(delta_vx, [-velocity_range, velocity_range], [-1, 1])
        relative_info[1][0] = -relative_info[0][0]
        relative_info[1][1] = -relative_info[0][1]
        observations[:, -2:] = relative_info

        if explore:
            for i in range(self.agent_num):
                action_n.append([np.random.randint(0, self.env.action_num)])

            for i in range(self.agent_num):
                supplied_observation.append(observations[i])
        else:
            for i in range(self.agent_num):
                supplied_observation.append(observations[i])
                action_n.append(self.train_agents[i].act(observations[i].reshape(1, -1)))
        action_n = np.asarray(action_n)
        
        pres_valid_conditions_n = []
        next_valid_conditions_n = []

        #self.env.render()
        '''
        for i in range(3):
            for j in range(3):
                print("q value for upper agent ", i, j, self.train_agents[0]._qf.get_values(np.hstack((observations[0], tf.one_hot(i, self.env.action_num))).reshape(1, -1)))
        print()
        for i in range(3):
            for j in range(3):
                print("q value for lower agent ", i, j, self.train_agents[1]._qf.get_values(np.hstack((observations[0], tf.one_hot(i, self.env.action_num))).reshape(1, -1)))
        
        #print('a0 = ', action_n[0])
        #print('a1 = ', action_n[1])
        '''
        next_observation_n, reward_n, done_n, info = self.env.step(action_n)
        delta_dx = self.env.road.vehicles[1].position[0] - self.env.road.vehicles[0].position[0]
        delta_vx = self.env.road.vehicles[1].velocity - self.env.road.vehicles[0].velocity 
        relative_info[0][0] = utils.remap(delta_dx, [-x_position_range, x_position_range], [-1, 1])
        relative_info[0][1] = utils.remap(delta_vx, [-velocity_range, velocity_range], [-1, 1])
        relative_info[1][0] = -relative_info[0][0]
        relative_info[1][1] = -relative_info[0][1]
        next_observations[:, -2:] = relative_info
        
        if self._global_reward:
            reward_n = np.array([np.sum(reward_n)] * self.agent_num)

        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32)
        self._total_samples += 1
        
        for i, agent in enumerate(self.agents):            

            agent.replay_buffer.add_sample(
                # observation=self._current_observation_n[i].astype(np.float32),
                observation=supplied_observation[i],
                # action=action_n[i].astype(np.float32),
                action = tf.one_hot(action_n[i], self.env.action_num),
                # reward=reward_n[i].astype(np.float32),
                reward = np.float32(reward_n[i]),
                # terminal=done_n[i].astype(np.float32),
                terminal=np.float32(done_n[i]),
                # next_observation=next_observation_n[i].astype(np.float32),
                next_observation=np.float32(next_observations[i]),
            )
        
             
        self._current_observation_n = next_observation_n
       
        if self.step % (25 * 1000) == 0:
            print("steps: {}, episodes: {}, mean episode reward: {}".format(
                        self.step, len(reward_n), np.mean(reward_n[-1000:])))

        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()

            #self.env.merge_count += 1
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