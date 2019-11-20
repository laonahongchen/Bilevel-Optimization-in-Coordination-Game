# Created by yingwen at 2019-03-16

# from malib.agents.agent_factory import *
# from malib.samplers.sampler import MASampler
# from malib.environments import DifferentialGame
# from malib.logger.utils import set_logger
# from malib.utils.random import set_seed
# from malib.trainers import MATrainer


from bilevel_pg.bilevelpg.environment.matrix_game import MatrixGame
from bilevel_pg.bilevelpg.environment.grid_game import GridEnv
from bilevel_pg.bilevelpg.trainer.bilevel_trainer import Bilevel_Trainer
from bilevel_pg.bilevelpg.utils.random import set_seed
from bilevel_pg.bilevelpg.logger.utils import set_logger
from bilevel_pg.bilevelpg.samplers.sampler import MASampler
from bilevel_pg.bilevelpg.agents.bi_follower_pg import FollowerAgent
from bilevel_pg.bilevelpg.agents.bi_leader_pg import LeaderAgent
from bilevel_pg.bilevelpg.agents.maddpg import MADDPGAgent
from bilevel_pg.bilevelpg.policy.base_policy import StochasticMLPPolicy
from bilevel_pg.bilevelpg.value_functions import MLPValueFunction
from bilevel_pg.bilevelpg.replay_buffers import IndexedReplayBuffer
from bilevel_pg.bilevelpg.explorations.ou_exploration import OUExploration
from bilevel_pg.bilevelpg.policy import DeterministicMLPPolicy

import tensorflow as tf


def gambel_softmax(x):
    u = tf.random.uniform(tf.shape(x))
    return tf.nn.softmax(x - tf.math.log(-tf.math.log(u)), axis=-1)


def get_leader_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return LeaderAgent(
        env_specs=env.env_specs,
        policy=StochasticMLPPolicy(
            input_shapes=(env.num_state, ),
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            output_activation=gambel_softmax,
            # preprocessor='LSTM',
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(num_sample * 2 + 1, (env.env_specs.action_space.flat_dim,)),
            # input_shapes=(num_sample * 2 + 1, ),
            input_shapes=(env.num_state + env.action_num * 2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.action_num,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_follower_stochasitc_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return FollowerAgent(
        env_specs=env.env_specs,
        policy=StochasticMLPPolicy(
            input_shapes=(env.num_state + env.action_num, ), # 1 for action1, 1 for state
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            output_activation=gambel_softmax,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(1 + 1, (env.env_specs.action_space.flat_dim,)),
            input_shapes=(env.num_state + env.action_num * 2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state + env.action_num,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.action_num,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_follower_deterministic_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return FollowerAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(env.action_num + 1, ), # 1 for action1, 1 for state
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(1 + 1, (env.env_specs.action_space.flat_dim,)),
            input_shapes=(2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.action_num + 1,
                                          action_dim=1,
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_maddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return MADDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            # input_shapes=(observation_space.shape, ),
            input_shapes=(1,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


set_seed(0)

agent_setting = 'bilevel'
game_name = 'climbing'
suffix = f'{game_name}/{agent_setting}'

set_logger(suffix)

agent_num = 2
action_num = 4
batch_size = 128
training_steps = 50000
exploration_step = 1000
hidden_layer_sizes = (30, 30, 30)
max_replay_buffer_size = 1e5

# env = MatrixGame(game_name, agent_num, action_num)
env = GridEnv()

agents = []

# for i in range(agent_num):
#     agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
#     agents.append(agent)

agent_0 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_1 = get_follower_stochasitc_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agents.append(agent_0)
agents.append(agent_1)

sampler = MASampler(agent_num)
sampler.initialize(env, agents)

trainer = Bilevel_Trainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions'], batch_size=batch_size
)

trainer.run()
