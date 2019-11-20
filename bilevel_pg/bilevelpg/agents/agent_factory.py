from bilevel_pg.bilevelpg.agents.bi_follower_pg import FollowerAgent
from bilevel_pg.bilevelpg.agents.bi_follower_ddpg import FollowerDDPGAgent
from bilevel_pg.bilevelpg.agents.bi_leader_pg import LeaderAgent
from bilevel_pg.bilevelpg.agents.maddpg import MADDPGAgent
from bilevel_pg.bilevelpg.policy.base_policy import StochasticMLPPolicy
from bilevel_pg.bilevelpg.value_functions import MLPValueFunction
from bilevel_pg.bilevelpg.replay_buffers import IndexedReplayBuffer
from bilevel_pg.bilevelpg.explorations.ou_exploration import OUExploration
from bilevel_pg.bilevelpg.policy import DeterministicMLPPolicy
from bilevel_pg.bilevelpg.agents.bi_leader_q import LeaderQAgent
from bilevel_pg.bilevelpg.agents.bi_follower_q import FollowerQAgent
from bilevel_pg.bilevelpg.agents.bi_leader_q2 import IQAgent
from bilevel_pg.bilevelpg.agents.td3 import TD3Agent

import tensorflow as tf



def gambel_softmax(x):
    u = tf.random.uniform(tf.shape(x))
    return tf.nn.softmax(x - tf.math.log(-tf.math.log(u)), axis=-1)


def get_independent_q_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    return IQAgent(
        env_specs=env.env_specs,
        policy=None,
        qf=MLPValueFunction(
            input_shapes=(env.num_state + env.action_num,),
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

def get_leader_q_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    return LeaderQAgent(
        env_specs=env.env_specs,
        policy=None,
        qf=MLPValueFunction(
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

def get_follower_q_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    return FollowerQAgent(
        env_specs=env.env_specs,
        policy=None,
        qf=MLPValueFunction(
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
            # output_activation=gambel_softmax,
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

def get_td3_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return TD3Agent(
        env_specs=env.env_specs,
        policy=StochasticMLPPolicy(
            input_shapes=(env.num_state, ),
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            # output_activation=gambel_softmax,
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
            # output_activation=gambel_softmax,
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

    # print(env.env_specs.action_space.shape)
    # print((observation_space.shape, (action_space.shape[0],)))
    # print(observation_space.shape, (env.env_specs.action_space.flat_dim,))

    return FollowerDDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape, (action_space.shape[0],)), # 1 for action1, 1 for state
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(1 + 1, (env.env_specs.action_space.flat_dim,)),
            # input_shapes=(2,),
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


def get_maddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print((env.env_specs.action_space.flat_dim,))
    # print(env.env_specs.action_space.shape)
    return MADDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape, ),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim * 2.5,)),
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