import highway_env
import gym

from bilevel_pg.bilevelpg.trainer.bilevel_trainer_highway_rommeo import Bilevel_Trainer
from bilevel_pg.bilevelpg.utils.random import set_seed
from bilevel_pg.bilevelpg.logger.utils import set_logger
from bilevel_pg.bilevelpg.samplers.sampler_highway_rommeo import MASampler
from bilevel_pg.bilevelpg.agents.rommeo_agents import ROMMEOAgent
from bilevel_pg.bilevelpg.policy.gaussian_policy import GaussianMLPPolicy
from bilevel_pg.bilevelpg.value_functions import MLPValueFunction
from bilevel_pg.bilevelpg.replay_buffers_highway import IndexedReplayBuffer
from bilevel_pg.bilevelpg.explorations.ou_exploration import OUExploration
from bilevel_pg.bilevelpg.policy import DeterministicMLPPolicy
import numpy as np
import gym
import tensorflow as tf

def get_rommeo_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    #observation_space = env.env_specs.observation_space[agent_id]
    #action_space = env.env_specs.action_space[agent_id]
    observation_space = np.zeros(env.num_state, )
    action_space = np.zeros(env.action_num, )
    opponent_action_shape = (env.env_specs.action_space.opponent_flat_dim(agent_id),)
    print(opponent_action_shape, 'opponent_action_shape')
    print(observation_space.shape, action_space.shape)
    return ROMMEOAgent(
        env_specs=env.env_specs,
        policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape, opponent_action_shape),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id),
            smoothing_coefficient=0.5
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, action_space.shape, opponent_action_shape),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          max_replay_buffer_size=max_replay_buffer_size,
                                          opponent_action_dim=opponent_action_shape[0],
        ),
        opponent_policy=GaussianMLPPolicy(
            input_shapes=(observation_space.shape,),
            output_shape=opponent_action_shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='opponent_policy_agent_{}'.format(agent_id)
        ),
        gradient_clipping=10.,
        agent_id=agent_id,
    )

for seed in range(80, 81):
    
    set_seed(seed)

    agent_setting = 'bilevel'
    #agent_setting = 'maddpg'
    #game_name = 'coordination_same_action_with_preference'
    #game_name = 'climbing_3x3'
    #game_name = 'climbing'
    game_name = 'merge_env'
    suffix = f'{game_name}/{agent_setting}'

    set_logger(suffix)

    env = gym.make("merge-v0")

    env.agent_num = 2
    env.leader_num = 1
    env.follower_num = 1
    action_num = 3
    batch_size = 32
    training_steps = 500000
    exploration_step = 0
    hidden_layer_sizes = (30, 30)
    max_replay_buffer_size = 1000

    #env = MatrixGame(game_name, agent_num, action_num)
    # env = GridEnv()
    agents = []
    train_agents = []
    # for i in range(agent_num):
    #     agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
    #     agents.append(agent)



    for i in range(env.agent_num):
        agents.append(get_rommeo_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size))
    train_agents.append(get_rommeo_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size))
    train_agents.append(get_rommeo_stochasitc_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size))

    sampler = MASampler(env.agent_num, env.leader_num, env.follower_num)
    sampler.initialize(env, agents, train_agents)

    trainer = Bilevel_Trainer(
        seed=seed, env=env, agents=agents, train_agents=train_agents, sampler=sampler,
        steps=training_steps, exploration_steps=exploration_step,
        extra_experiences=['target_actions'], batch_size=batch_size
    )
    '''
    restore_path = './models/agents_bilevel_1x1_'+'seed'+str(seed)+'.pickle'
    trainer.restore(restore_path)
    sampler.train_agents = trainer.train_agents
    '''
    trainer.run()
    trainer.save()