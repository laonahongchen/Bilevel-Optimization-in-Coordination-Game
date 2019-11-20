# Created by yingwen at 2019-03-16

# from malib.agents.agent_factory import *
# from malib.samplers.sampler import MASampler
# from malib.environments import DifferentialGame
# from malib.logger.utils import set_logger
# from malib.utils.random import set_seed
# from malib.trainers import MATrainer


from bilevel_pg.bilevelpg.environment.matrix_game import MatrixGame
from bilevel_pg.bilevelpg.environment.differential_game import DifferentialGame
from bilevel_pg.bilevelpg.environment.grid_game import GridEnv
from bilevel_pg.bilevelpg.environment.complex_grid import ComplexGridEnv
from bilevel_pg.bilevelpg.trainer.bilevel_trainer import Bilevel_Trainer
from bilevel_pg.bilevelpg.utils.random import set_seed
from bilevel_pg.bilevelpg.logger.utils import set_logger
from bilevel_pg.bilevelpg.samplers.sampler import MASampler, BiPGSampler, Bi_continuous_Sampler
from bilevel_pg.bilevelpg.samplers.bilevel_q_pg_sampler import BiSampler
from bilevel_pg.bilevelpg.agents.agent_factory import *




# set_seed(0)

agent_setting = 'bilevel'
game_name = 'coordination_same_action_with_preference'
suffix = f'{game_name}/{agent_setting}'

set_logger(suffix)

agent_num = 2
action_num = 2
batch_size = 512
training_steps = 30000
exploration_step = 500
hidden_layer_sizes = (20, 20)
max_replay_buffer_size = 10000

# env = DifferentialGame(game_name, agent_num)
# env = GridEnv()
# env = ComplexGridEnv()
env = MatrixGame(game_name, agent_num, action_num)

# for round in range(100):

agents = []

# for i in range(agent_num):
#     agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
#     agents.append(agent)

# agent_0 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_0 = get_leader_q_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_follower_q_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_1 = get_follower_stochasitc_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_0 = get_independent_q_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_independent_q_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)

# agent_0 = get_maddpg_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_follower_deterministic_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)



agents.append(agent_0)
agents.append(agent_1)


# sampler = MASampler(agent_num)
# sampler = BiPGSampler(agent_num)
# sampler = Bi_continuous_Sampler(agent_num)
sampler = BiSampler(agent_num)
sampler.initialize(env, agents)

trainer = Bilevel_Trainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions_q_pg'], batch_size=batch_size
)

trainer.run()
