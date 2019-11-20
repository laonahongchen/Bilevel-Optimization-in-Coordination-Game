import highway_env
import gym
from malib.agents.agent_factory import *
from malib.samplers.sampler import MASampler
from malib.environments import DifferentialGame
from malib.logger.utils import set_logger
from malib.utils.random import set_seed
from malib.trainers import MATrainer

env = gym.make("merge-v0")
print(len(env.all_vehicles))
agent_setting = ''

agent_num = 4
batch_size = 128
training_steps = 30
exploration_step = 1000
hidden_layer_sizes = (10, 10)
max_replay_buffer_size = 1e5

agents = []
for i in range(agent_num):
    agent = []
    #agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
    agents.append(agent)

sampler = MASampler(agent_num)
sampler.initialize(env, agents)

print(len(sampler.env.all_vehicles))



trainer = MATrainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions'],
)

trainer.run()