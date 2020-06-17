from environment import Environment
from stats import mkRunDir
from config import Config
from copy import deepcopy
from agent import Agent
import tensorflow as tf
import csv
import numpy as np

flags = tf.app.flags

# Environments:
# Apprentice Firemen:
# -ApprenticeFiremen_V{1,2}_C{INT}_{DET,PS,FS}_AP{1,2,3,4}
flags.DEFINE_string('environment', 'ApprenticeFiremen_V1_C0_DET', 'Environment.') 
flags.DEFINE_string('madrl', 'leniency', 'Options: None, hysteretic, leniency and nui')
flags.DEFINE_string('format', 'NHWC', 'Format for conv (NHWC or NCHW)')
flags.DEFINE_string('processor', '/gpu:0', 'GPU/CPU.')
flags.DEFINE_integer('save_steps', 100, 'Save model every n steps..')
flags.DEFINE_integer('steps', 30000, 'Max steps per episdoe.')
flags.DEFINE_integer('episodes', 30000, 'Number of episodes.')
flags.DEFINE_integer('agents', 2, 'Number of agents.')
flags.DEFINE_boolean('eval_render', False, 'Render environment.')
flags.DEFINE_boolean('render', False, 'Render environment.')
flags.DEFINE_float('sequenceid', 1, 'Used for parallel runs.')
flags.DEFINE_float('runid', 1, 'Used for parallel runs.') 
flags.DEFINE_float('gamma', 0, 'Discount rate.') 
FLAGS = flags.FLAGS

agent_num = 2
action_num = 3
explore_step = 500
env_dim = [1]
payoff_matrix = np.zeros((agent_num, action_num, action_num))

payoff_matrix[0] = [[20, 0, 0], [30, 10, 0], [0, 0, 5]]
payoff_matrix[1] = [[15, 0, 0], [0, 5, 0], [0, 0, 10]]

avg_rewards = np.zeros(agent_num, )
test_times = 10
run_times = 100
'''
payoff_matrix[0] =  [[15, 10, 0], [10, 10, 0], [0, 0, 30]]
payoff_matrix[1] = [[15, 10, 0], [10, 10, 0], [0, 0, 30]]
'''

def step(actions):
    rewards = np.zeros(agent_num, )
    observations = np.array([0., 0.], dtype=float)
    terminal = True
    rewards[0] = payoff_matrix[0][actions[0], actions[1]]
    rewards[1] = payoff_matrix[1][actions[0], actions[1]]

    return observations, rewards, terminal

# Environment is instantiated
#env = Environment(FLAGS)
#config = Config(env_dim, action_num, meta_actions=3, madrl=FLAGS.madrl, gpu=FLAGS.processor, gamma=FLAGS.gamma)
config = Config(env_dim, action_num, meta_actions=3, madrl=FLAGS.madrl, gpu=FLAGS.processor, gamma=FLAGS.gamma)

# Run dir and stats csv file are created
#statscsv, folder = mkRunDir(env, config, FLAGS.sequenceid, FLAGS.runid)

# Agents are instantiated
agents = []
for i in range(FLAGS.agents): 
    agent_config = deepcopy(config)
    # Copy of config can be modified here. 
    agents.append(Agent(agent_config))
    '''
    f = open(folder + 'agent' + str(i) + '_config.txt','w')
    f.write(str(vars(agent_config)))
    f.close()
    '''


   
    
for j in range(FLAGS.episodes):
    observations = np.array([0., 0.], dtype=float)
    # Load action for each agent
    actions = []
    for agent, observation in zip(agents, observations):
        if FLAGS.render: env.render()
        actions.append(agent.move(observation))            
        agent.opt()
    print(actions)
    # Feedback:
    #observations, rewards, t, equipment, reduced_observations = env.step(actions)
    observations, rewards, t = step(actions)
    
    for agent, o, r in zip(agents, observations, rewards):
        if FLAGS.madrl is 'leniency':
            agent.feedback(r, t, o)
        elif FLAGS.madrl is 'nui':
            agent.feedback(r, t, o, meta_action=e)
        else:
            agent.feedback(r, t, o)
        #print(agnet.drl.replay_buffer.getSize())

    print('step ', j)
    print('last return for agent 0: ', rewards[0])
    print('last return for agent 1: ', rewards[1])
    print('')
    print('')



converge_actions = np.zeros(agent_num, )
converge_matrix = np.zeros(action_num, action_num)
converge_rewards = np.zeros(agent_num, )

for t in range(test_times):
    action_1 = agents[0].move(0.0)
    action_2 = agents[1].move(0.0)
    converge_matrix[action_1][action_2] += 1
idx = np.argmax(converge_matrix)
converge_actions[0] = idx / action_num
converge_actions[1] = idx - action_num * x
converge_rewards[0] = payoff_matrix[0][converge_actions[0]][converge_actions[1]]
converge_rewards[1] = payoff_matrix[1][converge_actions[0]][converge_actions[1]]
avg_rewards[0] += converge_rewards[0]
avg_rewards[1] += converge_rewards[1]


'''
with open(statscsv, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
    writer.writerow(env.stats())
print(env.stats())
'''



