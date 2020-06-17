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
flags.DEFINE_integer('steps', 10000, 'Max steps per episdoe.')
flags.DEFINE_integer('episodes', 10000, 'Number of episodes.')
flags.DEFINE_integer('agents', 2, 'Number of agents.')
flags.DEFINE_boolean('eval_render', False, 'Render environment.')
flags.DEFINE_boolean('render', False, 'Render environment.')
flags.DEFINE_float('sequenceid', 1, 'Used for parallel runs.')
flags.DEFINE_float('runid', 1, 'Used for parallel runs.') 
flags.DEFINE_float('gamma', 0.95, 'Discount rate.') 
FLAGS = flags.FLAGS

agent_num = 2
action_num = 3
env_dim = 1
payoff_matrix = np.zeros((agent_num, action_num, action_num))
payoff_matrix[0] = [[20, 0, 0], [30, 10, 0], [0, 0, 5]]
payoff_matrix[1] = [[15, 0, 0], [0, 5, 0], [0, 0, 10]]

# Environment is instantiated
env = Environment(FLAGS)
#config = Config(env_dim, action_num, meta_actions=3, madrl=FLAGS.madrl, gpu=FLAGS.processor, gamma=FLAGS.gamma)
config = Config(env.dim, env.out, meta_actions=3, madrl=FLAGS.madrl, gpu=FLAGS.processor, gamma=FLAGS.gamma)

# Run dir and stats csv file are created
statscsv, folder = mkRunDir(env, config, FLAGS.sequenceid, FLAGS.runid)

# Agents are instantiated
agents = []
for i in range(FLAGS.agents): 
    agent_config = deepcopy(config)
    # Copy of config can be modified here. 
    agents.append(Agent(agent_config))
    f = open(folder + 'agent' + str(i) + '_config.txt','w')
    f.write(str(vars(agent_config)))
    f.close()

# Start training run
for i in range(FLAGS.episodes):
    if i%FLAGS.save_steps == 0:
        for agent in agents:
            agent.saveModel(folder, i)
       
    # Run episode
    #observations = np.zeros(2, )
    observations, reduced_observations = env.reset() # Get first observations
    observations = np.array(observations)
    
    for j in range(FLAGS.steps):

	# Load action for each agent
        actions = []
        for agent, observation in zip(agents, observations):
            if FLAGS.render: env.render()
            actions.append(agent.move(observation))
            agent.opt()
        print(actions)
        # Feedback:
        observations, rewards, t, equipment, reduced_observations = env.step(actions)
        
        # Check if last step has been reached
        if j == FLAGS.steps-1:
            t = True
            rewards = [-1,-1] 
        for agent, o, r, e, ro in zip(agents, observations, rewards, equipment, reduced_observations):
            if FLAGS.madrl is 'leniency':
                agent.feedback(r, t, o, reduced_observation=ro)
            elif FLAGS.madrl is 'nui':
                agent.feedback(r, t, o, meta_action=e)
            else:
                agent.feedback(r, t, o)

        if t: break # If t then terminal state has been reached
    
    with open(statscsv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
        writer.writerow(env.stats())
    print(env.stats())




'''
def step(actions):
'''