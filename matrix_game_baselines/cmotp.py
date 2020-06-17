from environment import Environment
from stats import mkRunDir
from config import Config
from copy import deepcopy
import tensorflow as tf
from agent import Agent
import csv

flags = tf.app.flags
# Environments:
# CMOTP:
# -CMOTP_V{1,2,3}
flags.DEFINE_string('environment', 'CMOTP_V1', 'Environment.') 
flags.DEFINE_string('madrl', 'hysteretic', 'MA-DRL extension. Options: None, leniency, hysteretic')
flags.DEFINE_string('format', 'NHWC', 'Format for conv (NHWC or NCHW)')
flags.DEFINE_string('processor', '/gpu:0', 'GPU/CPU.')
flags.DEFINE_integer('episodes', 5000, 'Number of episodes.')
flags.DEFINE_integer('steps', 10000, 'Max steps per episdoe.')
flags.DEFINE_integer('agents', 2, 'Number of agents.')
flags.DEFINE_boolean('render', False, 'Render environment.')
FLAGS = flags.FLAGS

# Environment is instantiated
# The dimension of the obsrvations and the number 
# of descrete actions can be accessed as follows:
# 
env = Environment(FLAGS) 

# Example:
config = Config(env.dim, env.out, madrl=FLAGS.madrl, gpu=FLAGS.processor, format=FLAGS.format)

# Run dir and stats csv file are created
statscsv, folder = mkRunDir(env, FLAGS)

# Agents are instantiated
agents = []
for i in range(FLAGS.agents): 
    agent_config = deepcopy(config)
    agents.append(Agent(agent_config)) # Init agent instances
    f = open(folder + 'agent' + str(i) + '_config.txt','w')
    f.write(str(vars(agent_config)))
    f.close()

# Start training run
for i in range(FLAGS.episodes):
    # Run episode
    observations = env.reset() # Get first observations
    for j in range(FLAGS.steps):

        # Renders environment if flag is true
        if FLAGS.render: env.render() 

        # Load action for each agent based on o^i_t
        actions = [] 
        for agent, observation in zip(agents, observations):
            actions.append(agent.move(observation)) 

            # Optimise agent
            agent.opt() 
        
        # Execute actions and get feedback lists:
        observations, rewards, t = env.step(actions)

        # Check if last step has been reached
        if j == FLAGS.steps-1:
            t = True
    
        for agent, o, r in zip(agents, observations, rewards):
            # Pass o^i_{t+1}, r^i_{t+1} to each agent i
            agent.feedback(r, t, o) 

        if t: break # If t then terminal state has been reached

    # Add row to stats: 
    with open(statscsv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
        writer.writerow(env.stats())
    print(env.stats())


