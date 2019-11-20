import bully_q.coop_game as coop_game
import bully_q.gs_game as gs_game
import bully_q.grid_game as grid_game
import bully_q.ipd as ipd
import numpy as np
import bully_q.table_q as table_q
#import ucb_q
#import stable_q
import datetime
from bully_q.utils import *
import subprocess
#import double_tableq_analysis as ana


def run(num_step = 600000, lr0 = 0.01, lr1 = 0.01, gamma0 = 0.99, gamma1 = 0.999):

	log_dir = './log/'
	# env = gs_game.GeneralSumEnv()
	# env = coop_game.BigCoopEnv()
	env = ipd.IpdEnv()
	#env = grid_game.GridEnv()

	agent0 = table_q.TableQAgent(env.num_state, env.num_action, lr=0.05, gamma = gamma1, epsilon_decay = 1, epsilon = 0.05)
	#agent0 = stable_q.StableQAgent(env.num_state, env.num_action, lr=0.02, gamma = gamma0, epsilon_decay = 1, epsilon = 0.05)
	#agent1 = stable_q.StableQAgent(env.num_state, env.num_action, lr=0.05, gamma = gamma1, epsilon_decay = 1, epsilon = 0.05)
	agent1 = table_q.TableQAgent(env.num_state, env.num_action, lr=0.05, gamma = gamma1, epsilon_decay = 1, epsilon = 0.02)
	
	num_recent = 100
	print_interval = 1
	state = env.reset()
	cumulative_reward = np.array([0, 0])
	recent_average_reward = [[0 for i in range(num_recent)] for j in range(2)]
	#policy_count = [0 for i in range(env.num_action ** (env.num_state * 2))]
	action_count = [0 for i in range(env.num_action ** 2)]
	start_step = 0

	resume = False

	if resume == True:
		filename = 'tableq_vs_tableq_20190716_104100'
		path = log_dir + 'tableq_vs_tableq/' + filename + '.log'
		print_with_time('Reading ' + path + '...')
		proc = subprocess.Popen(['tail', '-n', '4', path], stdout=subprocess.PIPE)
		lines = proc.stdout.readlines()
		tmp = str(lines[0]).split(',')
		start_step = eval(tmp[1]) + 1
		state = encode_action(*eval(tmp[2] + ',' + tmp[3]))
		cumulative_reward[0] = start_step * eval(tmp[6][8:])
		cumulative_reward[1] = start_step * eval(tmp[7][:-2])
		f = open(path, 'a')
		q0 = eval(lines[1])
		q1 = eval(lines[2])
		agent0.set_q(q0)
		agent1.set_q(q1)
	else:
		dt = datetime.datetime.now()
		filename = 'tableq_vs_tableq' + dt.strftime("_%Y%m%d_%H%M%S")
		path = log_dir + 'tableq_vs_tableq/' + filename + '.log'
		f = open(path, 'w')
		path = log_dir + 'tableq_vs_tableq/' + filename + '.cfg'
		f2 = open(path, 'w')
		config = {}
		config['num_step'] = num_step
		#config['lr0'] = agent0.lr
		config['lr1'] = agent1.lr
		#config['gamma0'] = agent0.gamma
		config['gamma1'] = agent1.gamma
		config['payoff'] = env.payoff
		f2.write(str(config))
		f2.close()
		#q0 = [[300, 104], [99, 100], [99, 100], [99, 100]]
		#q1 = [[300, 104], [99, 100], [99, 100], [99, 100]]
		#agent0.set_q(q0)
		#agent1.set_q(q1)

	print_with_time('Start training ' + str(num_step) + ' steps...')
	f_phase = open(log_dir + 'tableq_vs_tableq/phase' + dt.strftime("_%Y%m%d_%H%M%S") + '.log', 'w')
	for i in range(num_step):		
#		if state == 62 and agent1.episode[62] == 0:
#			f_phase.write(str((i, agent1.phase[62], agent1.explore_action[62], agent1.recover_action[62])) + '\n')
		action0 = agent0.act(state)
		#agent0_pi = [11. / 13., 0.5, 7. / 26., 0]
		#agent0_pi = [0.5, 1, 1, 1]

		#action0 = np.random.binomial(1, 1 - agent0_pi[state])
		action1 = agent1.act(state)
		joint_action = encode_action(action0, action1, env.num_action)
		action_count[joint_action] += 1
	
		#env.print()
		#direction = ['Up', 'Down', 'Left', 'Right']
		#print(direction[action0], direction[action1])
	
		[next_state, reward, done, info] = env.step(joint_action)
		
		#print('reward', reward)
		
		cumulative_reward[0] += reward[0]
		cumulative_reward[1] += reward[1]
		recent_average_reward[0][i % num_recent] = reward[0]
		recent_average_reward[1][i % num_recent] = reward[1]
		agent0.update(state, action0, reward[0], next_state, done)
		agent1.update(state, action1, reward[1], next_state, done)
		#policy0 = agent0.get_policy()
		policy1 = agent1.get_policy()
		#joint_policy = encode_policy(policy0, policy1, env.num_action)
		#policy_count[joint_policy] += 1
		# if i % print_interval == 0:
		# 	f.write(str((0, start_step + i, decode_action(state, env.num_action), decode_action(joint_action, env.num_action),
		# 		cumulative_reward / (i+1), np.mean(recent_average_reward[0]), np.mean(recent_average_reward[1]))) + '\n')
		# 	f.write(str(agent1.q) + '\n')
		# 	f.write(str(agent1.q) + '\n')
			#f.write(str((policy0, policy1)) + '\n')
			#f.write(agent0.output())
			#f.write(str((agent0.q[62], agent1.q[62])) + '\n')			

		if (i + 1) % (num_step // 10) == 0:
			print_with_time(str(i + 1) + '/' + str(num_step) + ' done.')
		if done:
			state = env.reset()
			#cumulative_reward[0] = 0
			#cumulative_reward[1] = 0
		else:
			state = next_state
			#state = env.reset()

	f_phase.close()
	f.close()
	print_with_time('Log saved in ' + filename + '.')

	# sort_policy_count = []
	# for i in range(len(policy_count)):
	# 	sort_policy_count.append((policy_count[i], i))
	# sort_policy_count.sort(reverse=True)
	# for i in range(len(policy_count)):
	# 	print_with_time(str((decode_policy(sort_policy_count[i][1], env.num_state, env.num_action), sort_policy_count[i][0] / num_step * 100)))
	for i in range(len(action_count)):
		print_with_time(str((decode_action(i, env.num_action), action_count[i] / num_step * 100)))

	#ana.analyze(filename, env.num_state, env.num_action)

if __name__ == "__main__":
	run()