import bully_q.coop_game as coop_game
import bully_q.gs_game as gs_game
import bully_q.grid_game as grid_game
import bully_q.complex_grid as complex_grid

import bully_q.ipd as ipd
import numpy as np
import bully_q.leader_q as leader_q
import bully_q.follower_q as follower_q
# import ucb_q
# import stable_q
import datetime
from bully_q.utils import *
import subprocess
import matplotlib.pyplot as plt


# import double_tableq_analysis as ana


def printGraph(X, Y, name):

	plt.figure('Draw')
	# X = np.append(X, X[0])
	# Y = np.append(Y, Y[0])
	plt.plot(X, Y)
	# plt.xlim(-1, 1)
	# plt.ylim(-1, 1)
	plt.scatter(X, Y, color = 'r', marker='.')
	plt.draw()
	plt.savefig("graph/" + name + ".png")
	plt.close()

def run(num_step=50000000, lr0=0.01, lr1=0.01, gamma0=0.99, gamma1=0.99):
	log_dir = './log/'
	# env = gs_game.GeneralSumEnv()
	# env = coop_game.BigCoopEnv()
	# env = complex_grid.ComplexGridEnv()
	env = ipd.IpdEnv()
	# env = grid_game.GridEnv()

	agent0 = leader_q.LeaderQAgent(env.num_state, env.num_action, env.num_action, lr=0.1, gamma=gamma0, epsilon_decay=1., epsilon=0.005)
	# agent0 = stable_q.StableQAgent(env.num_state, env.num_action, lr=0.02, gamma = gamma0, epsilon_decay = 1, epsilon = 0.05)
	# agent1 = stable_q.StableQAgent(env.num_state, env.num_action, lr=0.05, gamma = gamma1, epsilon_decay = 1, epsilon = 0.05)
	agent1 = follower_q.FollowerQAgent(env.num_state, env.num_action, env.num_action, lr=0.1, gamma=gamma1, epsilon_decay=1., epsilon=0.005)

	num_recent = 100
	print_interval = 100
	state = env.reset()
	cumulative_reward = np.array([0, 0])
	recent_average_reward = [[0 for i in range(num_recent)] for j in range(2)]
	# policy_count = [0 for i in range(env.num_action ** (env.num_state * 2))]
	action_count = [0 for i in range(env.num_action ** 2)]
	start_step = 0

	resume = False

	if resume == True:
		filename = 'bilevelq_vs_tableq_20190718_115153'
		path = log_dir + 'bilevelq_vs_tableq/' + filename + '.log'
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
		filename = 'bilevelq_vs_tableq' + dt.strftime("_%Y%m%d_%H%M%S")
		path = log_dir + 'bilevelq_vs_tableq/' + filename + '.log'
		f = open(path, 'w')
		path = log_dir + 'bilevelq_vs_tableq/' + filename + '.cfg'
		f2 = open(path, 'w')
		config = {}
		config['num_step'] = num_step
		# config['lr0'] = agent0.lr
		config['lr1'] = agent1.lr
		# config['gamma0'] = agent0.gamma
		config['gamma1'] = agent1.gamma
		config['payoff'] = env.payoff
		f2.write(str(config))
		f2.close()
	# q0 = [[300, 104], [99, 100], [99, 100], [99, 100]]
	# q1 = [[300, 104], [99, 100], [99, 100], [99, 100]]
	# agent0.set_q(q0)
	# agent1.set_q(q1)

	init_state = state
	cnt_step = 0
	cnt_done = 0

	done_print_interal = 10

	prt = [[[] for i in range(env.num_action ** 2)] for j in range(env.num_state)]
	steps_x = [[] for i in range(env.num_state)]

	print_with_time('Start training ' + str(num_step) + ' steps...')
	# f_phase = open(log_dir + 'bilevelq_vs_tableq/phase' + dt.strftime("_%Y%m%d_%H%M%S") + '.log', 'w')
	for i in range(num_step):
		#if state == 62 and agent1.episode[62] == 0:
		#	f_phase.write(str((i, agent1.phase[62], agent1.explore_action[62], agent1.recover_action[62])) + '\n')
		action0 = agent0.act(state, agent1.get_q())
		# agent0_pi = [11. / 13., 0.5, 7. / 26., 0]
		#agent0_pi = [0.5, 1, 1, 1]

		#action0 = np.random.binomial(1, 1 - agent0_pi[state])
		action1 = agent1.act(state, action0)
		joint_action = encode_action(action0, action1, env.num_action)
		if state == init_state:
			action_count[joint_action] += 1
			cnt_step += 1

		# print(state)
		if i % print_interval == 0:
			steps_x[state].append(i)
			for x in range(env.num_action):
				for y in range(env.num_action):
					prt[state][encode_action(x, y, env.num_action)].append(agent0.q[state][x][y])

		# env.print()
		# direction = ['Up', 'Down', 'Left', 'Right']
		# print(direction[action0], direction[action1])

		[next_state, reward, done, info] = env.step(joint_action)

		# print('reward', reward)

		cumulative_reward[0] += reward[0]
		cumulative_reward[1] += reward[1]
		recent_average_reward[0][i % num_recent] = reward[0]
		recent_average_reward[1][i % num_recent] = reward[1]
		agent0.update(state, action0, action1, reward[0], next_state, agent1.get_q(), done)
		agent1.update(state, action0, action1, reward[1], next_state, agent0.act(next_state, agent1.get_q()), done)
		# policy0 = agent0.get_policy()
		# policy1 = agent1.get_policy()
		# joint_policy = encode_policy(policy0, policy1, env.num_action)
		# policy_count[joint_policy] += 1
		if i >= 10000000 and i <= 10000100:
			f.write(str(
				(0, start_step + i, decode_action(state, env.num_action), decode_action(joint_action, env.num_action),
				 cumulative_reward / (i + 1), np.mean(recent_average_reward[0]),
				 np.mean(recent_average_reward[1]))) + '\n')
			f.write(str(agent0.q) + '\n')
			f.write(str(agent1.q) + '\n')
		# f.write(str((policy0, policy1)) + '\n')
		# f.write(agent0.output())
		# f.write(str((agent0.q[62], agent1.q[62])) + '\n')

		if (i + 1) % (num_step // 10) == 0:
			print_with_time(str(i + 1) + '/' + str(num_step) + ' done.')
		if done:
			state = env.reset()
			cnt_done = cnt_done + 1
			if cnt_done % done_print_interal == 0:
				print(cumulative_reward)
			cumulative_reward[0] = 0
			cumulative_reward[1] = 0
		else:
			state = next_state
	# state = env.reset()

	# f_phase.close()
	f.close()
	print_with_time('Log saved in ' + filename + '.')

	# sort_policy_count = []
	# for i in range(len(policy_count)):
	# 	sort_policy_count.append((policy_count[i], i))
	# sort_policy_count.sort(reverse=True)
	# for i in range(len(policy_count)):
	# 	print_with_time(str((decode_policy(sort_policy_count[i][1], env.num_state, env.num_action), sort_policy_count[i][0] / num_step * 100)))
	# print(cnt_step)
	for cur_state in range(env.num_state):
		for i in range(len(action_count)):
			print_with_time(str((decode_action(i, env.num_action), action_count[i] / cnt_step * 100)))
			printGraph(steps_x[cur_state], prt[cur_state][i], str(cur_state) + ',act=' + str(decode_action(i, env.num_action)))
	# agent0.print_q()
	# agent0.print_q_state(init_state)
	# agent1.print_q_state(init_state)
	# act0 = agent0.act(init_state, agent1.get_q(), True)
	# print(act0, agent0.q[init_state][act0])

if __name__ == "__main__":
	run()