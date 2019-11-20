from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import bully_q.ipd as ipd
import copy
from bully_q.utils import *


def calc_diff_dim(policy1, policy2):
	cnt = 0
	for i in range(4):
		if policy1[i] != policy2[i]:
			cnt += 1
	return cnt

a_policy = [0 for i in range(4)]
b_policy = [0 for i in range(4)]

env = ipd.IpdEnv()
joint_policy_value_a = [[0 for j in range(4)] for i in range(256)]
joint_policy_value_b = [[0 for j in range(4)] for i in range(256)]
# given joint policy, a state maps to a cycle. It is either in the cycle or will transit to the cycle.
state_cycle_map = [[-1 for j in range(4)] for i in range(256)]		
# given joint policy, there will be several cycles.
cycles = [[] for i in range(256)]

# compute the value of each joint policy
for joint_policy in range(256):
	a_policy, b_policy = decode_policy(joint_policy)
	for init_state in range(4):
		visited_state = []
		state = init_state
		while True: # skip the non-cycle states
			visited_state.append(state)
			a_action = a_policy[state]
			b_action = b_policy[state]
			joint_action = encode_action(a_action, b_action)
			[next_state, reward, done, info] = env.step(joint_action)
			state = next_state
			if state in visited_state:
				break
		visited_state = []
		while True: # compute average reward over one cycle
			visited_state.append(state)
			a_action = a_policy[state]
			b_action = b_policy[state]
			joint_action = encode_action(a_action, b_action)
			[next_state, reward, done, info] = env.step(joint_action)
			joint_policy_value_a[joint_policy][init_state] += reward[0]
			joint_policy_value_b[joint_policy][init_state] += reward[1]
			state = next_state
			if state in visited_state:
				break
		joint_policy_value_a[joint_policy][init_state] /= len(visited_state)
		joint_policy_value_b[joint_policy][init_state] /= len(visited_state)
		cycle = encode_state_set(visited_state)
		state_cycle_map[joint_policy][init_state] = cycle
		if cycle not in cycles[joint_policy]:
			cycles[joint_policy].append(encode_state_set(visited_state))

# compute the graph of joint policy											
switch_map = []
stable = [[True for i in range(4)] for j in range(256)]
max_diff = 4
for joint_policy in range(256):
	a_policy, b_policy = decode_policy(joint_policy)
	for init_state in range(4):
		for alt_policy in range(16):
			if calc_diff_dim(decode_single_policy(alt_policy), a_policy) <= max_diff:
				best_response_b_policy = 0
				best_response_b_value = -np.inf
				for b_alt_policy in range(16):
					if calc_diff_dim(decode_single_policy(alt_policy), b_policy) <= max_diff:
						tmp_joint_policy = encode_policy(decode_single_policy(alt_policy), decode_single_policy(b_alt_policy))
						if joint_policy_value_b[tmp_joint_policy][init_state] > best_response_b_value:
							best_response_b_value = joint_policy_value_b[tmp_joint_policy][init_state]
							best_response_b_policy = b_alt_policy
				alt_joint_policy = encode_policy(decode_single_policy(alt_policy), decode_single_policy(best_response_b_policy))
				if joint_policy_value_a[joint_policy][init_state] < joint_policy_value_a[alt_joint_policy][init_state]:
					switch_map.append([joint_policy, alt_joint_policy, init_state])
					stable[joint_policy][init_state] = False
			if calc_diff_dim(decode_single_policy(alt_policy), b_policy) <= max_diff:
				alt_joint_policy = encode_policy(a_policy, decode_single_policy(alt_policy))
				if joint_policy_value_b[joint_policy][init_state] < joint_policy_value_b[alt_joint_policy][init_state]:
					switch_map.append([joint_policy, alt_joint_policy, init_state])
					stable[joint_policy][init_state] = False

# create graph structure of joint policy, joint policies are linked regardless of the state
jp_graph = nx.DiGraph()
jp_graph.add_nodes_from([i for i in range(256)])
for [source_jp, target_jp, state] in switch_map:
	jp_graph.add_edge(source_jp, target_jp)

# compute the graph of the combination of joint policy and cycle
switch_map_with_cycle = [{} for i in range(256)]
for i in range(len(switch_map)):
	[source_jp, target_jp, state] = switch_map[i]
	# if state is not in a cycle under the source joint policy, we just ignore.
	if state in decode_state_set(state_cycle_map[source_jp][state]): 
		source_cycle = state_cycle_map[source_jp][state]
		target_cycle = state_cycle_map[target_jp][state]
		if source_cycle not in switch_map_with_cycle[source_jp]:
			switch_map_with_cycle[source_jp][source_cycle] = []
		switch_map_with_cycle[source_jp][source_cycle].append([target_jp, target_cycle])

# compute the stable joint policies w.r.t. all states
stable_wrt_all_states = {}
for i in range(256):
	valid = True
	values = None
	for j in range(4):
		if not stable[i][j]:
			valid = False
			break
		if values == None:
			values = (joint_policy_value_a[i], joint_policy_value_b[i])
		elif values != (joint_policy_value_a[i], joint_policy_value_b[i]):
			valie = False
			break
	if valid:
		stable_wrt_all_states[i] = values


# compute target count
target_count = {}
for source_jp in range(256):
	for source_cycle in switch_map_with_cycle[source_jp]:
		for [target_jp, target_cycle] in switch_map_with_cycle[source_jp][source_cycle]:
			if (target_jp, target_cycle) not in target_count:
				target_count[(target_jp, target_cycle)] = 0
			target_count[(target_jp, target_cycle)] += 1

# compute distribution of random walk 
random_walk_count = {}
num_episode = 100000
num_step = 50
for episode in range(num_episode):
	joint_policy = np.random.randint(256)
	state = np.random.randint(4)
	cycle = state_cycle_map[joint_policy][state]
	for step in range(num_step):
		if cycle in switch_map_with_cycle[joint_policy]:
			candidates = switch_map_with_cycle[joint_policy][cycle]
			(joint_policy, cycle) = candidates[np.random.randint(len(candidates))]
		else:
			break
	if (joint_policy, cycle) not in random_walk_count:
		random_walk_count[(joint_policy, cycle)] = 0
	random_walk_count[(joint_policy, cycle)] += 1
random_walk_count_sorted = []
for (jp, cycle) in random_walk_count:
	random_walk_count_sorted.append((random_walk_count[(jp, cycle)] / num_episode * 100, jp, cycle))
random_walk_count_sorted.sort(reverse=True)



print('joint policy value:')
f = open('./log/ipd_analysis/joint_policy_average_reward.txt', 'w')
for i in range(256):
	for j in range(4):
		print(decode_policy(i), decode_action(j), joint_policy_value_a[i][j], joint_policy_value_b[i][j])
		f.write('\t'.join([str(decode_policy(i)), str(decode_action(j)), str(joint_policy_value_a[i][j]), str(joint_policy_value_b[i][j])]) + '\n')
f.close()

print('cycles:')
for i in range(256):
	for cycle in cycles[i]:
		print(decode_policy(i), decode_state_set(cycle))

print('switch_map:')
for i in range(len(switch_map)):
	state = switch_map[i][2]
	print(decode_policy(switch_map[i][0]), joint_policy_value_a[switch_map[i][0]][state], joint_policy_value_b[switch_map[i][0]][state],
		'->', 
		decode_policy(switch_map[i][1]), joint_policy_value_a[switch_map[i][1]][state], joint_policy_value_b[switch_map[i][1]][state],
		' in ', 
		decode_action(state))

print('switch_map_with_cycle:')
f = open('./log/ipd_analysis/joint_policy_switch_graph.txt', 'w')
for source_jp in range(256):
	for source_cycle in switch_map_with_cycle[source_jp]:
		for [target_jp, target_cycle] in switch_map_with_cycle[source_jp][source_cycle]:
			if target_cycle in switch_map_with_cycle[target_jp]:
				if [source_jp, source_cycle] in switch_map_with_cycle[target_jp][target_cycle]:
					bi_direction = 'bi-direction'
				else:
					bi_direction = ''
			print(decode_policy(source_jp), decode_state_set(source_cycle), '->', decode_policy(target_jp), decode_state_set(target_cycle), bi_direction)
			f.write(str(decode_policy(source_jp)) + '\t' + str(decode_state_set(source_cycle)) +  '->' + str(decode_policy(target_jp)) + '\t' + str(decode_state_set(target_cycle)) + '\n')
f.close()

print('stable:')
for i in range(256):
	for j in range(4):
		if stable[i][j]:
			print(decode_policy(i), decode_action(j), joint_policy_value_a[i][j], joint_policy_value_b[i][j])

print('stable cycle:')
for source_jp in range(256):
	for cycle in cycles[source_jp]:
		if cycle not in switch_map_with_cycle[source_jp]:
			print(decode_policy(source_jp), decode_state_set(cycle))

print('stable joint policy w.r.t. all states:')
for i in stable_wrt_all_states:
	print(decode_policy(i), stable_wrt_all_states[i])

print('target count:')
target_count_list = []
for (target_jp, target_cycle) in target_count:
	target_count_list.append((target_count[(target_jp, target_cycle)], target_jp, target_cycle))
target_count_list.sort(reverse=True)
for (count, jp, cycle) in target_count_list:
	print(decode_policy(jp), decode_state_set(cycle), count)

print('random walk:')
for (count, jp, cycle) in random_walk_count_sorted:
	print(decode_policy(jp), decode_state_set(cycle), count)

# print('cycles in jp graph:')
# print(list(nx.simple_cycles(jp_graph)))

# filename = './log/ipd_analysis/jp_graph_circular_layout.pdf'
# print('joint policy graph saved in ' + filename)
# pos = nx.layout.circular_layout(jp_graph, scale=10)
# nodes = nx.draw_networkx_nodes(jp_graph, pos, node_size=1, node_color='blue')
# edges = nx.draw_networkx_edges(jp_graph, pos, node_size=1, arrowstyle='->',
#                                arrowsize=1, edge_color='grey',
#                                edge_cmap=plt.cm.Blues, width=0.01)
# ax = plt.gca()
# ax.set_axis_off()
# plt.savefig(filename)
