import datetime
import matplotlib
matplotlib.use('Agg')

def encode_action(action1, action2, num_action = 2):
	return action1 * num_action + action2

def decode_action(action, num_action = 2):
	return [action // num_action, action % num_action]

def encode_policy(a_policy, b_policy, num_action = 2):
	result = 0
	for i in range(len(a_policy) - 1, -1, -1):
		result *= num_action
		result += a_policy[i]
	for i in range(len(b_policy) - 1, -1, -1):
		result *= num_action
		result += b_policy[i]
	return result

def decode_policy(policy, num_state = 4, num_action = 2):
	a_policy = []
	b_policy = []
	for i in range(num_state):
		b_policy.append(policy % num_action)
		policy = policy // num_action
	for i in range(num_state):
		a_policy.append(policy % num_action)
		policy = policy // num_action
	return a_policy, b_policy

def encode_single_policy(policy, num_action = 2):
	result = 0
	for i in range(3, -1, -1):
		result *= num_action
		result += policy[i]
	return result

def decode_single_policy(policy, num_state = 4, num_action = 2):
	result = []
	for i in range(num_state):
		result.append(policy % num_action)
		policy = policy // num_action
	return result

def encode_state_set(state_set):
	states = 0
	for state in state_set:
		states += 2**state
	return states

def decode_state_set(states):
	state_set = []
	for state in range(4):
		state_encode = 2**state
		if (states // state_encode) % 2 == 1:
			state_set.append(state)
	return state_set

def print_with_time(content):
	print(content + '\t' + datetime.datetime.now().isoformat())