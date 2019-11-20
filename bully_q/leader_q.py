import numpy as np

class LeaderQAgent():

	def __init__(self, num_state, num_action_leader, num_action_follower, gamma = 0.99, epsilon = 0.005, lr = 0.001, epsilon_decay = 0.999999):
		self.num_state = num_state
		self.num_action_leader = num_action_leader
		self.num_action_follower = num_action_follower
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr
		self.epsilon_decay = epsilon_decay
		self.reset()

	def reset(self):
		#self.q = [[(np.random.random() - 0.5) * 100 for j in range(self.num_action)] for i in range(self.num_state)]
		self.q = [[[0 for j in range(self.num_action_follower)] for k in range(self.num_action_leader)] for i in range(self.num_state)]
		#self.q = [[1, 0], [99, 100], [0, 1], [99, 100]]
		#self.q = [[20, 10]]

	def act(self, state, q_follower, debug = False):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_action_follower)
			# print('Random action ' + str(action))
			return action
		else:
			max_q_action_leader = max_q_action_follower = -1

			for action_leader in range(self.num_action_leader):
				cur_max_q_action_follower = 0
				for action_follower in range(self.num_action_follower):
					if q_follower[state][action_leader][action_follower] > q_follower[state][action_leader][
						cur_max_q_action_follower]:
						cur_max_q_action_follower = action_follower
				if max_q_action_leader < 0 or self.q[state][action_leader][cur_max_q_action_follower] > self.q[state][max_q_action_leader][max_q_action_follower]:
					max_q_action_leader, max_q_action_follower = action_leader, cur_max_q_action_follower
					if debug:
						print("update::-----------", max_q_action_leader, max_q_action_follower)
			return max_q_action_leader

	def update(self, state, action_leader, action_follower, reward, next_state, q_follower, done=False):
		max_q_action = max_q_action_follower = -1
		for next_action in range(self.num_action_leader):
			cur_max_q_action = 0
			for next_action_follower in range (self.num_action_follower):
				if q_follower[next_state][next_action][next_action_follower] > q_follower[next_state][max_q_action][cur_max_q_action]:
					cur_max_q_action = next_action_follower
			if max_q_action < 0 or self.q[next_state][next_action][cur_max_q_action] > self.q[next_state][max_q_action][max_q_action_follower]:
				(max_q_action, max_q_action_follower) = (next_action, cur_max_q_action)
		if done:
			diff_q = reward - self.q[state][action_leader][action_follower]
		else:
			diff_q = reward + self.gamma * self.q[next_state][max_q_action][max_q_action_follower] - self.q[state][action_leader][action_follower]
		self.q[state][action_leader][action_follower] = self.q[state][action_leader][action_follower] + self.lr * diff_q
		self.epsilon *= self.epsilon_decay

	def print_q(self):
		print(self.q)

	def print_q_state(self, state):
		print(self.q[state])

	def get_policy(self):
		policy = []
		for state in range(self.num_state):
			for action_leader in range(self.num_action_leader):
				max_action = 0
				for action in range(self.num_action_follower):
					if self.q[state][action] > self.q[state][max_action]:
						max_action = action
				policy.append(max_action)
		return policy

	def set_q(self, q):
		self.q = q
