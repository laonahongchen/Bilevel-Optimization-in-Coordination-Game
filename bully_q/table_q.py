import numpy as np

class TableQAgent():

	def __init__(self, num_state, num_action, gamma = 0.99, epsilon = 0.005, lr = 0.001, epsilon_decay = 0.999999):
		self.num_state = num_state
		self.num_action = num_action
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr
		self.epsilon_decay = epsilon_decay
		self.reset()

	def reset(self):
		self.q = [[(np.random.random() - 0.5) * 100 for j in range(self.num_action)] for i in range(self.num_state)]
		# self.q = [[0 for j in range(self.num_action)] for i in range(self.num_state)]
		#self.q = [[1, 0], [99, 100], [0, 1], [99, 100]]
		#self.q = [[20, 10]]

	def act(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_action)
			#print('Random action ' + str(action))
			return action
		else:
			max_q_action = 0
			for action in range(self.num_action):
				if self.q[state][action] > self.q[state][max_q_action]:
					max_q_action = action
			return max_q_action

	def update(self, state, action, reward, next_state, done = False):
		max_q_action = 0
		for next_action in range(self.num_action):
			if self.q[next_state][next_action] > self.q[next_state][max_q_action]:
				max_q_action = next_action
		if done:
			diff_q = reward - self.q[state][action]
		else:
			diff_q = reward + self.gamma * self.q[next_state][max_q_action] - self.q[state][action]
		self.q[state][action] = self.q[state][action] + self.lr * diff_q
		self.epsilon *= self.epsilon_decay

	def print_q(self):
		print(self.q)

	def get_policy(self):
		policy = []
		for state in range(self.num_state):
			max_action = 0
			for action in range(self.num_action):
				if self.q[state][action] > self.q[state][max_action]:
					max_action = action
			policy.append(max_action)
		return policy

	def set_q(self, q):
		self.q = q