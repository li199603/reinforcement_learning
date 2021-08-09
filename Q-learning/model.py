import numpy as np

class Q_learning():
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon):
        self.Q_table = np.zeros([state_dim, action_dim])
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            Q_row = self.Q_table[state, :]
            max_q = np.max(Q_row)
            tmp = np.where(Q_row == max_q)
            action = np.random.choice(tmp[0])
        else:
            action = np.random.choice(self.action_dim)
        return action

    def learn(self, s_pre, a, r, s_cur, done):
        Q_predict = self.Q_table[s_pre, a]
        if done:
            Q_target = r
        else:
            Q_target = r + self.gamma * np.max(self.Q_table[s_cur, :])
        self.Q_table[s_pre, a] += self.lr * (Q_target - Q_predict)

    def save(self, path):
        np.save(path, self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path)