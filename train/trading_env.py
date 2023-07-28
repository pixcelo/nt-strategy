import numpy as np

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        # この部分で報酬と次の状態を計算するロジックを実装
        self.current_step += 1
        next_state = self.data[self.current_step]
        reward = 0  # とりあえず0としますが、実際には適切な報酬を計算する必要があります
        done = self.current_step == len(self.data) - 1
        return next_state, reward, done

    def get_state_dim(self):
        return self.data.shape[1]

    def get_action_dim(self):
        return 2  # 例: 買う or 売る
