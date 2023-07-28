import numpy as np

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action): 
        self.current_step += 1
        next_state = self.data[self.current_step]
        done = self.current_step == len(self.data) - 1

        reward = 0

        # calculate the reward
        if action == 1: # Buy
            self.current_position = self.data[self.current_step] # store the buying price
        elif action == 2 and self.current_position is not None: # Sell
            reward = self.data[self.current_step] - self.current_position # selling price - buying price
            self.current_position = None # Reset the buying price
        else:
            reward = 0

        return next_state, reward, done

    def get_state_dim(self):
        return self.data.shape[1]

    def get_action_dim(self):
        return 2  # 例: 買う or 売る
