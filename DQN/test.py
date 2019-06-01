import random 
import gym 
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

y = 0.95
l_rate = 0.001

exp_memory_size = 1000000
batch_size = 20

exploration_max = 1.0
exploration_min = 0.01
exploration_decay = 0.9995

class DQNSolver():
    def __init__(self, observation_space, action_space):
        self.exploration_rate = exploration_max
        
        self.action_space = action_space
        self.memory = deque(maxlen = exp_memory_size)
        
        self.model = Sequential()
        self.model.add(Dense(24, input_shape = [observation_space, ], activation = 'relu'))
        self.model.add(Dense(24, activation = 'relu'))
        self.model.add(Dense(self.action_space, activation = 'linear'))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = l_rate))

    def act(self, state):
        #if np.random.rand() < self.exploration_rate:
        #    return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

def test():
    env = gym.make('CartPole-v1')
    env.reset()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        dqn_solver.model.load_weights('../weights/dqn_weights_%i.h5' % (run))
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        total_reward = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            env.render()            
            reward = reward if not terminal else -reward
            total_reward += reward
            state_next = np.reshape(state_next, [1, observation_space])
            state = state_next
            if terminal:
                print('episode : ' + str(run) + ' reward : ' + str(total_reward))
                break

if __name__ == "__main__":
    test()

