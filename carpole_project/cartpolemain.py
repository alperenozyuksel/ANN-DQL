import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        # Hyperparameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0  # Explore
        self.epsilon_decay = 0.995  # Decrease exploration over time
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        # Neural network for deep Q-learning
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Decide whether to explore or exploit
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Train the model using experiences in memory
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGready(self):
        # Decrease epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    # Initialize environment and agent
    env = gym.make('CartPole-v1', render_mode='human')  # 'render_mode' parametresi eklendi
    agent = DQLAgent(env)
    episodes = 1000

    for e in range(episodes):
        # Initialize environment and get initial state
        state, _ = env.reset()  # Adjusted to correctly unpack the tuple
        state = np.reshape(state, [1, agent.state_size])
        time = 0
        batch_size = 16

        while True:
            # Render environment for visualization
            env.render()

            # Agent takes action
            action = agent.act(state)

            # Environment processes action and returns next state and reward
            next_state, reward, done, truncated, _ = env.step(action)  # Unpack all 5 values
            next_state = np.reshape(next_state, [1, agent.state_size])

            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Train the agent with replay
            agent.replay(batch_size)

            # Decrease exploration rate
            agent.adaptiveEGready()

            time += 1
            if done or truncated:  # Check if done or truncated
                print(f"Episode: {e+1}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2f}")
                break

    # Close the environment when done
    env.close()
