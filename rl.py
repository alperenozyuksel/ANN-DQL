import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Ortamı oluştur ve render_mode parametresini belirt
env = gym.make("Taxi-v3", render_mode="ansi")

# Q Table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting metrix
reward_list = [] # toplam ödüller
dropout_list = [] # ceza

episode_number = 1000

for i in range(1, episode_number):

    # Initialize environment
    state, _ = env.reset()

    reward_count = 0
    dropout = 0
    while True:

        # Exploit and explore to find action
        # %10 explore %90 exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Action process and take reward/take observation
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q learning function
        old_value = q_table[state, action]   # olası durumlar ve hareketler
        next_max = np.max(q_table[next_state]) # en yüksek puanın seçimi

        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Q LEARNING DENKLEMI

        # Q Table update
        q_table[state, action] = next_value

        # Update state
        state = next_state

        # Find wrong dropouts
        if reward == -10:
            dropout += 1

        reward_count += reward

        if done:
            break

    dropout_list.append(dropout)
    reward_list.append(reward_count)
    print("Episode: {}, reward {}, wrong dropout {}".format(i, reward_count, dropout))
