import random

import gym
import numpy as np


def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Do action and get result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Get corresponding q value from state, action pair
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])

            # Update Q value using Bellman equation
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # When episode is done, print reward
            if done:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t+1, total_reward))
                break

        # Exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay


if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    MAX_EPISODES = 5
    MAX_TRY = 5
    epsilon = 1.0
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_actions = env.action_space.n
    q_table = np.zeros((env.observation_space.n, num_actions))
    simulate()