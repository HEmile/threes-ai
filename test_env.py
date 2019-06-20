import gym
import random

env = gym.make('gym_threes:threes-v0')
env.reset()
for _ in range(1000):
    action = random.randint(0, 3)
    print(action, env.render())
    env.step(action) # take a random action
env.close()