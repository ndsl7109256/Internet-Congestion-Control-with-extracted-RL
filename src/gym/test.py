import gym
import network_sim
import numpy as np
import time

env = gym.make('PccNs-v0')

######Train teacher first
MAX_EPISODES = 10#200
MAX_EP_STEPS = 400#200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

var = 3  # control exploration
t1 = time.time()
for i in range(400):
    s = env.reset()
    ep_reward = 0
    a_ep = []
    a = np.array([0.87])
    s_, r, done, info = env.step(a)
    print(s_,r,done,info)
