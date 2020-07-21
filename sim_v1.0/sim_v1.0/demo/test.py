import numpy as np
from SimInterface import SimInterface
env = SimInterface(agent_num=10, time=5, render=True)
env.reset()
for i_episode in range(400):
    env.reset()
    test_orders = np.zeros((10, 8), dtype='int8')
    while True:
        s_, r, done, info = env.step(test_orders)
        if done == True:
            break