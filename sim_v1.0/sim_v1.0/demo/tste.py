import numpy as np


from SimInterface import SimInterface
game = SimInterface(agent_num=10, time=5, render=True)
game.reset()
test_orders = np.zeros((10,8), dtype='int8')
num_step = 0
while True:
    num_step= num_step+1
    step_info = game.step(test_orders)
    print('step:', num_step, '-', step_info)




