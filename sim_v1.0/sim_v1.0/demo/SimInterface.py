# -*- coding: utf-8 -*-

from kernel import kernel
import numpy as np


class SimInterface(object):

    def __init__(self, agent_num, time, render=True):
        self.game = kernel(car_num=agent_num, time=time, render=render)
        self.g_map = self.game.get_map()
        self.memory = []
        sum

    def reset(self):
        self.state = self.game.reset()
        # state, object
        self.obs = self.get_observation(self.state)
        return self.obs

    def step(self, actions):
        state = self.game.step(actions)
        obs = self.get_observation(state)
        rewards = self.get_reward(state)

        self.memory.append([self.obs, actions, rewards])
        self.state = state

        return obs, rewards, state.done, None
        # return state

    def get_observation(self, state):
        # personalize your observation here
        obsagent = state.agents
        obs =np.array(obsagent).flatten()
        return obs

    def get_reward(self, state):
        EnemyAlive = 0
        WeAlive = 0
        if not state.done:
            rewards = 0
        if state.done:
            for i in range(5):
                if state.agents[i+5][6]!=0:
                    EnemyAlive=1
                if state.agents[i][6]!=0:
                    WeAlive=1
            if WeAlive==1 and EnemyAlive==0:
                rewards=1
            else:
                rewards=-1
                print(rewards)
        return rewards

    def play(self, cars_info, cars_guide, endFlag):
        return self.game.play(cars_info, cars_guide, endFlag)
