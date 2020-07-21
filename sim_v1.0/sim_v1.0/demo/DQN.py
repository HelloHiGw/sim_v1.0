import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from SimInterface import SimInterface
from collections import namedtuple
import os
from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs/exp1')
# Hyper Parameters
BATCH_SIZE = 2
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100
SAVEMOLDE_ITER=1000# target update frequency
SAVE_POSITION=0
MEMORY_CAPACITY = 10
env = SimInterface(agent_num=10, time=5, render=True)
N_ACTIONS = 14
N_STATES = 165
ENV_A_SHAPE = 0
ActionHash3 = [-1,0,1]
ActionHash2 = [0,1]
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print("# loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("# loaded checkpoint '{}'".format(filename))
    else:
        print("# no checkpoint found at '{}'".format(filename))
    return model, optimizer


def save_checkpoint(filename, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def log(ep, r, res_queue, global_nr_steps, steps):
    with ep.get_lock():
        ep.value += 1
    with ep_r.get_lock():
        if ep_r.value == 0.:
            ep_r.value = -1  # ep_r
            steps.value = steps
        else:
            ep_r.value = r.value * 0.99 + ep_r * 0.01
            global_nr_steps.value = global_nr_steps.value * 0.99 + steps * 0.01
    res_queue.put(ep_r)
    print(
        name,
        "Ep:", ep.value,
        "| Avg Ep_r: %.2f" % ep_r.value,
        "| Avg Steps: %d" % steps.value,
        "| Ep_r / Steps: %.2f" % (ep_r / steps),
    )

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, t):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 900)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(900, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.saveModle_counter=0
        self.evalfilename = './evalModel'
        self.targetfilename = './targetModel'

    def choose_action(self, x):
        #action shape = [0,0,0,0,0,0,0,0]
        action = []
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x) # 20维
            actions_value=actions_value.data.numpy()
            a1 = ActionHash3[np.argmax(actions_value[:,0:3])]
            a2 = ActionHash3[np.argmax(actions_value[:, 3:6])]
            a3 = ActionHash3[np.argmax(actions_value[:,6:9])]
            a4 = ActionHash3[np.argmax(actions_value[:,9:12])]
            a5 = ActionHash2[np.argmax(actions_value[:,12:14])]
            action=[a1,a2,a3,a4,a5]
        else:   # random
            for i in range(4):
                in_action = np.random.randint(-1, 2)
                action.append(in_action)
            action.append(np.random.randint(0, 2))

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.saveModle_counter % SAVEMOLDE_ITER == 0:
            print("-----------------------MODLE---SAVED-------------------------------")
            self.evalfilename+=str(self.saveModle_counter/SAVEMOLDE_ITER)
            self.evalfilename +='.pth'
            self.targetfilename += str(self.saveModle_counter/SAVEMOLDE_ITER)
            self.targetfilename += '.pth'
            torch.save(self.eval_net.state_dict(), self.evalfilename)
            torch.save(self.target_net.state_dict(), self.targetfilename)
        self.saveModle_counter += 1

        # sample batch transitions
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # b_memory = self.memory[sample_index, :]
        b_memory= Memreplay.sample(BATCH_SIZE)
        b_s = torch.tensor([t.state for t in b_memory]).float()
        b_a = torch.tensor([t.action for t in b_memory]).long()
        b_r = torch.tensor([t.reward for t in b_memory]).float()
        b_s_ = torch.tensor([t.next_state for t in b_memory]).float()
        # b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a).sum(dim=1) # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_nextmax=q_next[:,0:3].max(1)[0]+q_next[:,3:6].max(1)[0]+\
                  q_next[:,6:9].max(1)[0]+q_next[:,9:12].max(1)[0]+q_next[:,12:14].max(1)[0]
                  # q_next[:,14:16].max(1)[0]+q_next[:,16:18].max(1)[0]+q_next[:,18:20].max(1)[0]
        q_target = b_r + GAMMA * q_nextmax  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        writer.add_scalar('loss', loss, self.learn_step_counter)
        self.optimizer.step()


Memreplay = ReplayMemory(MEMORY_CAPACITY)


def train():
    dqn = DQN()
    wintime = 0
    losetime = 0
    filename = './A3C_TP_trained_critic.pth'
    print('\nCollecting experience...')
    for i_episode in range(40000):
        s = env.reset()
        ep_steps = 0
        ep_r=-1.0
        while True:
            agent1s = np.concatenate((s, s[0:15]))
            agent2s = np.concatenate((s, s[15:30]))
            agent3s = np.concatenate((s, s[30:45]))
            agent4s = np.concatenate((s, s[45:60]))
            agent5s = np.concatenate((s, s[60:75]))
            random_action1 = []
            random_action2 = []
            random_action3 = []
            random_action4 = []
            random_action5 = []
            for i in range(8):
                in_action1 = np.random.randint(-1, 2)
                random_action1.append(in_action1)
                in_action2 = np.random.randint(-1, 2)
                random_action2.append(in_action2)
                in_action3 = np.random.randint(-1, 2)
                random_action3.append(in_action3)
                in_action4 = np.random.randint(-1, 2)
                random_action4.append(in_action4)
                in_action5 = np.random.randint(-1, 2)
                random_action5.append(in_action5)
            DNQ_action1 = dqn.choose_action(agent1s)
            DNQ_action2 = dqn.choose_action(agent2s)
            DNQ_action3 = dqn.choose_action(agent3s)
            DNQ_action4 = dqn.choose_action(agent4s)
            DNQ_action5 = dqn.choose_action(agent5s)
            DNQ_action1full = DNQ_action1 + [0, 1, 0]
            DNQ_action2full = DNQ_action2 + [0, 1, 0]
            DNQ_action3full = DNQ_action3 + [0, 1, 0]
            DNQ_action4full = DNQ_action4 + [0, 1, 0]
            DNQ_action5full = DNQ_action5 + [0, 1, 0]
            offset = np.array([1, 4, 7, 10, 12])
            a = np.array([DNQ_action1full, DNQ_action2full, DNQ_action3full, DNQ_action4full, DNQ_action5full,
                          random_action1, random_action2, random_action3, random_action4, random_action5])
            # take action
            s_, r, done, info = env.step(a)
            ep_steps+=1
            print(ep_steps)
            agent1nexts = np.concatenate((s_, s_[0:15]))
            agent2nexts = np.concatenate((s_, s_[15:30]))
            agent3nexts = np.concatenate((s_, s_[30:45]))
            agent4nexts = np.concatenate((s_, s_[45:60]))
            agent5nexts = np.concatenate((s_, s_[60:75]))
            transition = Transition(agent1s, DNQ_action1 + offset, r, agent1nexts)

            transition = Transition(agent2s, DNQ_action2 + offset, r, agent1nexts)
            transition = Transition(agent3s, DNQ_action3 + offset, r, agent1nexts)
            transition = Transition(agent4s, DNQ_action4 + offset, r, agent1nexts)
            transition = Transition(agent5s, DNQ_action5 + offset, r, agent1nexts)
            # dqn.store_transition(s, a, r, s_)
            Memreplay.push(transition)
            dqn.memory_counter += 5

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    if r==1:
                        str='GameWin'
                        wintime+=1
                    else:
                        str='Gamelose'
                        losetime+=1
                    ep_r = ep_r* 0.99 + r * 0.01
                    print("Episode:",i_episode,str,"Avg_Loss:",ep_r)

            if done:
                print("Episode Finish")
                break
            s = s_


def eval(world, init_gmodel=False):
    dqn=DQN()
    env = SimInterface(agent_num=10, time=180, render=True)

    do_print = True
    done = None
    reward = 0
    last_reward = [0, 0, 0, 0]

    while True:
        done, s = False, env.reset()
        agent1s = np.concatenate((s, s[0:15]))
        agent2s = np.concatenate((s, s[15:30]))
        agent3s = np.concatenate((s, s[30:45]))
        agent4s = np.concatenate((s, s[45:60]))
        agent5s = np.concatenate((s, s[60:75]))
        t = 0
        while True:
            random_action1 = []
            random_action2 = []
            random_action3 = []
            random_action4 = []
            random_action5 = []
            for i in range(8):
                in_action1 = np.random.randint(-1, 2)
                random_action1.append(in_action1)
                in_action2 = np.random.randint(-1, 2)
                random_action2.append(in_action2)
                in_action3 = np.random.randint(-1, 2)
                random_action3.append(in_action3)
                in_action4 = np.random.randint(-1, 2)
                random_action4.append(in_action4)
                in_action5 = np.random.randint(-1, 2)
                random_action5.append(in_action5)
            DNQ_action1 = dqn.choose_action(agent1s)
            DNQ_action2 = dqn.choose_action(agent2s)
            DNQ_action3 = dqn.choose_action(agent3s)
            DNQ_action4 = dqn.choose_action(agent4s)
            DNQ_action5 = dqn.choose_action(agent5s)
            DNQ_action1full = DNQ_action1 + [0, 1, 0]
            DNQ_action2full = DNQ_action2 + [0, 1, 0]
            DNQ_action3full = DNQ_action3 + [0, 1, 0]
            DNQ_action4full = DNQ_action4 + [0, 1, 0]
            DNQ_action5full = DNQ_action5 + [0, 1, 0]
            offset = np.array([1, 4, 7, 10, 12])
            a = np.array([DNQ_action1full, DNQ_action2full, DNQ_action3full, DNQ_action4full, DNQ_action5full,
                          random_action1, random_action2, random_action3, random_action4, random_action5])
            # take action
            s_, r, done, info = env.step(a)

if __name__ == '__main__':
    train()