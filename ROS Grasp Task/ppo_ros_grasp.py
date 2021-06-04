## For the TD3
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from IPython.display import clear_output
import collections
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import time




## For the Unity env
#from baselines import deepq
#from baselines import logger

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

unity_env = UnityEnvironment("robot_test2")
env = UnityToGymWrapper(unity_env) #uint8_visual=True)


device = torch.device("cuda" if torch.cuda.isavailable() else "cpu")

##cuda = torch.device('cuda')

'''



action_dim = env.action_space.shape[0]

for i in range(100) :
    env.reset()
    done = False
    while not done :
        action = np.random.random(action_dim)
        print(action)
        state,reward,done,_ =env.step(action)
        print(state.shape)

'''
print("Start")




## Hyperparameter
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
batch_size = 10
discount_factor = 0.99
eps = 0.2
max_episode_num = 20000

## 결과값 프린트 주기
print_interval = 10

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

## Replay_Buffer(Experience Replay)
class Replay_buffer():
    def __init__(self, max_size=1000):
        self.memory = []
        self.max_size = max_size
        self.position = 0
        self.buffer_size = 0

    def push(self, data):
        if len(self.memory) == self.max_size:
            self.memory[int(self.position)] = data
            self.position = (self.position + 1) % self.max_size
        else:
            self.memory.append(data)
            self.buffer_size += 1

    def sample(self):
        old_actions,probs,states= torch.FloatTensor().to(device),torch.FloatTensor().to(device), torch.FloatTensor().to(device)
        next_states, rewards, done = [], [], []

        ## 받은 샘플들을 합쳐서 내보냄
        for i in range(self.buffer_size):
            old_action,prob,state,next_state, reward, done_ = self.memory[i]
            old_actions = torch.cat((old_actions,old_action))
            probs = torch.cat((probs,prob))
            states = torch.cat((states,state))
            next_states.append([next_state])
            rewards.append([reward])
            done.append([done_])

        next_states = torch.FloatTensor(next_states)
        ## Return 값이 각 요소에 따른 텐서들을 전달
        return old_actions.detach(),probs,states.detach(),  \
               next_states, torch.FloatTensor(rewards).to(device), torch.FloatTensor(done)

    def clear(self):
        self.memory = []
        self.position = 0
        self.buffer_size = 0

    def size(self):
        return self.buffer_size




class Model(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Model,self).__init__()

        self.actor_layer = nn.Sequential(
            nn.Conv2d(in_channels= 3 ,out_channels= 16, kernel_size= 5 , stride =2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels= 16 ,out_channels= 32 , kernel_size= 5 , stride =2),
            nn.BatchNorm2d(32),

        )
        self.actor_fn = nn.Sequential(
            nn.Linear(32*13*13,128),
            nn.Tanh(),
            nn.Linear(128,action_dim)
        )
        self.critic_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
        )
        self.critic_fn = nn.Sequential(
            nn.Linear(32 * 13 * 13, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )



    def forward(self):
        raise NotImplementedError

    def act(self,state):
        x = self.actor_layer(state)
        x = x.view((x.size(0),-1))
        probs = self.actor_fn(x)
        dist = Normal(probs, torch.tensor([0.1]).to(device))
        action = dist.sample()
        action_prob = dist.log_prob(action)
        return action,action_prob

    def evaluate(self,state,action):
        x = self.actor_layer(state)
        x = x.view((x.size(0),-1))
        probs = self.actor_fn(x)
        dist = Normal(probs,torch.tensor([0.1]).to(device))
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        x = self.critic_layer(state)
        x = x.view((x.size(0),-1))
        value = self.critic_fn(x)

        return action_logprob, dist_entropy, value




def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    m = torch.min(ratio * advantage, clipped)
    return -m



class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()


        self.network = Model(state_dim,action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = actor_learning_rate)
        self.old_network = Model(state_dim,action_dim).to(device)
        self.old_network.load_state_dict(self.network.state_dict())

        self.mse = nn.MSELoss()
        self.epoch = 3

    def train_net(self):

        if memory.size() == 0 :
            return

        old_actions,old_prob,state, next_state, reward, done = memory.sample()


        for _ in range(self.epoch) :
            log_prob,entropy,state_value = self.network.evaluate(state,old_actions)

            ratio = torch.exp(log_prob - old_prob.detach())

            advantages = reward - state_value.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,1-eps,1+eps) * advantages
            loss = - torch.min(surr1,surr2) + 0.5*self.mse(state_value,reward) - 0.2*entropy
            print("Total Loss : ",loss.mean().item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_network.load_state_dict(self.network.state_dict())




## Train
total_reward = 0

agent = PPO()

memory = Replay_buffer()
list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'robot_ppo_making_grasp.pth'
## 전에 사용했던 모델 가져오기
load = False
if load == True:
    temp = torch.load(PATH)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()


for num_episode in range(max_episode_num):
    state = env.reset()
    step = 0
    done = False
    reward = 0
    state = state.transpose((2,0,1))
    state = state/255.0
    prev_action_prob = None
    while not done:
        step += 1

        state = torch.FloatTensor([state]).to(device)
        mem_state = state
        state.unsqueeze(0)

        action,action_prob = agent.old_network.act(state)
        mem_action = action
        #dist = Normal(probs, torch.tensor([0.1]))
        #action = dist.sample()
        #action_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy()

        next_state, reward, done, _ = env.step(action)
        print("Reward : " ,reward)

        ## Trajectory
        if(step >1 ) :
            memory.push((mem_action, action_prob,mem_state ,next_state, reward, done))
        next_state = next_state.transpose((2,0,1))
        state = next_state/255.0

        total_reward += reward

        ##prev_action_prob = action_prob


        ## Memory size가 커지고 나서 학습시작
        if (memory.size() == batch_size) and (memory.size() != 0):
            agent.train_net()
            memory.clear()

        if done:
            agent.train_net()
            memory.clear()
            break


    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)
        total_reward = 0.0
        torch.save({
            'model_state_dict': agent.state_dict(),
        }, 'robot_ppo_making_grasp.pth')
    #if (num_episode % 100 ==0) and (num_episode != 0):

#plt.plot(list_total_reward)
plt.plot(list_total_reward)
plt.show()

torch.save({
            'model_state_dict': agent.state_dict(),
            }, 'robot_my_ppo_making.pth')

