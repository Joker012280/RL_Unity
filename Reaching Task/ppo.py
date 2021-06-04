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

unity_env = UnityEnvironment("robot_test")
env = UnityToGymWrapper(unity_env) #uint8_visual=True)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
actor_learning_rate = 4e-4
critic_learning_rate = 4e-4
batch_size = 10
discount_factor = 0.99
eps = 0.2
max_episode_num = 20000

## 결과값 프린트 주기
print_interval = 1

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
        prev_probs, probs,states= torch.FloatTensor(), torch.FloatTensor() , torch.FloatTensor()
        next_states, rewards, done = [], [], []

        ## 받은 샘플들을 합쳐서 내보냄
        for i in range(self.buffer_size):
            prev_prob, prob,state,next_state, reward, done_ = self.memory[i]
            prev_probs = torch.cat((prev_probs,prev_prob))
            probs = torch.cat((probs,prob))
            states = torch.cat((states,state))
            next_states.append([next_state])
            rewards.append([reward])
            done.append([done_])

        next_states = torch.FloatTensor(next_states)
        ## Return 값이 각 요소에 따른 텐서들을 전달
        return prev_probs,probs,states,  \
               next_states, torch.FloatTensor(rewards), torch.FloatTensor(done)

    def clear(self):
        self.memory = []
        self.position = 0
        self.buffer_size = 0

    def size(self):
        return self.buffer_size


## Network는 학습 속도를 위해 변화할 수 있음 / Network Size는 상황에 따라 조정
## 후에 Noisy action을 위해서 critic network에 2개의 변수를 받는 걸로 변환

class Actor_network(nn.Module):
    def __init__(self):
        super(Actor_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Critic_network(nn.Module):
    def __init__(self, state_dim):
        super(Critic_network, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    m = torch.min(ratio * advantage, clipped)
    return -m



class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()


        self.critic_network = Critic_network(state_dim)
        self.critic_network_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_learning_rate)


        self.actor_network = Actor_network()
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_learning_rate)


    def train_net(self):

        if memory.size() == 0 :
            return

        prev_prob, prob,state, next_state, reward, done = memory.sample()

        advantage = reward + discount_factor * self.critic_network(next_state) - self.critic_network(state)
        if memory.size() >1 :
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)



        actor_loss = policy_loss(prev_prob.detach(),prob, advantage.detach(),eps)
        actor_loss = torch.mean(actor_loss)
        print("Actor Loss : ",actor_loss.item())
        self.actor_network_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network_optimizer.step()

        critic_loss = torch.mean(advantage)
        self.critic_network_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_network_optimizer.step()




## Train
total_reward = 0

agent = PPO()

memory = Replay_buffer()
list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'robot_my_ppo.pth'
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
    prev_action_prob = None
    while not done:
        step += 1
        state = torch.FloatTensor([state])
        probs = agent.actor_network(state)
        dist = Normal(probs, torch.tensor([0.1]))
        action = dist.sample()
        action_prob = dist.log_prob(action)
        action = action.detach().cpu().numpy()

        next_state, reward, done, _ = env.step(action)
        print("Reward : " ,reward)

        ## Trajectory
        if(step >1 ) :
            memory.push((prev_action_prob , action_prob,state ,next_state, reward, done))

        state = next_state

        total_reward += reward

        prev_action_prob = action_prob


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
    #if (num_episode % 100 ==0) and (num_episode != 0):

#plt.plot(list_total_reward)
plt.plot(list_total_reward)
plt.show()

torch.save({
            'model_state_dict': agent.state_dict(),
            }, 'robot_my_ppo.pth')

