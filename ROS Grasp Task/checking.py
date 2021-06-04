## For the TD3
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
batch_size = 128
discount_factor = 0.99
tau = 0.005
max_episode_num = 5000

## 결과값 프린트 주기
print_interval = 1

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


## Noise Generator /// Td3 는 target action에 노이즈를 추가한다.
class Noisegenerator:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def generate(self):
        return np.random.normal(self.mean, self.sigma, 1)[0]


## Replay_Buffer(Experience Replay) => Off-policy
class Replay_buffer():
    def __init__(self, max_size=100000):
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

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.memory), size=batch_size)
        states, actions, next_states, rewards, done = [], [], [], [], []
        ## state , action, next_state , done , reward에 따라 나눔

        ## 받은 샘플들을 합쳐서 내보냄
        for i in ind:
            state, action,next_state, reward, done_ = self.memory[i]
            states.append([state])
            actions.append([action])
            next_states.append([next_state])
            rewards.append([reward])
            done.append([done_])

        states = torch.FloatTensor(states)
        states = states.squeeze()
        next_states = torch.FloatTensor(next_states)
        next_states = next_states.squeeze()


        ## Return 값이 각 요소에 따른 텐서들을 전달
        return states.to(device), torch.FloatTensor(actions).to(device), \
               next_states.to(device), torch.FloatTensor(rewards).to(device), torch.FloatTensor(done).to(device)

    def size(self):
        return self.buffer_size


## Network는 학습 속도를 위해 변화할 수 있음 / Network Size는 상황에 따라 조정
## 후에 Noisy action을 위해서 critic network에 2개의 변수를 받는 걸로 변환

class Actor_network(nn.Module):
    def __init__(self,action_dim):
        super(Actor_network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= 3 ,out_channels= 16, kernel_size= 5 , stride =2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels= 16 ,out_channels= 32 , kernel_size= 5 , stride =2)
        self.bn2 = nn.BatchNorm2d(32)


        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        ## 64 means shape of input
        convw = conv2d_size_out(conv2d_size_out(64))
        convh = conv2d_size_out(conv2d_size_out(64))
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size,64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view((x.size(0),-1))

        x = F.relu(self.fc1(x))
        ## Action의 범위 생각
        x = self.fc2(x)
        return x


class Critic_network(nn.Module):
    def __init__(self, action_dim):
        super(Critic_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        ## 64 means shape of input
        convw = conv2d_size_out(conv2d_size_out(64))
        convh = conv2d_size_out(conv2d_size_out(64))
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 64)
        ## For action
        self.fc_1 = nn.Linear(action_dim, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,1)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view((x.size(0), -1))
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc_1(y))

        x = x.squeeze()
        y = y.squeeze()

        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x





class TD3(nn.Module):

    def __init__(self):
        super(TD3, self).__init__()

        ## Twin Q network
        self.critic1_network = Critic_network(action_dim).to(device)
        self.critic2_network = Critic_network(action_dim).to(device)

        ## Optimize 할 때 두개의 네트워크를 같이함.
        self.critic_network_optimizer = optim.Adam(list(self.critic1_network.parameters()) \
                                                   + list(self.critic2_network.parameters()), lr=critic_learning_rate)

        ## Actor Network 설정
        self.actor_network = Actor_network(action_dim).to(device)
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_learning_rate)

        ## Target Network를 생성
        ## DeepCopy를 이용함. 서로 영향을 안줌

        self.critic1_target_network = copy.deepcopy(self.critic1_network)
        self.critic2_target_network = copy.deepcopy(self.critic2_network)
        self.actor_target_network = copy.deepcopy(self.actor_network)

        self.noise_generator = Noisegenerator(0, 0.2)

        ## TD3 는 Delayed 라는 특성을 가지고 있음
        self.train_num = 1
        self.delay = 2

    def train_net(self):
        state, action, next_state, reward, done = memory.sample(batch_size)
        ## Cliping 과 noise를 추가함. / Exploration 효과
        ## Pseudo Code 12,13번 참고
        noisy_action = self.actor_target_network(next_state) + torch.tensor(
            np.clip(self.noise_generator.generate(), -0.5, 0.5))
        noisy_action = torch.clamp(noisy_action, -2, 2).detach()
        noisy_action.to(device)
        ## Twin 이기 때문에 2개의 네트워크에서 값을 가져옴
        backup_value = reward + discount_factor * torch.min(self.critic1_target_network(next_state, noisy_action), \
                                                            self.critic2_target_network(next_state, noisy_action))

        ## 두개의 네트워크를 이용해 MSBE loss 구함
        ## Pseudo Code 14번 참고
        q_loss = F.mse_loss(backup_value.detach(), self.critic1_network(state, action)) \
                 + F.mse_loss(backup_value.detach(), self.critic2_network(state, action))

        ## Optimizer (Critic Network)
        self.critic_network_optimizer.zero_grad()
        q_loss.backward()
        self.critic_network_optimizer.step()

        ## Delay (Optimize 하는 시간이 즉각적으로 이루어지지않음)
        ## Pseudo Code 15,16번 참고
        if self.train_num % self.delay == 0:
            ## Optimizer (Actor Network)
            actor_loss = -self.critic1_network(state, self.actor_network(state)).mean()
            print('Actor_Loss : ', actor_loss.item())
            self.actor_network_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_network_optimizer.step()

            ## Parameter Copy
            ## Pseudo code 17번 참고 / Polyak Average
            for param, target_param in zip(self.critic1_network.parameters(), self.critic1_target_network.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic2_network.parameters(), self.critic2_target_network.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor_network.parameters(), self.actor_target_network.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.train_num += 1


## Train
total_reward = 0

agent = TD3()
memory = Replay_buffer()

list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'robot_my.pth'
## 전에 사용했던 모델 가져오기
load = False
if load == True:
    temp = torch.load(PATH)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()


for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    state = state.transpose((2, 0, 1))

    while not done:
        global_step += 1
        replay_state = state
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        action = agent.actor_network(state.to(device))
        ## noise 추가

        action = action.detach().cpu().numpy()
        ## Action 값이 범위를 넘어서지 않도록 설정

        next_state, reward, done, _ = env.step(action)
        print("Reward : " ,reward)
        next_state = next_state.transpose((2,0,1))

        ## Replay Buffer의 저장
        ##memory.push((replay_state, action, next_state, reward, done))

        state = next_state

        total_reward += reward

        if done:
            break
        ## Memory size가 커지고 나서 학습시작
        #if memory.size() > 500:
            #agent.train_net()
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
            }, 'robot_my.pth')

