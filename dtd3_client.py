'''
Jake's Distributed TD3 client. This is the learner, which just pulls in experience from the buffer, trains, and sends
back updated networks and new priorities.
'''

from __future__ import print_function
import logging
import json

import grpc

import dtd3_pb2
import dtd3_pb2_grpc

import numpy as np
import time
import os
import math
import multiprocessing

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from tensorboardX import SummaryWriter

from envs.ADCS_gym_cont import AttitudeControlEnv

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

# ---- setup ----
port = 'localhost:50051'   # !! ENSURE this is same as server

logpath = './log/20210227_monitor/'
runname = '20210227_monitor_test1'



value_fxn_epsilon = 0.001
n_step_return = 5
rescale_value_fxn = False

num_tests = 1
log_interval = 5000           # print avg reward after interval
send_network_interval = 10
gamma = 0.99                # discount for future rewards
lr = 3e-4
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter (paper suggests 2)
max_timesteps = 500        # max timesteps in one episode

env = AttitudeControlEnv(torque_scale=0.5, steps=max_timesteps)

if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'\n\n using learner hardware: {torch.cuda.get_device_name(torch.cuda.current_device())} \n\n')
else:
    raise Exception('GPU not found for learner. Reprogram for CPU learner. Jake hasnt done this yet. Use the GPU.')

state_dim = 11
action_dim = 3
max_action = float(1)

os.makedirs(logpath, exist_ok=True)

logpath2 = f'./agents/{runname}/'
os.makedirs(logpath2, exist_ok=True)


'''
see https://arxiv.org/pdf/1805.11593.pdf, proposition A.2
'''


def value_fxn_rescale(x):
    if rescale_value_fxn:
        returnx = (torch.sign(x)*(torch.sqrt(torch.abs(x) + 1.) - 1.)) + value_fxn_epsilon*x
    else:
        returnx = x
    return returnx


def value_fxn_rescale_inverse(x):
    if rescale_value_fxn:
        numerator = torch.sqrt(1. + 4.*value_fxn_epsilon*(torch.abs(x) + 1. + value_fxn_epsilon)) - 1.
        hinv = torch.sign(x)*(((numerator/(2*value_fxn_epsilon))**2) - 1.)
        returnx = hinv
    else:
        returnx = x
    return returnx


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        # a = F.relu(self.l12(a))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.l1(state_action))
        # q = F.relu(self.l12(q))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action

        self.n_update = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def generate_initial_priority(self, exp_tuple):
        # pass the state-action thru the Q network
        # NOTE: none of the TD3 smoothing is going on here.
        state = exp_tuple[0]
        action = exp_tuple[1]
        reward = exp_tuple[2]
        next_state = exp_tuple[3]
        done = exp_tuple[4]

        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(np.array([reward])).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        done = torch.from_numpy(np.array([done])).float().to(device)

        next_action = self.actor_target(next_state).unsqueeze(0)

        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        next_state = next_state.unsqueeze(0)

        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = value_fxn_rescale_inverse(torch.min(target_Q1, target_Q2))
        target_Q = value_fxn_rescale(reward + ((1 - done) * (gamma ** n_step_return) * target_Q).detach())

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        current_Q = torch.min(current_Q1, current_Q2)

        absolute_TD_error = torch.abs(target_Q - current_Q).detach().squeeze(0).cpu().numpy()[0]

        return absolute_TD_error

    def update(self, state, action_, reward, next_state, done, is_weights, indices, gamma, polyak,
               policy_noise, noise_clip, policy_delay):

        self.n_update += 1

        # NOTE: might want to go back and just use critic1 vals and not do min().
        # Sample a batch of transitions from replay buffer:
        # this is the "transition tuple" like sarsa but just SARS'
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action_).to(device)
        reward = torch.FloatTensor(reward).reshape((-1, 1)).to(device)
        # APPLY BATCH NORM ON REWARD
        # reward = (reward - reward.mean())/reward.std()
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape((-1, 1)).to(device)
        is_weights = torch.FloatTensor(is_weights).reshape((-1, 1)).to(device)

        # Select next action according to target policy:
        noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise)
        next_action = next_action.clamp(-self.max_action, self.max_action)

        # Compute target Q-value:
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = value_fxn_rescale_inverse(torch.min(target_Q1, target_Q2))
        target_Q = value_fxn_rescale(reward + ((1 - done) * (gamma**n_step_return) * target_Q).detach())

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        current_Q = torch.min(current_Q1, current_Q2)
        abs_TD_errors = torch.abs(target_Q - current_Q).detach().squeeze(1).cpu().numpy()
        # send these back to buffer to update

        # Optimize Critic 1:
        loss_Q1 = (is_weights * ((current_Q1 - target_Q) * (current_Q1 - target_Q))).mean()
        # loss_Q1 = (F.mse_loss(current_Q1, target_Q))
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        loss_Q2 = (is_weights * ((current_Q2 - target_Q) * (current_Q2 - target_Q))).mean()
        # loss_Q2 = (F.mse_loss(current_Q2, target_Q))
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates:
        if self.n_update % policy_delay == 0:
            # Compute actor loss:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak averaging update:
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

        return indices, abs_TD_errors, loss_Q1


def state_dict_to_json_generator(state_dict_list, indices=None, priorities=None):
    if priorities:
        new_list = [None]*6
    else:
        # convert torch state dict to numpy
        new_list = []
        for state_dict in state_dict_list:
            new_dict = {}
            for entry in state_dict:
                new_dict[entry] = state_dict[entry].cpu().data.numpy().tolist()
            new_list.append(new_dict)

    if priorities:
        new_list.append(indices)
        new_list.append(priorities)
    else:
        new_list.append(None)
        new_list.append(None)

    for dictx in new_list:
        json_out = json.dumps(dictx)
        yield dtd3_pb2.LearnerSend(network_params=bytes(json_out, 'utf-8'))


def save_all_nets(policy, fname):
    torch.save(policy.actor.state_dict(), fname +'_policy'+ '.dat')
    torch.save(policy.actor_target.state_dict(), fname +'_policy_target'+ '.dat')
    torch.save(policy.critic_1.state_dict(), fname +'_critic_1'+ '.dat')
    torch.save(policy.critic_1_target.state_dict(), fname +'_critic_1_target'+ '.dat')
    torch.save(policy.critic_2.state_dict(), fname +'_critic_2'+ '.dat')
    torch.save(policy.critic_2_target.state_dict(), fname +'_critic_2_target'+ '.dat')


def find_agent_status(num_tests, policy):
    # run for given num episodes, calc stats, using no-noise policy
    rewsum_list = []
    meanminang_list = []
    for _ in range(num_tests):
        
        done = False
        obs = env.reset()

        reward_list = []
        obs_list = []
        
        while not done:
            act = policy.select_action(obs)
            # action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            act = act.clip(-1, 1)
            obs, reward, done, _ = env.step(act)
            reward_list.append(reward)
            obs_list.append(obs)
        
        q4 = [i[0] for i in obs_list]
        q1 = [i[1] for i in obs_list]
        q2 = [i[2] for i in obs_list]
        q3 = [i[3] for i in obs_list]

        curr_minang = 2*np.arccos(np.max(q4))*(180/np.pi)
        curr_rew_sum = np.sum(reward_list)
        
        rewsum_list.append(curr_rew_sum)
        meanminang_list.append(curr_minang)
    
    mean_min_ang = np.mean(meanminang_list)
    mean_rew_sum = np.mean(rewsum_list)
    
    return mean_rew_sum, mean_min_ang


def find_agent_status2(num_tests, policy):
    # run for given num episodes, calc stats, using no-noise policy
    rewsum_list = []

    for _ in range(num_tests):

        done = False
        obs = env.reset()

        reward_list = []
        obs_list = []

        while not done:
            env.render()
            act = policy.select_action(obs)
            act = act.clip(-1, 1)
            obs, reward, done, _ = env.step(act)
            reward_list.append(reward)
            obs_list.append(obs)
        env.close()

        curr_rew_sum = np.sum(reward_list)

        rewsum_list.append(curr_rew_sum)

    mean_rew_sum = np.mean(rewsum_list)

    return mean_rew_sum


def train(learner_stub):
    global epoch

    #learner_stub = dtd3_pb2_grpc.LearnerStub(channel)

    #epoch = 0   # is this where an issue is?

    t1 = time.time()

    while True:
        # --------------fetch data from buffer----------------
        # note: could anneal beta from here. predefined batch size, from buffer side?
        responses = learner_stub.ReadData(dtd3_pb2.LearnerRequest(status=1))
        # ----- reading sampled data -----
        # TODO: switch to collections.Deque type
        states, actions, rewards, next_states, dones, is_weights, indices = [], [], [], [], [], [], []

        for i, response in enumerate(responses):    # error happens here. need a checker before errors come in.
            if i == 0:
                states = np.array(json.loads(response.train_data))
            elif i == 1:
                actions = np.array(json.loads(response.train_data))
            elif i == 2:
                rewards = np.array(json.loads(response.train_data))
            elif i == 3:
                next_states = np.array(json.loads(response.train_data))
            elif i == 4:
                dones = np.array(json.loads(response.train_data))
            elif i == 5:
                is_weights = np.array(json.loads(response.train_data))
            elif i == 6:
                indices = np.array(json.loads(response.train_data))

        # ------------------------update policy-----------------------------
        indices, abs_TD_errors, loss_Q1 = policy.update(states, actions, rewards, next_states, dones, is_weights, indices, gamma,
                                               polyak, policy_noise, noise_clip, policy_delay)

        net_params_generator = state_dict_to_json_generator([], indices=indices.tolist(),
                                                            priorities=abs_TD_errors.tolist())
        response = learner_stub.UpdateNetworks(net_params_generator)

        if response.status == 1:
            pass
        else:
            print('[!] ERROR: priority send failure [!]')
            # TODO: handle something here.

        epoch += 1

        if epoch % send_network_interval == 0:

            curr_dict_list = [
                policy.actor.state_dict(),
                policy.actor_target.state_dict(),
                policy.critic_1.state_dict(),
                policy.critic_1_target.state_dict(),
                policy.critic_2.state_dict(),
                policy.critic_2_target.state_dict(),
            ]

            net_params_generator = state_dict_to_json_generator(curr_dict_list)

            response = learner_stub.UpdateNetworks(net_params_generator)

            if response.status == 1:
                pass
            else:
                print('[!] ERROR: network send failure [!]')
                # TODO: handle something here.

        # --------------------logging step-------------------
        # save all nets and then send to writer and print
        if epoch % log_interval == 0:
            t2 = time.time()

            save_all_nets(policy, logpath2 + runname + '_latest')

            #curr_avg_reward, mean_min_ang = find_agent_status(num_tests, policy)

            #writer.add_scalar("curr avg reward", curr_avg_reward, epoch)

            print(f'Train epoch: {epoch} \t Loss: {round(loss_Q1.cpu().detach().numpy().tolist(), 4)} \t Current throughput (updates/sec): {round(log_interval/(t2-t1), 4)}')

            t1 = time.time()

        if epoch % 20000 == 0:
            # TODO: tune this scale of saving
            save_all_nets(policy, logpath2 + runname + '_' + str(epoch))


if __name__ == '__main__':
    logging.basicConfig()
    print(f'connecting to port {port}...')
    channel = grpc.insecure_channel(port)
    print(f'connected to port {port}. building policy...')
    policy = TD3(lr, state_dim, action_dim, max_action)
    print(f'policy built.')

    # logging variables:
    #writer = SummaryWriter(logpath + runname)
    #print(f'Tensorboard is running at path: {logpath + runname}.')

    learner_stub = dtd3_pb2_grpc.LearnerStub(channel)

    print('now starting training.')

    print('\n ------ [begin] ------ \n')
    epoch = 0

    while True:
        try:
            train(learner_stub)
        except grpc.RpcError as e:
            print(f'[!] ERROR: {e}')
        except KeyboardInterrupt as e:
            print(f'\n ERROR: terminated. {e}')
            print(f'please wait for safe shutdown...')
            # TODO: put extra save here?
            channel.close()
            break

