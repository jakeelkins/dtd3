'''
SERVER of Jake's Distributed TD3 implementation. I guess call it Distr Twin Delayed DDPG
DTD3. This is done to transition to a recurrent version.

This server is essentially the replay buffer, maintained as a SumTree structure and prioritized, that
maintains and operates copies of the Q-nets and policy nets and all actors using Ray.

This implements:
- value function rescaling
- TD3 Q-val minimization of approximation errors
- prioritized experience replay
- n-step returns

TODO: optimize batch sample building to increase throughput. Optimal would be about 50x greater than current.
- this would likely need leaving about 50 cores open for that job & multiprocess it
- include a README for operation
'''

from concurrent import futures
import logging

import grpc

import dtd3_pb2
import dtd3_pb2_grpc

import time
import json
import numpy as np
import os
import math
import numpy as np
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#import gym

from envs.ADCS_gym_cont import AttitudeControlEnv

# only doing this so we dont see that annoying pickle warning. Ray uses pickle. fix in future: send as arrays
# and then iterate thru the state_dicts and cast to FloatTensors like I do in UpdateNetworks().
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn as nn
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=FutureWarning)
import ray

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

# -----------
max_timesteps = 500        # max timesteps in one episode
env = AttitudeControlEnv(torque_scale=0.5, steps=max_timesteps)

device = 'cpu'

max_episodes = 1000000   # for when we do beta annealing eventually
num_cores = multiprocessing.cpu_count()

print(f'num cores: {num_cores}')

num_actors = 8

value_fxn_epsilon = 0.001
n_step_return = 5   # n-step return to use
rescale_value_fxn = False

num_episodes_per_act = 2   # number episodes each actor runs locally before rechecking for state dict

gamma = 0.99

num_episodes_for_monitor = 10

buffer_max_size = 2**21   # must be power of 2
buffer_alpha = 0.6
buffer_beta = 0.4

batch_size = 128

max_exploration_noise = 0.3
min_exploration_noise = 0.01

state_dim = 11
action_dim = 3
max_action = float(1)
exploration_noises = np.linspace(min_exploration_noise, max_exploration_noise, num_actors)

ray.init(num_cpus=num_actors, num_gpus=0)
# -----------

class Node:
    def __init__(self, left, right, is_leaf=False, idx=None, exp_tuple=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        self.experience = exp_tuple
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx, exp_tuple):
        leaf = cls(None, None, is_leaf=True, idx=idx, exp_tuple=exp_tuple)
        leaf.value = value
        leaf.experience = exp_tuple
        return leaf


def create_tree(input_list):
    '''
    takes list of tuples in form [(experience_tuple, priority), ... ]
    '''
    # nodes = [Node.create_leaf(v, i) for i, v in enumerate(input_list)]
    nodes = []
    for i in range(len(input_list)):
        value = input_list[i][1]
        idx = i
        exp_tuple = input_list[i][0]
        curr_node = Node.create_leaf(value, idx, exp_tuple)
        nodes.append(curr_node)

    leaf_nodes = nodes

    # backfill node left and right vals
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


class PERBuffer:
    def __init__(self, max_size=buffer_max_size):
        '''
        Prioritized experience replay by Schaul et al (2015)
        implements a sumtree data structure to store priorities

        -note: no annealing of anything yet.

        tree total is value of the root_node.
        '''

        self.max_size = max_size

        self.alpha = buffer_alpha
        self.beta_initial = buffer_beta
        self.beta = self.beta_initial
        self.base_priority = 1e-6

        # generate empty buffer
        empty_list = []
        for i in range(int(max_size)):
            # sars'd. would need edited for r2d2
            empty_exp_tuple = (0., 0., 0., 0., 0.)
            empty_priority = 0.
            empty_tuple = tuple([empty_exp_tuple, empty_priority])
            empty_list.append(empty_tuple)

        root_node, leaf_nodes = create_tree(empty_list)

        self.root_node = root_node
        self.leaf_nodes = leaf_nodes

        # this keeps track of which node we update. goes 0-->max_size, repeat.
        self.leaf_node_counter = 0

        # this is used for N in the IS weights. tracks how many adds we've called.
        # maxes out at max_size.
        self.N = 0

        # checked at every update of the tree. for calc max IS weight
        self.min_priority = 9e+99

        # ---calc linear slopes for annealing based on episode---
        self.beta_slope = (1.0 - self.beta) / max_episodes

    def _update_node_priority(self, node, new_value):
        # new value is raw priority, without scale
        new_value = (new_value + self.base_priority) ** self.alpha
        change = new_value - node.value
        node.value = new_value
        self._propagate_changes(change, node.parent)

    def _propagate_changes(self, change, node):
        '''
        used internally to update the sumtree when new experience added
        '''
        node.value += change
        if node.parent is not None:
            self._propagate_changes(change, node.parent)

    def _retrieve(self, value, node):
        '''
        internal method to retrieve a value from a node. pass root_node to fully sample
        '''
        if node.is_leaf:
            return node
        if node.left.value >= value:
            return self._retrieve(value, node.left)
        else:
            return self._retrieve(value - node.left.value, node.right)

    def update_priorities(self, indices, new_values):
        '''
        called from train loop, update the priority vals with ones calcd from loop using leaf indices
        '''
        for i, idx in enumerate(indices):
            node = self.leaf_nodes[idx]
            new_value = new_values[i]

            new_value = (new_value + self.base_priority) ** self.alpha

            if new_value < self.min_priority:
                self.min_priority = new_value

            self._update_node_priority(node, new_value)

    def add(self, new_value, exp_tuple):
        '''
        really more of an "update". cycles thru buffer nodes circularly (FIFO)

        new value is raw priority, without scales. we do that here
        '''
        # first check to see where the index is
        if self.leaf_node_counter >= self.max_size:
            self.leaf_node_counter = 0

        # run this to set N to size of replay_buffer
        if self.N < self.max_size:
            self.N += 1

        node = self.leaf_nodes[self.leaf_node_counter]

        # adjustment, straight from paper. (TD_error + epsilon)^alpha
        new_value = (new_value + self.base_priority) ** self.alpha

        if new_value < self.min_priority:
            self.min_priority = new_value

        change = new_value - node.value
        node.value = new_value
        node.experience = exp_tuple

        self._propagate_changes(change, node.parent)

        self.leaf_node_counter += 1

    def sample(self):
        '''
        return one priority-sampled experience tuple and its current priority.
        '''
        # select a random uniform val for us to naviagte the tree with
        rand_val = np.random.uniform(0, self.root_node.value)
        sampled_node = self._retrieve(rand_val, self.root_node)

        sampled_exp = sampled_node.experience
        sampled_priority = sampled_node.value

        return sampled_exp, sampled_priority

    def sample_batch(self, batch_size, num_episode):
        '''
        return a batch of priority-sampled experience.
        '''
        # update beta first, linearly:
        self.beta = (self.beta_slope*num_episode) + self.beta_initial

        state, action, reward, next_state, done, is_weights, indices = [], [], [], [], [], [], []

        for i in range(batch_size):
            rand_val = np.random.uniform(0, self.root_node.value)
            sampled_node = self._retrieve(rand_val, self.root_node)

            sampled_exp = sampled_node.experience
            sampled_priority = sampled_node.value
            sampled_index = sampled_node.idx

            if sampled_priority <= 0.:
                sampled_priority = self.base_priority**self.alpha

            # --- IS weight calc ---
            # P(j) = p_j / sum(p_i)
            prob_of_j = sampled_priority / self.root_node.value
            is_weight = (self.N * prob_of_j) ** (-self.beta)

            s, a, r, s_, d = sampled_exp

            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
            is_weights.append(np.array(is_weight, copy=False))
            indices.append(np.array(sampled_index, copy=False))

        # perform normalization step like paper
        is_weights = np.array(is_weights)
        # max_is_weight = (self.N * self.min_priority / self.root_node.value) ** (-self.beta)
        # is_weights = is_weights / max_is_weight
        is_weights = is_weights / is_weights.max()

        return np.array(state).tolist(), np.array(action).tolist(), np.array(reward).tolist(), np.array(next_state).tolist(), np.array(
            done).tolist(), is_weights.tolist(), np.array(indices).tolist()


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNet, self).__init__()

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


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()

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

def print_intro():
    print('\n Welcome to... \n')
    time.sleep(1)
    print('''
  _____  _     _        _ _           _           _      _______ _____    ____  
 |  __ \(_)   | |      (_) |         | |         | |    |__   __|  __ \  |___ \ 
 | |  | |_ ___| |_ _ __ _| |__  _   _| |_ ___  __| |       | |  | |  | |   __) |
 | |  | | / __| __| '__| | '_ \| | | | __/ _ \/ _` |       | |  | |  | |  |__ < 
 | |__| | \__ \ |_| |  | | |_) | |_| | ||  __/ (_| |       | |  | |__| |  ___) |
 |_____/|_|___/\__|_|  |_|_.__/ \__,_|\__\___|\__,_|       |_|  |_____/  |____/ 
                                                                                
    ''')
    time.sleep(2)
    print('\n built by Jake Elkins, the wizard himself. I hope this works. \n\n\n ------ [begin] -----')
    time.sleep(1)

@ray.remote(num_cpus=1, num_gpus=0)
class Actor(object):

    def __init__(self):

        self.local_device = 'cpu'
        self.env = env

        self.n = n_step_return
        self.num_episodes_per_act = num_episodes_per_act

        self.actor = ActorNet(state_dim, action_dim, max_action).to(self.local_device)
        self.actor_target = ActorNet(state_dim, action_dim, max_action).to(self.local_device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = CriticNet(state_dim, action_dim).to(self.local_device)
        self.critic_1_target = CriticNet(state_dim, action_dim).to(self.local_device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = CriticNet(state_dim, action_dim).to(self.local_device)
        self.critic_2_target = CriticNet(state_dim, action_dim).to(self.local_device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action

    def _value_fxn_rescale(self, x):
        if rescale_value_fxn:
            returnx = (torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.)) + value_fxn_epsilon * x
        else:
            returnx = x
        return returnx

    def _value_fxn_rescale_inverse(self, x):
        if rescale_value_fxn:
            numerator = torch.sqrt(1. + 4. * value_fxn_epsilon * (torch.abs(x) + 1. + value_fxn_epsilon)) - 1.
            hinv = torch.sign(x) * (((numerator / (2 * value_fxn_epsilon)) ** 2) - 1.)
            returnx = hinv
        else:
            returnx = x
        return returnx

    def _select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.local_device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _generate_initial_priority(self, exp_tuple):
        # pass the state-action thru the Q network
        # NOTE: none of the TD3 smoothing is going on here.
        state = exp_tuple[0]
        action = exp_tuple[1]
        reward = exp_tuple[2]
        next_state = exp_tuple[3]
        done = exp_tuple[4]

        state = torch.from_numpy(state).float().to(self.local_device)
        action = torch.from_numpy(action).float().to(self.local_device)
        reward = torch.from_numpy(np.array([reward])).float().to(self.local_device)
        next_state = torch.from_numpy(next_state).float().to(self.local_device)
        done = torch.from_numpy(np.array([done])).float().to(self.local_device)

        next_action = self.actor_target(next_state).unsqueeze(0)

        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        next_state = next_state.unsqueeze(0)

        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = self._value_fxn_rescale_inverse(torch.min(target_Q1, target_Q2))
        target_Q = self._value_fxn_rescale(reward + ((1 - done) * (gamma ** self.n) * target_Q).detach())

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        current_Q = torch.min(current_Q1, current_Q2)

        absolute_TD_error = torch.abs(target_Q - current_Q).detach().squeeze(0).cpu().numpy()[0]

        return absolute_TD_error

    def run_episodes(self, state_dict_list, exploration_noise):
        '''
        INPUT: torch-readable versions of the 6 state-dicts. should be:
        actor, actor-target, critic1, critic1-target, critic2, ciritc2-target
        called in replay buffer with a ray.wait()
        '''
        # check to see if we've started learner yet. If not, state_dicts will be None
        if None not in state_dict_list:
            self.actor.load_state_dict(state_dict_list[0])
            self.actor_target.load_state_dict(state_dict_list[1])
            self.critic_1.load_state_dict(state_dict_list[2])
            self.critic_1_target.load_state_dict(state_dict_list[3])
            self.critic_2.load_state_dict(state_dict_list[4])
            self.critic_2_target.load_state_dict(state_dict_list[5])

            self.actor.eval()
            self.actor_target.eval()
            self.critic_1.eval()
            self.critic_1_target.eval()
            self.critic_2.eval()
            self.critic_2_target.eval()

        env = self.env
        n = self.n

        local_buffer = []
        # run X episodes
        for _ in range(self.num_episodes_per_act):

            states = []
            actions = []
            rewards = []
            dones = []

            state = env.reset()
            done = False

            while not done:
                # select action and add exploration noise:
                action = self._select_action(state)
                action = action + np.random.normal(0, exploration_noise, size=3)
                action = action.clip(-1, 1)

                # take action in env:
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))

                state = next_state
            states.append(next_state)

            # accumulator for n-step returns
            acc = 0
            returns = []
            for i, r in enumerate(reversed(rewards)):
                acc = r + gamma * acc

                if i >= n:
                    acc = acc - (list(reversed(rewards))[i - n]) * (gamma ** n)

                returns.append(acc)

            returns = list(reversed(returns))

            # recording n-th step dones and nth step states
            nth_states = []
            nth_dones = []

            for i in range(len(dones)):
                if i < (len(dones) - n):
                    nth_done = dones[i + n]
                    nth_state = states[i + n]
                else:
                    nth_done = dones[-1]
                    nth_state = states[-1]

                nth_dones.append(nth_done)
                nth_states.append(nth_state)

            # drop last val of states now, dont need it
            states = states[:-1]

            for i in range(len(states)):
                init_priority = self._generate_initial_priority(
                    (states[i], actions[i], returns[i], nth_states[i], nth_dones[i]))
                local_buffer.append((states[i], actions[i], returns[i], nth_states[i], nth_dones[i], init_priority))

        return local_buffer

# ---------------------------------------------------------------------------------------------------------------------


class BufferNetworkHandler(dtd3_pb2_grpc.LearnerServicer):
    # this class just maintains the easy methods we need for reading, writing, and changing networks
    def __init__(self):
        # note: I could probably fix these around better
        self.actor_dict = None
        self.actor_target_dict = None

        self.critic_1_dict = None
        self.critic_1_target_dict = None

        self.critic_2_dict = None
        self.critic_2_target_dict = None

        self.network_list = [self.actor_dict, self.actor_target_dict,
                             self.critic_1_dict, self.critic_1_target_dict,
                             self.critic_2_dict, self.critic_2_target_dict]

        self.replay_buffer = PERBuffer()

        self.actors = [Actor.remote() for _ in range(num_actors)]

        # this was used for debug
        self.msg = None
        self.indices = None
        self.reading_priorities = None
        self.reading_indices = None
        self.first_dict = None

        # -- for monitor --
        self.num_episodes_for_monitor = num_episodes_for_monitor
        self.actor_copy = ActorNet(state_dim, action_dim, max_action).to('cpu')

    def UpdateNetworks(self, request_iterator, context):
        '''
        this handles the sending of both networks and new priorities.
        messages will differ in where the Nones are
        update networks:
        [actor, actor_target, critic1, critic1_target, critic2, critic2_target, None, None]
        update priorities:
        [None, None, None, None, None, None, indices, priorities]
        '''

        '''for i, request in enumerate(request_iterator):
            curr_dict = json.loads(request.network_params)
            # switch network back into torch-readable tensors
            for entry in curr_dict:
                curr_dict[entry] = torch.FloatTensor(curr_dict[entry]).to(device)

            self.network_list[i] = curr_dict'''
        indices = None
        priorities = None

        for i, request in enumerate(request_iterator):
            curr_dict = json.loads(request.network_params)
            if (i <= 5) and (curr_dict is not None):
                # these are still state_dicts
                for entry in curr_dict:
                    curr_dict[entry] = torch.FloatTensor(curr_dict[entry]).to(device)

                self.network_list[i] = curr_dict
            elif (i > 5) and (curr_dict is not None):
                # we're reading priorities
                if i == 6:
                    self.reading_indices = True
                    indices = curr_dict
                elif i == 7:
                    self.reading_priorities = True
                    priorities = curr_dict

        if (indices is not None) and (priorities is not None):
            self.replay_buffer.update_priorities(indices, priorities)
            self.indices = indices

        if (None not in self.network_list) and (self.first_dict is None):
            self.first_dict = True

        return dtd3_pb2.BufferStatus(status=1)

    def act(self):
        '''
        this is where the Ray actors are sent and handled. note the infinite loop: this guy has to be terminated.
        :return:
        nil
        '''

        print(f'acting started. running {num_actors} across CPUs')
        loop_num = 0
        while True:
            loop_num += 1
            # print(f'reading indices: {self.reading_indices}')
            # print(f'reading priorities: {self.reading_priorities}')
            if (None in self.network_list) and (self.first_dict):
                print(f"[!] ERROR [!] network list contains a NoneType [!] ERROR [!]")

            not_done_actor_ids = []
            for i, actor in enumerate(self.actors):
                exploration_noise = exploration_noises[i]
                not_done_actor_ids.append(actor.run_episodes.remote(self.network_list, exploration_noise))

            while len(not_done_actor_ids):
                done_actor_ids, not_done_actor_ids = ray.wait(not_done_actor_ids)

                # if done_id == actors, add to replay buffer
                if done_actor_ids:
                    for actor_id in done_actor_ids:
                        local_buffer_out = ray.get(actor_id)
                        for tuple_ in local_buffer_out:
                            priority = tuple_[-1]
                            exp_tuple = tuple_[:-1]
                            self.replay_buffer.add(priority, exp_tuple)

            if loop_num%100 == 0:
                print(f'Current buffer size: {self.replay_buffer.N}')
                # print(f'indices read: {self.indices}')
                # print(f'curr main dict (should be None before the epoch): {self.network_list[0]}')

    def ReadData(self, request, context):

        # print(f'read_data request received. status: {request.status}')

        # TODO: tune beta annealing. Zero for now (constant at initial beta val)
        state, action_, reward, next_state, done, is_weights, indices = self.replay_buffer.sample_batch(batch_size, 0)

        send_list = [state, action_, reward, next_state, done, is_weights, indices]

        for data_section in send_list:
            #print(data_section)
            #data_section_json = json.dumps(data_section.tolist())
            data_section_json = json.dumps(data_section)
            yield dtd3_pb2.BufferResponse(train_data=bytes(data_section_json, 'utf-8'))

    def RunAgentStats(self, request, context):
        '''
        RPC that the monitor asks for. this asks for an agent stat update using current networks.
        '''
        #print(f'actor_copy keys: {self.actor_copy.state_dict().keys()}')

        # load up most recent policy net
        if self.network_list[0] is not None:
            #print(f'network_list keys: {self.network_list[0].keys()}')
            self.actor_copy.load_state_dict(self.network_list[0])

        rewsum_list = []
        meanminang_list = []

        for ep_num in range(self.num_episodes_for_monitor):

            done = False
            obs = env.reset()

            reward_list = []
            obs_list = []
            
            while not done:
                #act = policy.select_action(obs)
                obs = torch.FloatTensor(obs.reshape(1, -1)).to('cpu')
                act = self.actor_copy(obs).cpu().data.numpy().flatten()
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

        return dtd3_pb2.AgentStats(reward=mean_rew_sum, additional_data=mean_min_ang)


# ---------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig()

    handler = BufferNetworkHandler()

    # --- server specifics ---
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dtd3_pb2_grpc.add_LearnerServicer_to_server(handler, server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print_intro()

    try:
        # start acting
        handler.act()
        server.wait_for_termination() # FOR DEBUG
    except KeyboardInterrupt as e:
        print(f'ERROR: terminated. {e}')
        print(f'please wait for safe shutdown...')
        if ray.is_initialized():
            ray.shutdown()
        server.stop(3)
# TODO: print buffer checkup and stats every now and then. also--ray handles ctrl-c really weirdly. how do we exit
#  before errors are caught?

# EOF
