import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
FLOAT = torch.FloatTensor
import torch.nn.functional as F
from torch.distributions import Categorical
from auto_esn.esn.esn import GroupedDeepESN2
from auto_esn.esn.reservoir.activation import self_normalizing_default, tanh
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer

seed = 3
def regular_graph_initializer(seed, degree):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight = CompositeInitializer().with_seed(seed).uniform()

    reservoir_weight = CompositeInitializer() \
        .with_seed(seed) \
        .uniform() \
        .regular_graph(degree) \
        .spectral_normalize() \
        .scale(1.)

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)

def sparse_initializer(seed, density=0.2, spectral_radius=0.9):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight = CompositeInitializer().with_seed(seed).uniform()

    reservoir_weight = CompositeInitializer() \
        .with_seed(seed) \
        .uniform() \
        .sparse(density) \
        .spectral_normalize(spectral_radius) \
        .scale(1.)

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    print('Target update!')

def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

criterion = nn.MSELoss()


class DDPGRC:
    def __init__(self, N_STATES, N_ACTIONS, batch_size, memory_size, lr_actor, lr_critic, hidden_unit_1, hidden_unit_2,
                 train_iter=32, train_steps=1000000, density=0.9, n_layers=[1], units=50, leaky=[1.0,0.3], sr=0.9, tag=0, 
                 one_state=0):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.BATCH_SIZE = batch_size
        self.LR_actor = lr_actor  # learning rate
        self.LR_critic = lr_critic
        self.one_state = one_state

        self.GAMMA = 0.9  # reward discount
        self.epsilon = 1
        self.depsilon = 1/train_steps

        self.MEMORY_CAPACITY = memory_size
        self.hidden_unit_1 = hidden_unit_1
        self.hidden_unit_2 = hidden_unit_2
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 1 + self.N_ACTIONS))
        self.train_iter = train_iter
        self.replace_counter = 1
        self.s_his = []
        self.tag = tag

        self.tau = 0.001
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_mu = 0
        self.random_process = OrnsteinUhlenbeckProcess(size=N_ACTIONS, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)

        groups = sum(n_layers)

        self.actor = Actor(nb_states=self.N_STATES, nb_actions=self.N_ACTIONS, hidden1=self.hidden_unit_1,
                           hidden2=self.hidden_unit_2, groups=groups,num_layers=n_layers,density=density,
                           hidden_size=units, regularization=1.0, washout=0, lr=leaky, sr=sr, tag=tag, one_state=one_state)

        self.actor_target = Actor(nb_states=self.N_STATES, nb_actions=self.N_ACTIONS, hidden1=self.hidden_unit_1,
                                  hidden2=self.hidden_unit_2, groups=groups,num_layers=n_layers,density=density,
                                  hidden_size=units, regularization=1.0, washout=0, lr=leaky, sr=sr, tag=tag,one_state=one_state)
        self.actor_optim = Adam(self.actor.para, lr=self.LR_actor)

        self.critic = Critic(nb_states=self.N_STATES, nb_actions=self.N_ACTIONS, hidden1=self.hidden_unit_1,
                             hidden2=self.hidden_unit_2, groups=groups,num_layers=n_layers,density=density,
                             hidden_size=units, regularization=1.0, washout=0, lr=leaky, sr=sr, tag=tag, one_state=one_state)

        self.critic_target = Critic(nb_states=self.N_STATES, nb_actions=self.N_ACTIONS, hidden1=self.hidden_unit_1,
                                    hidden2=self.hidden_unit_2, groups=groups,num_layers=n_layers,density=density,
                                    hidden_size=units, regularization=1.0, washout=0, lr=leaky, sr=sr, tag=tag, one_state=one_state)
        self.critic_optim = Adam(self.critic.para, lr=self.LR_critic)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def choose_action(self, obs, decay_epsilon=True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(obs).detach().numpy().squeeze(0)
        noise = self.random_process.sample()
        # print('decayed noise', max(self.epsilon, 0) * noise)

        if np.sum(np.isnan(noise)) > 0:
            noise.reset_states()
        noise[np.isnan(noise)] = 0  # avoid bug of nan action, do not know what cause it yet.


        action += max(self.epsilon, 0) * noise

        action = np.clip(action, -1., 1.)
        if decay_epsilon:
            self.epsilon -= self.depsilon

        action[np.isnan(action)] = -1  # avoid bug of nan action, do not know what cause it yet.
        return action

    def choose_action_test(self, obs, type):
        if type == 0:
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actor(obs).detach().numpy().squeeze(0)
            action = np.clip(action, -1., 1.)
        elif type == 1 or type == 3:  # when action type is 3, define fixed action in the scheduler code.
            action = 0
        elif type == 2:
            action = np.random.uniform(-1., 1., self.N_ACTIONS)
        else:
            raise Exception('test type not defined')
        return action

    def learn(self):
        actor_loss_his = []
        critic_loss_his = []

        if self.MEMORY_CAPACITY < self.memory_counter:
            memo_index = self.MEMORY_CAPACITY 
    
        else:
            memo_index = self.memory_counter

        for i in range(self.train_iter):
            # Pick training samples from replay memory
            # if self.memory_counter > self.MEMORY_CAPACITY:
            #     sample_index = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
            # else:
            #     sample_index = np.random.choice(self.memory_counter, size=self.BATCH_SIZE)

            if memo_index < self.BATCH_SIZE:
                batch_memory = self.memory[:memo_index,:]
                indexs = np.arange(0,memo_index)
            else:
                sample_index = np.random.randint(low=0, high = memo_index-self.BATCH_SIZE+1)
                indexs = np.arange(sample_index, sample_index + self.BATCH_SIZE)

            batch_memory = self.memory[indexs[:self.BATCH_SIZE]] 
                   
            # batch_memory = self.memory[sample_index, :]

            actor_loss, critic_loss = self._train(batch_memory)
            actor_loss_his.append(actor_loss.detach().numpy())
            critic_loss_his.append(critic_loss.detach().numpy())
        # self.replace_counter += 1
        # if self.replace_counter % self.replace_iter == 0:
        #     hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        #     hard_update(self.critic_target, self.critic)
            self.actor.reset()
            self.critic.reset()
            self.actor_target.reset()
            self.critic_target.reset()
        for i in range(5):
            print('*****target soft update*****')
        return np.mean(actor_loss_his), np.mean(critic_loss_his)

    def _train(self, batch_memory):
        b_s = torch.FloatTensor(batch_memory[:, :self.N_STATES]) #(batch, state_size)
        b_a = torch.FloatTensor(batch_memory[:, self.N_STATES:self.N_STATES+self.N_ACTIONS]) #(batch, 1)
        b_r = torch.FloatTensor(batch_memory[:, self.N_STATES+self.N_ACTIONS:self.N_STATES+self.N_ACTIONS+1])  #(batch, 1)
        b_s_ = torch.FloatTensor(batch_memory[:, self.N_STATES+self.N_ACTIONS+1:self.N_STATES*2+1+self.N_ACTIONS]) #(batch, state_size)

        # b_s = batch_memory[:, :self.N_STATES] #(batch, state_size)
        # b_a = batch_memory[:, self.N_STATES:self.N_STATES+1] #(batch, 1)
        # b_r = batch_memory[:, self.N_STATES+1:self.N_STATES+2]  #(batch, 1)
        # b_s_ = batch_memory[:, self.N_STATES+2:self.N_STATES*2+2] #(batch, state_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([b_s_, self.actor_target(b_s_).detach()])

        target_q_batch = b_r + self.GAMMA * next_q_values

        # Critic update
        self.critic.zero_grad()
        

        q_batch = self.critic([b_s, b_a])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
       
        self.actor.zero_grad()
        policy_loss = -self.critic([b_s, self.actor(b_s)])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # total_loss = value_loss + policy_loss
        # total_loss.backward()

        # self.critic_optim.step()
        # self.actor_optim.step()


        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        # print('target soft update')

        return policy_loss, value_loss

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))

        # replace the old memory with new memory

        # index = self.memory_counter % self.MEMORY_CAPACITY
        # self.memory[index, :] = transition

        self.memory = np.roll(self.memory, -1, axis=0)
        self.memory[-1,:] = transition

        self.memory_counter += 1

    def store_s(self, s):
        self.s_his.append(s)

    def save_s_his(self, filepath):
        np.save(filepath, np.array(self.s_his))

    def reset(self):
        self.actor.reset()
        self.critic.reset()


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3,
                 groups=4,  # choose number of groups
                 num_layers=(2, 2, 2),  # choose number of layers for each group
                 hidden_size=150,  # choose hidden size for all reservoirs
                 initializer=sparse_initializer(seed=seed),  # choose sparse as reservoir structure
                 regularization=0.5,
                 lr=[1.0, 0.3], sr=10.0,  # assign activation
                 washout=50, density=0.2,
                 teacher_scaling=None, teacher_shift=None, bias=True, n_reservoirs = 4, tag=0,
                 one_state=0
                 ):
        super(Actor, self).__init__()
        if len(lr) != len(num_layers) and len(lr) != 1:
            print("******Dimension mismatch between leaky rate and layers!*****\n")
        if len(lr) == 1:
            self.activation = self_normalizing_default(leaky_rate=lr[0], spectral_radius=sr)
            # self.activation = tanh(leaky_rate=lr[0])
        else:
            self.activation = [self_normalizing_default(leaky_rate=leaky, spectral_radius=sr) for leaky in lr]
            # self.activation = [tanh(leaky_rate=leaky) for leaky in lr]
        self.initializer = sparse_initializer(seed=seed, density=density, spectral_radius=1.0)
        self.tag = tag
        self.one_state = one_state
        print("\n ********Set cascaded reservoirs and concatenate inputs********\n")
        # self.esn = []
        # self.fcs = []
        hidden1 = groups * hidden_size
        fc_input_size = hidden1
        if self.tag == 1:
            fc_input_size = fc_input_size + nb_states
        input_size = nb_states
        if self.tag != 0:
            input_size = self.one_state
        self.esn = GroupedDeepESN2(
                        input_size=input_size,
                        output_dim=nb_actions,
                        groups=groups,  # choose number of groups
                        num_layers=num_layers,  # choose number of layers for each group
                        hidden_size=hidden_size,  # choose hidden size for all reservoirs
                        initializer=self.initializer,  # choose sparse as reservoir structure
                        regularization=regularization,
                        activation=self.activation,  # assign activation
                        washout=washout,
                        bias=bias
                        )
        
        # self.esn.append(GroupedDeepESN2(
        #                 input_size=nb_states,
        #                 output_dim=nb_actions,
        #                 groups=groups,  # choose number of groups
        #                 num_layers=num_layers,  # choose number of layers for each group
        #                 hidden_size=hidden_size,  # choose hidden size for all reservoirs
        #                 initializer=self.initializer,  # choose sparse as reservoir structure
        #                 regularization=regularization,
        #                 activation=self.activation,  # assign activation
        #                 washout=washout,
        #                 bias=bias
        #                 ))
        # self.fcs.append(nn.Linear(fc_input_size, hidden1))

        # for i in range(n_reservoirs-1):
        #     self.esn.append(GroupedDeepESN2(
        #                 input_size=hidden1,
        #                 output_dim=nb_actions,
        #                 groups=groups,  # choose number of groups
        #                 num_layers=num_layers,  # choose number of layers for each group
        #                 hidden_size=hidden_size,  # choose hidden size for all reservoirs
        #                 initializer=self.initializer,  # choose sparse as reservoir structure
        #                 regularization=regularization,
        #                 activation=self.activation,  # assign activation
        #                 washout=washout,
        #                 bias=bias
        #                 ))
        #     self.fcs.append(nn.Linear(fc_input_size, hidden1))
        
        self.fc1 = nn.Linear(nb_states, hidden1)
        if self.tag == 2:
            self.fc2 = nn.Linear(hidden1+nb_states, hidden2)
        else:
            self.fc2 = nn.Linear(hidden1, hidden2)

        if self.tag == 1:
            self.fc3 = nn.Linear(hidden2+nb_states, nb_actions)
        else:
            self.fc3 = nn.Linear(hidden2, nb_actions)
    
        self.fcone = nn.Linear(hidden1, nb_actions)

        # self.fc_final = nn.Linear(hidden1,nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights(init_w)
        self.para = []

        # for fc in self.fcs:
        #     self.para.extend(fc.parameters())

        self.para.extend(self.fc2.parameters())
        self.para.extend(self.fc3.parameters())


    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fcone.weight.data.uniform_(-init_w, init_w)

        # for fc in self.fcs:
        #     fc.weight.data = fanin_init(fc.weight.data.size())
        # self.fc_final.weight.data.uniform_(-init_w,init_w)

    def forward(self, x):
       
        if self.tag != 0:
            # print('x: ', x[:, -self.one_state:], '\n')
            # exit(1)
            out = self.esn(x[:, -self.one_state:])
        else:
            out = self.esn(x)
        
        if self.tag == 2:
            out = self.fc2(torch.cat([out, x], 1)) 
        else:
            out = self.fc2(out)

        out = self.relu(out)

        if self.tag ==  1:
           out = self.fc3(torch.cat([out, x], 1)) 
        else:
            out = self.fc3(out)
        out = self.tanh(out)

        # out = self.fcone(out)

        # if self.tag == 1:
        #     out = self.esn[0](x)
        #     out = self.fcs[0](torch.cat([out, x], 1))
        #     out = self.relu(out)
        #     for n in range(len(self.esn)-1):
        #         out = self.esn[n+1](out)
        #         out = self.fcs[n+1](torch.cat([out, x], 1))
        #         out = self.relu(out)

        #     out = self.fc_final(out)
        #     out = self.tanh(out)

        # else:    
        #     out = self.esn[0](x)
        #     out = self.fcs[0](out)
        #     out = self.relu(out)
        #     for n in range(len(self.esn)-1):
        #         out = self.esn[n+1](out)
        #         out = self.fcs[n+1](out)
        #         out = self.relu(out)

        #     out = self.fc_final(out)
        #     out = self.tanh(out)

        return out

    def reset(self):
        # for e in self.esn:
        #     e.reset_hidden()

        self.esn.reset_hidden()

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3,
                 groups=4,  # choose number of groups
                 num_layers=(2, 2, 2),  # choose number of layers for each group
                 hidden_size=150,  # choose hidden size for all reservoirs
                 initializer=sparse_initializer(seed=seed),  # choose sparse as reservoir structure
                 regularization=0.5,
                 lr=[1.0, 0.3], sr=10,  # assign activation
                 washout=50, density=0.2,
                 teacher_scaling=None, teacher_shift=None, bias=True, n_reservoirs=4, tag=0,
                 one_state=0
                 ):
        super(Critic, self).__init__()
        if len(lr) != len(num_layers) and len(lr) != 1:
            print("******Dimension mismatch between leaky rate and layers!*****\n")
        if len(lr) == 1:
            self.activation = self_normalizing_default(leaky_rate=lr[0], spectral_radius=sr)
            # self.activation = tanh(leaky_rate=lr[0])
        else:
            self.activation = [self_normalizing_default(leaky_rate=leaky, spectral_radius=sr) for leaky in lr]
            # self.activation = [tanh(leaky_rate=leaky) for leaky in lr]

        self.initializer = sparse_initializer(seed=seed, density=density,spectral_radius=1.0)
        hidden1 = groups * hidden_size

        # self.esn = []
        # self.fcs = []

        self.tag = tag
        self.one_state = one_state
        hidden1 = groups * hidden_size
        fc_input_size = hidden1
        if self.tag == 1:
            fc_input_size = fc_input_size + nb_actions + nb_states

        input_size = nb_states
        if self.tag != 0:
            input_size = self.one_state
        self.esn = GroupedDeepESN2(
                        input_size=input_size,
                        output_dim=nb_actions,
                        groups=groups,  # choose number of groups
                        num_layers=num_layers,  # choose number of layers for each group
                        hidden_size=hidden_size,  # choose hidden size for all reservoirs
                        initializer=self.initializer,  # choose sparse as reservoir structure
                        regularization=regularization,
                        activation=self.activation,  # assign activation
                        washout=washout,
                        bias=bias
                        )
        
        # self.esn.append(GroupedDeepESN2(
        #                 input_size=(nb_states+nb_actions),
        #                 output_dim=nb_actions,
        #                 groups=groups,  # choose number of groups
        #                 num_layers=num_layers,  # choose number of layers for each group
        #                 hidden_size=hidden_size,  # choose hidden size for all reservoirs
        #                 initializer=self.initializer,  # choose sparse as reservoir structure
        #                 regularization=regularization,
        #                 activation=self.activation,  # assign activation
        #                 washout=washout,
        #                 bias=bias
        #                 ))
        # self.fcs.append(nn.Linear(fc_input_size, hidden1))

        # for i in range(n_reservoirs-1):
        #     self.esn.append(GroupedDeepESN2(
        #                 input_size=hidden1,
        #                 output_dim=nb_actions,
        #                 groups=groups,  # choose number of groups
        #                 num_layers=num_layers,  # choose number of layers for each group
        #                 hidden_size=hidden_size,  # choose hidden size for all reservoirs
        #                 initializer=self.initializer,  # choose sparse as reservoir structure
        #                 regularization=regularization,
        #                 activation=self.activation,  # assign activation
        #                 washout=washout,
        #                 bias=bias
        #                 ))
        #     self.fcs.append(nn.Linear(fc_input_size, hidden1))

        # self.fc_final = nn.Linear(hidden1, 1)

        self.fc1 = nn.Linear(nb_states, hidden1)
        if self.tag == 2:
            self.fc2 = nn.Linear(hidden1 + nb_actions + nb_states, hidden2)
        else:
            self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        
        if self.tag == 1:
            self.fc3 = nn.Linear(hidden2+nb_states, 1)
        else:
            self.fc3 = nn.Linear(hidden2, 1)
        self.fcone = nn.Linear(hidden1 + nb_actions, 1)


        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

        self.para = []

        # for fc in self.fcs:
        #     self.para.extend(fc.parameters())

        self.para.extend(self.fc2.parameters())
        self.para.extend(self.fc3.parameters())

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fcone.weight.data.uniform_(-init_w, init_w)

        # for fc in self.fcs:
        #     fc.weight.data = fanin_init(fc.weight.data.size())
        # self.fc_final.weight.data.uniform_(-init_w,init_w)




    def forward(self, xs):
        x, a = xs
        if self.tag != 0:
            out = self.esn(x[:, -self.one_state:])
        else:
            out = self.esn(x)

        if self.tag == 2:
            out = self.fc2(torch.cat([out, x, a], 1))
        else:  
            out = self.fc2(torch.cat([out, a], 1))

        out = self.relu(out)
        if self.tag == 1:
            out = self.fc3(torch.cat([out, x], 1))
        else:   
            out = self.fc3(out)

        # out = self.fcone(torch.cat([out, a], 1))

        # if self.tag == 1:
        #     out = self.esn[0](torch.cat([x, a], 1))
        #     out = self.fcs[0](torch.cat([out, x, a], 1))
        #     out = self.relu(out)
        #     for n in range(len(self.esn)-1):
        #         out = self.esn[n+1](out)
        #         out = self.fcs[n+1](torch.cat([out, x, a], 1))
        #         out = self.relu(out)

        #     out = self.fc_final(out)

        # else:
        #     out = self.esn[0](torch.cat([x, a], 1))
        #     out = self.fcs[0](out)
        #     out = self.relu(out)
        #     for n in range(len(self.esn)-1):
        #         out = self.esn[n+1](out)
        #         out = self.fcs[n+1](out)
        #         out = self.relu(out)

        #     out = self.fc_final(out)

        return out

    def reset(self):
        # for e in self.esn:
            # e.reset_hidden()        

        self.esn.reset_hidden()

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
