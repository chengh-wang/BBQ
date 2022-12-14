from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader

from utils.utils import state2equistate, state2equistate_single , StateAction2EqualSA, mu_map, mu_map_single

class SAC(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(SAC,self).__init__()
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std)
        self.actor = self.actor.model.to(device)

        self.q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
        self.q_1 = self.q_1.model.to(device)
        self.q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation)
        self.q_2 = self.q_2.model.to(device)
        
        self.target_q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        self.target_q_1 = self.target_q_1.model.to(device)
        self.target_q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        self.target_q_2 = self.target_q_2.model.to(device)


        self.soft_update(self.q_1, self.target_q_1, 1.)
        self.soft_update(self.q_2, self.target_q_2, 1.)
        
        self.alpha = nn.Parameter(torch.tensor(self.args.alpha_init))
        
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
        self.target_entropy = - torch.tensor(action_dim)

        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=self.args.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.args.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.args.alpha_lr)
        
        self.device = device
        self.writer = writer
        
    def put_data(self,transition):  
        self.data.put_data(transition)
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
    
    def get_action_single(self,state):

        state = state2equistate_single(state)
        state = state.to(self.device)
        # Start getting action
        mu_ = self.actor.forward(state)
        mu_ = mu_.to(self.device)
        mu = mu_map_single(mu_)
        mu = mu.to(self.device)
        # logstd = self.actor.parameters(torch.zeros(1, 8))
        logstd = nn.Parameter(torch.zeros(1, 8))
        std = torch.exp(logstd)
        std = std.to(self.device)
        # logstd = torch.zeros_like(mu)
        # std = torch.exp(logstd)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a = a.to(self.device)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) +1e-3)
        a_log_prob = a_log_prob.to(self.device)
        return a, a_log_prob.sum(-1, keepdim=True)



    def get_action(self,state):
        # print(f'State Before transformation:\n {state}, \nState shape: {state.shape}')
        # print(f"State: {state}. \nSize: {len(state)}")
        state = state2equistate(state).to(self.device)
        # print(f'State:\n {state},\nState Shape: {state.shape}')
        '''
        # # # Normalization 
        state = state.view(-1,1)
        state = torch.nn.functional.normalize(state,dim=0)
        state = state.view(-1)
        '''
        
        # Start getting action
        mu_ = self.actor.forward(state)
        mu_ = mu_.to(self.device)
        mu = mu_map(mu_)
        mu = mu.to(self.device)
        # logstd = self.actor.parameters(torch.zeros(1, 8))
        logstd = nn.Parameter(torch.zeros(1, 8))
        logstd = logstd.to(self.device)
        std = torch.exp(logstd)
        std.to(self.device)
        # logstd = torch.zeros_like(mu)
        # std = torch.exp(logstd)
        dist = Normal(mu, std)
        u = dist.rsample()
        u = u.to(self.device)
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a = a.to(self.device)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) +1e-3)
        a_log_prob = a_log_prob.to(self.device)
        return a, a_log_prob.sum(-1, keepdim=True)
    
    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, dones):
        ###target
        with torch.no_grad():
            next_states = next_states.to(self.device)
            next_actions, next_action_log_prob = self.get_action(next_states)
            next_actions = next_actions.to(self.device)
            # next_actions = next_actions.detach().numpy()
            next_s_a = StateAction2EqualSA(next_states,next_actions,self.device)
            next_s_a = next_s_a.to(self.device)
            # q_1 = self.target_q_1(next_states,next_actions)
            q_1 = self.target_q_1.forward(next_s_a)
            q_1 = q_1.to(self.device)
            # q_2 = self.target_q_2(next_states,next_actions)
            q_2 = self.target_q_2.forward(next_s_a)
            q_2 = q_2.to(self.device)
            q = torch.min(q_1,q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)
            targets = rewards + self.args.gamma * v
        
        # actions = actions.detach().numpy()
        now_s_a = StateAction2EqualSA(states,actions,self.device)
        now_s_a = now_s_a.to(self.device)
        # q = Q(states, actions)
        q = Q.forward(now_s_a)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        return loss
    
    def actor_update(self, states):
        states = states.to(self.device)
        now_actions, now_action_log_prob = self.get_action(states)
        now_actions = now_actions.to(self.device)
        now_action_log_prob = now_action_log_prob.to(self.device)
        # now_actions = now_actions.detach().numpy()[0]
        now_s_a = StateAction2EqualSA(states, now_actions,self.device)
        now_s_a = now_s_a.to(self.device)
        # q_1 = self.q_1(states, now_actions)
        # q_2 = self.q_2(states, now_actions)
        q_1 = self.q_1.forward(now_s_a)
        q_2 = self.q_2.forward(now_s_a)
        q = torch.min(q_1, q_2)
        
        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss,now_action_log_prob
    
    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()    
        loss.backward()
        self.alpha_optimizer.step()
        return loss
    
    def train_net(self, batch_size, n_epi):
        # print(f"Training Batch size:{batch_size}. \nType: {type(batch_size)}")
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        # print(f"States:\n {states.view(-1)}")
        # print(f"Actions:\n {actions.view(-1)}")
        # print(f"Rewards:\n {rewards}")
        # print(f"Next States:\n {next_states}")
        # print(f"Dones:\n {dones}")
        q_1_loss = self.q_update(self.q_1, self.q_1_optimizer, states, actions,rewards, next_states, dones)
        q_2_loss = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, dones)
        
        ### actor updaten
        actor_loss,prob = self.actor_update(states)
        
        ###alpha update
        alpha_loss = self.alpha_update(prob)
        
        self.soft_update(self.q_1, self.target_q_1, self.args.soft_update_rate)
        self.soft_update(self.q_2, self.target_q_2, self.args.soft_update_rate)
        if self.writer != None:
            self.writer.add_scalar("loss/q_1", q_1_loss, n_epi)
            self.writer.add_scalar("loss/q_2", q_2_loss, n_epi)
            self.writer.add_scalar("loss/actor", actor_loss, n_epi)
            self.writer.add_scalar("loss/alpha", alpha_loss, n_epi)
        
        
