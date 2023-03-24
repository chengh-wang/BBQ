from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'Ant-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'sac', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)

#args.render = True
# args.tensorboard = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'


    
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

 
env = gym.make(args.env_name)
# env = gym.make('Ant-v3',reset_noise_scale = 0.8)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)


if args.algo == 'ppo' :
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac' :
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'ddpg' :
    from utils.noise import OUNoise
    noise = OUNoise(action_dim,0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)
'''
elif args.algo == "equal_sac":
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
    state_dim = 52
    state_rms = RunningMeanStd(state_dim)
'''
    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
    
score_lst = []
state_lst = []
if agent_args.on_policy == True:
    score = 0.0
    state_ = (env.reset())
    
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    #print(f"State: {state_}")
    #print(f"State Dim :{len(state_)}.")
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            if args.render:    
                env.render()
            state_lst.append(state_)
            mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu,sigma[0])
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1,keepdim = True)
            next_state_, reward, done, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state,\
                                         action.cpu().numpy(),\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += reward
            if done:
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_

        agent.train_net(n_epi)
        state_rms.update(np.vstack(state_lst))
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
            
else : # off policy 
    print("Start Off Policy------------------------------------------------------")
    for n_epi in range(args.epochs):
        score = 0.0
        state = env.reset()
        done = False
        while not done :
            if args.render:    
                env.render()
            # print(f'State Before Getting into the method:\n {state}, \nState Size: {state.shape}')
            action, _ = agent.get_action_single(torch.from_numpy(state).float().to(device))
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action[0])
            # reward = (state[0] - 0.3)*50 - (state[13] + state[14] + state[15])*10
            transition = make_transition(state,\
                                        action,\
                                        #np.array([reward*args.reward_scaling]),\
                                        np.array(reward),\
                                        next_state,\
                                        np.array([done])\
                                        )
            agent.put_data(transition) 
            state = next_state
            # print(f'Next State is:\n {state}')
            # print(f"Is done ? : {done}")
            score += reward
            if agent.data.data_idx > agent_args.learn_start_size: 
                # print("Yeah !!!!!!!!!!!!!!!!!")
                print("Start Training!!!")
                print("device",device)
                agent.train_net(agent_args.batch_size, n_epi)
                print("Training ends!!!")
        score_lst.append(score)
        # print("The program continues!!")
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
