import numpy as np
import torch
import math

# from utils.utils import quaternion2euler

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else: 
                self[key] = eval(value)
    def __getattr__(self,val):
        return self[val]
    
def make_transition(state,action,reward,next_state,done,log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition

def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
        yield [x[indices] for x in value[1:]]
        
def convert_to_tensor(*value):
    device = value[0]
    return [torch.tensor(x).float().to(device) for x in value[1:]]

class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.action_prob_exist = action_prob_exist
        self.data = {}
        
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        if self.action_prob_exist :
            self.data['log_prob'] = np.zeros((self.max_size, 1))
    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        self.data['state'][idx] = transition['state']
        self.data['action'][idx] = transition['action']
        self.data['reward'][idx] = transition['reward']
        self.data['next_state'][idx] = transition['next_state']
        self.data['done'][idx] = float(transition['done'])
        if self.action_prob_exist :
            self.data['log_prob'][idx] = transition['log_prob']
        
        self.data_idx += 1
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist :
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

# Equalvariance Section:

# Quaternion to Euler Coordiantion 
def quaternion2euler(x,y,z,w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2 * (x * x + y * y)
    roll_x = math.atan2(t0,t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 <-1.0 else t2 
    pitch_y = math.asin(t2)
    
    t3 = 2.0 * (w*z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3,t4)
    
    return roll_x, pitch_y, yaw_z

# Transfer the state to the form of equalvariance

# Normal Equalvariant 
'''
def state2equistate(state):
    state_ = state[:27]
    state_transfer = []
    x_eu,y_eu,z_eu = quaternion2euler(state_[1],state_[2],state_[3],state_[4])
    x_coorv, y_coorv, z_coorv = state_[13], state_[14], state_[15]
    x_anguv, y_anguv, z_anguv = state_[16], state_[17], state_[18]
    # common_lst = [x_eu,y_eu,z_eu,x_coorv,y_coorv,z_coorv,x_anguv,y_anguv,z_anguv]
    for i in range(5,12,2):
        state_transfer.append(state_[i])
    for i in range(6,13,2):
        state_transfer.append(state_[i])
    for i in range(19,26,2):
        state_transfer.append(state_[i])
    for i in range(20,27,2):
        state_transfer.append(state_[i])
    # Add normal elements
    state_transfer += list(x_eu * np.array([-1,1,-1,1]))
    state_transfer += list(y_eu * np.array([1,1,-1,-1]))
    state_transfer += list(z_eu * np.array([1,1,1,1]))
    state_transfer += list(x_coorv * np.array([-1,1,-1,1]))
    state_transfer += list(y_coorv * np.array([1,1,-1,-1]))
    state_transfer += list(z_coorv * np.array([1,1,1,1]))
    state_transfer += list(x_anguv * np.array([-1,1,-1,1]))
    state_transfer += list(y_anguv * np.array([1,1,-1,-1]))
    state_transfer += list(z_anguv * np.array([1,1,1,1]))
    #state_transfer += [state_[0]] * 4 
    return np.array(state_transfer,dtype=np.double)
'''





def state2equistate_single(state): 
    state_ = state.cpu().detach().numpy()[:27]
    state_transfer = []
    # x_eu,y_eu,z_eu = quaternion2euler(state_[1],state_[2],state_[3],state_[4])
    x_quar,y_quar,z_quar,w_quar = state_[1], state_[2], state_[3], state_[4]
    x_coorv, y_coorv, z_coorv = state_[13], state_[14], state_[15]
    x_anguv, y_anguv, z_anguv = state_[16], state_[17], state_[18]
    common_lst = [x_quar,y_quar,z_quar,w_quar,x_coorv,y_coorv,z_coorv]

    idx_lst = [5,6,19,20,7,8,21,22]
    for idx in idx_lst:
        state_transfer.append(state_[idx])
        state_transfer.append(state_[idx+4])
    for item in common_lst:
        state_transfer += [item]*2
    state_transfer = torch.from_numpy(np.array(state_transfer)).to(torch.float32)
    return state_transfer




# EqualVariant with front and back

def state2equistate(state):
    state_stack = torch.empty((len(state),30), dtype=torch.float32)
    for i in range(len(state)):
        state_ = state[i].cpu().detach().numpy()[:27]
        state_transfer = []
        # x_eu,y_eu,z_eu = quaternion2euler(state_[1],state_[2],state_[3],state_[4])
        x_quar,y_quar,z_quar,w_quar = state_[1], state_[2], state_[3], state_[4]
        x_coorv, y_coorv, z_coorv = state_[13], state_[14], state_[15]
        x_anguv, y_anguv, z_anguv = state_[16], state_[17], state_[18]
        common_lst = [x_quar,y_quar,z_quar,w_quar,x_coorv,y_coorv,z_coorv]

        idx_lst = [5,6,19,20,7,8,21,22]
        for idx in idx_lst:
            state_transfer.append(state_[idx])
            state_transfer.append(state_[idx+4])
        for item in common_lst:
            state_transfer += [item]*2
        state_transfer = torch.from_numpy(np.array(state_transfer)).to(torch.float32)
        state_stack[i] = state_transfer
    return state_stack

'''
def state2equistate_batch(state):
    state_stack = torch.empty((len(state),30), dtype=torch.float32)
    for i in range(len(state)):
        state_ = state[i].detach().numpy()[:27]
        state_transfer = []
        # x_eu,y_eu,z_eu = quaternion2euler(state_[1],state_[2],state_[3],state_[4])
        x_quar,y_quar,z_quar,w_quar = state_[1], state_[2], state_[3], state_[4]
        x_coorv, y_coorv, z_coorv = state_[13], state_[14], state_[15]
        x_anguv, y_anguv, z_anguv = state_[16], state_[17], state_[18]
        common_lst = [x_quar,y_quar,z_quar,w_quar,x_coorv,y_coorv,z_coorv]

        idx_lst = [5,6,19,20,7,8,21,22]
        for idx in idx_lst:
            state_transfer.append(state_[idx])
            state_transfer.append(state_[idx+4])
        for item in common_lst:
            state_transfer += [item]*2
        state_transfer = torch.from_numpy(np.array(state_transfer)).to(torch.float32)
        state_stack[i] = state_transfer
    return state_stack

'''



'''
def state2equistate(state):
    state_ = state[:27]
    state_transfer = []
    # x_eu,y_eu,z_eu = quaternion2euler(state_[1],state_[2],state_[3],state_[4])
    x_quar,y_quar,z_quar,w_quar = state_[1], state_[2], state_[3], state_[4]
    x_coorv, y_coorv, z_coorv = state_[13], state_[14], state_[15]
    x_anguv, y_anguv, z_anguv = state_[16], state_[17], state_[18]
    common_lst = [x_quar,y_quar,z_quar,w_quar,x_coorv,y_coorv,z_coorv]
    
    idx_lst = [5,6,9,10,19,20,23,24]
    for idx in idx_lst:
        state_transfer.append(state_[idx])
        state_transfer.append(state_[idx+2])
    for item in common_lst:
        state_transfer += [item]*2
    return np.array(state_transfer)
'''


def StateAction2EqualSA(state, action,device):
    # Concatenate the two tensors together 
    state_ = state2equistate(state)
    state_ = state_.to(device)
    action_ = mu_map(action)
    action_ = action_.to(device)
    state_action = torch.cat((state_,action_),dim=1).to(torch.float32)
    state_action = state_action.to(device)
    return state_action






'''
def mu_map(mu_):
    mu = mu_.detach().numpy()
    A = []
    for i in range(int(len(mu)/2)):
        A.append(mu[i])
        A.append(mu[i+4])
    mu = torch.from_numpy(np.array(A)).to(torch.float32)
    return mu 
'''

# Equalvariant between front and back 

def mu_map(mu_):
    mu = torch.empty_like(mu_, dtype=torch.float32)
    for i in range(len(mu_)):
        for j in range(0,len(mu_[0]),2):
            mu[i][int(j/2)] = mu_[i][j]
        for k in range(1,len(mu_[0])+1,2):
            mu[i][int((k-1)/2)+4] = mu_[i][k]
    return mu  

def mu_map_single(mu_):
    mu = torch.empty_like(mu_,dtype=torch.float32)
    for j in range(0,len(mu_),2):
        mu[int(j/2)] = mu_[j]
    for k in range(1,len(mu_)+1,2):
        mu[int((k-1)/2)+4] = mu_[k]
    
    return mu






'''
# Equalvariant between left and right
def mu_map(mu_):
    mu = mu_.detach().numpy()
    temp = mu[1]
    mu[1] = mu[2]
    mu[2] = temp

    temp = mu[5]
    mu[5] = mu[6]
    mu[6] = temp 
    
    chance = np.random.rand()
    if chance < 0.5:
        mu = mu * (-1)
    mu = torch.from_numpy(mu).to(torch.float32)
    return mu 
'''
