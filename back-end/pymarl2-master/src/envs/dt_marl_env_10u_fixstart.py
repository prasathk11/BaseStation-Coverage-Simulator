import logging
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
import math
import random
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from envs.multiagentenv import MultiAgentEnv
import time

logger = logging.getLogger(__name__)

MIN_X, MAX_X = -150, 150
MIN_Y, MAX_Y = -50, 50
DIRECTIONS = ['up', 'down', 'left', 'right']
DELTA_L = 1

start_points = np.array([[[-125.4306251 ,   20.51331404],
        [ 137.81382638,   48.29665261],
        [ 115.59449927,   -4.7794654 ],
        [-118.01007245,  -13.07478183],
        [ 105.50412309,  -36.84555974],
        [   7.15893811,  -12.51119844],
        [  68.98712553,   39.23053367],
        [ 139.84563543,  -37.60172574],
        [-100.90155901,  -21.27687776],
        [  75.89906037,  -19.31872153]],

       [[  87.39660584,   23.43948164],
        [  10.95856567,  -43.90026933],
        [  23.38490688,   -4.38450438],
        [ 124.13152905,  -31.84381023],
        [-131.17283987,   10.77007298],
        [ -13.65573348,   12.55804706],
        [ -35.29468658,  -45.5123177 ],
        [ -74.19270604,  -32.09632332],
        [-118.73326196,  -39.68581228],
        [  51.92133022,  -24.96109594]],

       [[  29.70997871,  -38.65625988],
        [-134.77829624,    0.75210267],
        [ -34.92838375,    0.74856612],
        [ 119.77181614,  -40.00953488],
        [ -54.28051115,  -16.44757654],
        [ -10.47875171,  -14.36965909],
        [ -91.71037613,   46.19254356],
        [ -86.90029333,  -20.64918119],
        [  40.23988442,   -6.64759506],
        [-130.80867773,   15.3958494 ]],

       [[  35.9642058 ,  -32.43346502],
        [ 120.49292274,    6.71067312],
        [ -31.16537732,    3.41382676],
        [  69.57683389,  -47.23599669],
        [  92.69455595,  -27.2687079 ],
        [-141.43637589,   -3.32848443],
        [  -9.71838991,   16.26103207],
        [  82.70717253,   -6.9231642 ],
        [ -12.33378111,   44.28655756],
        [ -12.43627525,   39.9616957 ]],

       [[  11.12038522,   -0.29805978],
        [ -81.40221797,   33.30226946],
        [ -24.31795866,   28.85623744],
        [ 142.00059746,  -13.42934081],
        [-117.66647417,  -45.53805524],
        [  21.09608849,   31.00958944],
        [  62.92717287,   30.98007883],
        [ 144.46879323,  -31.96130199],
        [  84.06179742,   11.73451734],
        [ 145.95357394,   44.72478234]],

       [[   0.76597334,  -21.49945129],
        [ -22.78172745,   11.24587176],
        [ -85.62205491,   37.15478479],
        [  91.21878218,  -18.04947949],
        [ 149.09213996,   36.50866434],
        [ 144.25496823,   43.27208057],
        [ -32.97252396,   -1.24576835],
        [  -1.10236635,   47.46533945],
        [-101.70693843,   -6.54355892],
        [ -34.53271211,    1.02824548]],

       [[  -2.80610992,  -37.0010549 ],
        [ -97.26894609,   -6.42198889],
        [ -40.82709945,   28.80476297],
        [  71.56013575,  -49.87816343],
        [ -11.65848375,   25.7400842 ],
        [ -17.21268725,   16.64685558],
        [  40.79528859,  -28.41654267],
        [  58.49391121,   43.46752958],
        [ -64.26847137,  -38.7128141 ],
        [ -69.51857925,   48.26006922]],

       [[-121.86184733,  -46.68356087],
        [ -29.3840182 ,  -13.21564513],
        [ -60.51070987,   12.33208848],
        [   4.63927675,   41.55463576],
        [-142.73455065,   21.56542269],
        [-147.65375195,   12.66993727],
        [ 147.98812127,  -33.9179857 ],
        [  48.23822333,   -9.56741119],
        [  30.60932419,   47.37016052],
        [ -52.4768079 ,    1.0633288 ]],

       [[ 108.93419848,  -37.667768  ],
        [  55.71822591,    2.26911984],
        [  12.15960339,   41.14671449],
        [-117.84054256,   38.60420699],
        [  69.05916855,  -18.84464062],
        [ 129.15138939,   40.88974729],
        [ -46.10461368,  -48.02623023],
        [ -17.64666473,  -37.74776052],
        [ -20.66863976,   -5.98565068],
        [ -63.49404695,   42.90015415]],

       [[ -76.13477109,  -18.96786039],
        [  17.27206682,   -6.38521792],
        [ -15.00362225,   45.34090464],
        [ 142.09258268,   38.77272966],
        [ -60.84062354,   37.54793127],
        [ -83.50187164,   39.30457221],
        [ -26.07585709,  -16.72514429],
        [ 138.74381775,    2.99207306],
        [ -92.59808022,   45.85444756],
        [  24.25905531,   25.84693164]]])

# GRU model
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=20, 
            hidden_size=128, 
            batch_first=True, 
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 64), 
            nn.LeakyReLU(), 
            nn.Linear(64, 20) 
            # nn.LeakyReLU(), 
            # nn.Linear(16, 12) 
        )

    def forward(self, input):
        output, h_n = self.gru(input, None)
        # print(output.shape)
        output = output[:, -1, :]
        output = self.mlp(output)
        return output
    
gru_model = GRU()
gru_model.load_state_dict(torch.load('/home/hanzhi/Documents/dt_rb_allocation/pymarl2-master_epsilon/gru_10u_norm.pt'))

def ans_calculation(mat, rb_index, user_index):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    rb_allocation = {}
    for i in range(len(rb_index)):
        total += mat[rb_index[i], user_index[i]]
        ans_mat[rb_index[i], user_index[i]] = mat[rb_index[i], user_index[i]]
        rb_allocation[rb_index[i]] = user_index[i]
    return total, ans_mat

# The Class of the physical network & DT network
# 6 users, 4 RBs, and 1 cloud server are included
class Setting():
    def __init__(self, model):
        self.BS_loc = np.array([[-100,0], [0,0], [100,0]])
        self.num_BS = len(self.BS_loc)
        self.BS_coverage = 60
        self.usr_mov_prob = [
            (0.85, 0, 0, 0.15), 
            (0.85, 0, 0.15, 0), 
            (0, 0, 1, 0),   
            (0, 1, 0, 0), 
            (1, 0, 0, 0), 
            (0, 0, 0, 1), 
            (0, 0.85, 0.15, 0), 
            (0, 0.85, 0, 0.15), 
            (0.15, 0, 0.85, 0), 
            (0, 0.15, 0.85, 0), 
            # (0.15, 0, 0, 0.85), 
            # (0, 0.15, 0, 0.85), 
            # (0.7, 0, 0.15, 0.15), 
            # (0.7, 0.15, 0, 0.15), 
            # (0.7, 0.15, 0.15, 0)
        ]
        self.num_ue = len(self.usr_mov_prob)

        self.fixstart = start_points[model]
        self.UserLocation = np.copy(self.fixstart)
        # self.UserLocation = np.array([[random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)] for _ in range(self.num_ue)])

        self.routines = np.array(self.UserLocation.reshape(1,2*self.num_ue))
        self.CloudSeverLocation = np.array([0,50],dtype=np.float32)
        self.RbInfo = [0.017, 0.021, 0.025, 0.009]   # 按需自定义设置rb的相关信息，这里是I_n
        self.RbAllocation = np.zeros((self.num_BS, len(self.RbInfo))) #记录rb分配状态，可以用分配完了当作任务结束的标志
        self.bandwidth = 1 # 带宽
        self.p_t = 1  # 发射功率
        self.alpha = 2 # 路损因子
        self.sigma = 0.00001 # 噪声的单边功率谱密度
        self.o = [np.random.rayleigh(scale=1) for _ in range(self.num_ue+1)]

    def random_move(self): 
        for u in range(self.num_ue): 
            x_current = self.UserLocation[u][0]
            y_current = self.UserLocation[u][1]
            move = random.choices(DIRECTIONS, weights=self.usr_mov_prob[u])[0]
        
            if move == 'up': 
            #and y_current + DELTA_L <= MAX_Y:
                y_current += DELTA_L
            elif move == 'down': 
            # and y_current - DELTA_L >= MIN_Y: 
                y_current -= DELTA_L
            elif move == 'left': 
            # and x_current - DELTA_L >= MIN_X:
                x_current -= DELTA_L
            elif move == 'right': 
            # and x_current + DELTA_L <= MAX_X:
                x_current += DELTA_L

            self.UserLocation[u] = np.array([x_current, y_current])
        
        self.routines = np.vstack([self.routines, self.UserLocation.reshape(1,2*self.num_ue)])
        return self.UserLocation

    def reset_step(self): 
        # 每次分配完成后，用户移动，再重新进行下一轮分配
        self.RbAllocation = np.zeros((self.num_BS, len(self.RbInfo)))
        self.random_move() 
        self.o = [np.random.rayleigh(scale=1) for _ in range(self.num_ue+1)]

    def reset_epi(self): 
        # 每个episode长度为15，在移动15次后，用户随机选择一个新的起始位置
        self.RbAllocation = np.zeros((self.num_BS, len(self.RbInfo)))

        self.UserLocation = np.copy(self.fixstart)
        # self.UserLocation = np.array([[random.uniform(MIN_X, MAX_X), random.uniform(MIN_Y, MAX_Y)] for _ in range(self.num_ue)])

        self.routines = np.array(self.UserLocation.reshape(1,2*self.num_ue))
        self.o = [np.random.rayleigh(scale=1) for _ in range(self.num_ue+1)]


    def reward_matrix(self, BS_id, associate_ues, user_loc, rb_allocation_list):  
        # print(f"len(self.RbAllocation): {rb_allocation_list}")
        # print(f"len(associate_ues): {associate_ues}")
        # 计算用户在当下位置的reward matrix用于Hungarian algorithm
        reward_matrix = np.zeros((len(self.RbAllocation[BS_id]), len(associate_ues)), dtype=np.float32)
        for rb_index in range(len(self.RbAllocation[BS_id])): 
            for row, user_id in enumerate(associate_ues): 
                # print(reward_matrix)
                signal_power = self.p_t*self.o[user_id] / np.sum((np.array(self.BS_loc[BS_id])-np.array(user_loc[user_id]))**2)

                int_power = 0
                for agent_id in range(BS_id): 
                    int_source = rb_allocation_list[agent_id][rb_index]
                    if int_source != 0: 
                        int_power += self.p_t*self.o[user_id] / np.sum((np.array(self.BS_loc[agent_id])-np.array(user_loc[user_id]))**2)
                
                noise_power = self.bandwidth * self.sigma

                reward_matrix[rb_index][row] = self.bandwidth * math.log(1+signal_power/(int_power+noise_power), 2)
                # print(reward_matrix)
        # 对reward matrix进行normalization，不然后面data rate相关的reward会因为位置的不同差异很大
        # normed_reward_matrix = (reward_matrix - np.min(reward_matrix)) / (np.max(reward_matrix) - np.min(reward_matrix))
        return reward_matrix
        
# Environment class
class DT_MARL_10u_fix(MultiAgentEnv):

    def __init__(self, epsilon, model, seed):
        # 基本设置
        self.epsilon = epsilon
        self.setting = Setting(model) # 实例化资源分配对象
        self.n_agents = self.setting.num_BS
        self.usr_dist = self.usr_distribution(self.setting.UserLocation)
        self.agent_id = [i for i in range(self.n_agents)]
        self.observation_space = gym.spaces.Box(low=-150, high=150, shape=(self.setting.num_ue,2), dtype=np.float32) 
        self.action_space = gym.spaces.MultiBinary(self.setting.num_ue+1) # Action: 1 - allocate an RB to the information trans, 0 - otherwise
        self.n_actions = self.get_total_actions()
        self.n_rbs = len(self.setting.RbInfo)

        self.gamma = 1  # 折扣因子 ，资源分配里不区分先后分配的重要性区别，设为1就行。直接不设也可以。

        # self.state = self.get_obs()
        # self.dt_state = self.setting.UserLocation # DT network的state。action=1时为PN的state，action=0时为GRU的预测值
        self.dt_routine = np.array(self.setting.UserLocation.reshape(1,2*self.setting.num_ue)) # 记录DT的所有state，用于GRU预测
        # self.dt = twin_Setting()
        # self.reward = 0 
        self.reward = np.zeros(self.n_agents)
        self.episode_limit = 30
        self.T = 0
        self.terminated = False
        self.stepInfo = {}
    
    def get_obs(self): 
        obs = []
        for bs_id in range(self.n_agents): 
            if_usr_in_range = self.usr_dist[bs_id]
            partial_obs = np.where(if_usr_in_range[:, np.newaxis], np.array(self.setting.UserLocation), np.array([0, 0]))
            partial_obs = partial_obs.reshape((2*self.setting.num_ue,))
            norm_obs = partial_obs / 10
            obs.append(norm_obs)
        return obs
    
    def get_obs_agent(self, agent_id): 
        if_usr_in_range = self.usr_dist[agent_id]
        partial_obs = np.where(if_usr_in_range[:, np.newaxis], np.array(self.setting.UserLocation), np.array([0, 0]))
        partial_obs = partial_obs.reshape((2*self.setting.num_ue,))
        norm_obs = partial_obs / 10
        return norm_obs
    
    def get_obs_size(self): 
        # Should be the size of the obs list, which should pass to initialize the controller
        # Based on our use case, we use this function to return the size of the local obs, since the 
        # structure (input_shape) of the q network is changed. 
        return 2*self.setting.num_ue
    
    def get_state(self):
        return self.setting.UserLocation.reshape((2*self.setting.num_ue,))

    def get_state_size(self):
        return 2*self.setting.num_ue

    def get_avail_actions(self): 
        # Return a bool np_array in the shape of (num_bs, (num_ue+1))
        # True: the UE or the central server can be served by the BS
        avail_actions = []
        for agent_id in range(self.n_agents): 
            avail_actions_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_actions_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        avail_usr_conn = np.array(self.usr_dist[agent_id])
        invalid_usr = np.where(avail_usr_conn == False)[0]

        avail_actions = np.ones(self.n_actions)
        for action in range(self.n_actions): 
            action_bits = format(action, "011b")
            # Connect to the users which are not in the range
            if any(action_bits[i_u] == '1' for i_u in invalid_usr): 
                avail_actions[action] = 0
            # The number of the connections is more than the number of available RBs
            if (np.sum([int(b) for b in action_bits]) > self.n_rbs): 
                avail_actions[action] = 0

        return avail_actions
    
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        n_actions= 2 ** np.prod(self.action_space.shape)
        return n_actions

    def usr_distribution(self, user_loc): 
        usr_dist = []
        for bs_id in range(self.n_agents): 
            bs_loc = self.setting.BS_loc[bs_id]
            # print(f"Base station location: {np.array(bs_loc)}")
            # print(f"User locations: \n{np.array(user_loc)}")
            if_usr_in_range = np.abs(np.array(user_loc) - np.array(bs_loc))
            # print(f"User distances to base station: {if_usr_in_range}")
            if_usr_in_range = np.all(if_usr_in_range <= self.setting.BS_coverage, axis=1)
            # print(f"Users in range: {if_usr_in_range}")
            usr_dist.append(if_usr_in_range)
        return usr_dist
        
   
    def seed(self, seed=None):  # 这个不管就行
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions): # 必须实现的函数，输入动作，输出下一步的状态、奖励、结束与否、其他可选info
        # self.reward = 0
        self.reward = np.zeros(self.n_agents)
        # print(actions)

        # 需要5个历史数据来预测。如果不够，就把当前记录的数据重复几次，凑成5个
        dt_routine = np.array(self.dt_routine).reshape(len(self.dt_routine), -1)
        if len(self.dt_routine) < 5: 
            repeat_time = 5 - len(dt_routine)
            repeat_row = np.tile(dt_routine[0], (repeat_time,1))
            # print(repeat_row)
            input_rout = np.vstack((repeat_row, dt_routine))
            # print(input_rout)
            # Normalization
            input_rout /= 10
            input_rout = torch.from_numpy(input_rout).float()
            input_rout = input_rout.unsqueeze(0)
            est_st = gru_model(input_rout)
        else: 
            input_rout = dt_routine[-5:]
            # Normalization 
            input_rout /= 10
            input_rout = torch.tensor(input_rout, dtype=torch.float32)
            input_rout = input_rout.unsqueeze(0)
            est_st = gru_model(input_rout)

        est_st = est_st.detach().numpy()
        est_st = est_st.reshape((self.setting.num_ue,2))
        est_loc = 10 * est_st
        # if similarity > 1: 
        #     reward_syn = -5
        # else: 
        #     reward_syn = 0
        # print('reward_syn:', reward_syn)

        actions_vec = []
        actions_str = [format(action, "011b") for action in actions]
        for action_str in actions_str: 
            action_vec = [int(b) for b in action_str]
            actions_vec.append(action_vec)

        associate_users = [[] for _ in range(self.n_agents)]
        agent_syn = []

        # Regarding actions, update the DT state
        # print(self.setting.UserLocation, self.usr_dist)
        for agent_id in range(self.n_agents): 
            action = actions_vec[agent_id]
            # print(action)
            if action[-1] == 1: 
                agent_syn.append(agent_id)
                obs = self.get_obs_agent(agent_id).reshape((self.setting.num_ue,2))
                in_range_users = [user_id for user_id, if_in_range in enumerate(self.usr_dist[agent_id]) if if_in_range]
                for user_id in in_range_users: 
                    est_loc[user_id] = 10 * obs[user_id]
                # self.reward[agent_id] += 0
        # print(agent_syn)
        acc = np.mean((np.array(est_loc) - np.array(self.setting.UserLocation)) ** 2) 
        
        for agent_id in range(self.n_agents): 
            if actions_vec[agent_id][-1] == 0: 
                bs_loc = self.setting.BS_loc[agent_id]
                if_usr_in_range = abs(np.array(est_loc) - np.array(bs_loc))
                if_usr_in_range = np.all(if_usr_in_range <= self.setting.BS_coverage, axis=1)
                dt_obs = np.where(if_usr_in_range[:, np.newaxis], np.array(est_loc), np.array([0, 0]))
                dt_obs = dt_obs.reshape((2*self.setting.num_ue,))
                dt_obs = dt_obs / 10

                obs = self.get_obs_agent(agent_id)

                self.reward[agent_id] -= self.epsilon * np.mean((np.array(dt_obs) - np.array(obs)) ** 2) 
                
        # 与synchronization相关的reward
        # reward_syn = - np.mean((est_loc - self.setting.UserLocation) ** 2)
        # self.reward += reward_syn

        
        # Add the DT state to the routine
        # print(self.dt_routine)
        # print(est_loc.reshape(1,2*self.setting.num_ue))
        self.dt_routine = np.vstack((self.dt_routine, est_loc.reshape(1,2*self.setting.num_ue)))
        
        for agent_id in range(self.n_agents): 
            rb_indices = [rb_id for rb_id in range(len(self.setting.RbAllocation))]
            action = actions_vec[agent_id]
            associate_users[agent_id] = np.where(np.array(action[:self.setting.num_ue]) == 1)[0]

        # print(associate_users)
        
        for u in range(self.setting.num_ue): 
            options = [o for o, row in enumerate(associate_users) for _, val in enumerate(row) if val == u]
            if len(options) > 1: 
                bs = np.argmin([np.mean((est_loc[u]-self.setting.BS_loc[o])**2) for o in options])
                bs = options[bs]
                for o in options: 
                    if o != bs: 
                        associate_users[o] = associate_users[o][associate_users[o] != u]
        # print(associate_users)

        data_rate = np.zeros(self.n_agents)
        for agent_id in range(self.n_agents): 
            if len(associate_users[agent_id]) > 0: 
                rb_indices = [rb_id for rb_id in range(len(self.setting.RbAllocation[agent_id]))]
                # Determine the associate users for those agents who didn't do synchronization
                if agent_id not in agent_syn: 
                    reward_mat = self.setting.reward_matrix(agent_id, associate_users[agent_id], est_loc, self.setting.RbAllocation)
                # Allocate one RB to the synchronazation
                else: 
                    min_int = float("inf")
                    for rb_index in rb_indices: 
                        int_power = 0
                        for prev_agent_id in range(agent_id): 
                            int_source = self.setting.RbAllocation[prev_agent_id][rb_index]
                            if int_source != 0: 
                                int_power += self.setting.p_t*self.setting.o[-1] / np.sum((np.array(self.setting.BS_loc[prev_agent_id])-np.array(self.setting.CloudSeverLocation))**2)
                            
                        if int_power < min_int: 
                            min_int = int_power
                        rb_syn = rb_index
                    reward_mat = self.setting.reward_matrix(agent_id, associate_users[agent_id], est_loc, self.setting.RbAllocation)
                    reward_mat = np.delete(reward_mat, rb_syn, axis=0)
                    self.setting.RbAllocation[agent_id][rb_syn] = 11
                    rb_indices.remove(rb_syn)
                
                # print(reward_mat)
                # Calculate reward matrix
                max_value = np.max(reward_mat)
                cost_matrix = max_value - reward_mat 
                (row, column) = linear_sum_assignment(cost_matrix.copy()) # Get the RB allocation to users. 
                ans, ans_mat = ans_calculation(reward_mat, row, column) # Get the maximum sum of data rates. 

                # self.reward += ans

                for r,c in zip(row,column): 
                    self.setting.RbAllocation[agent_id][rb_indices[r]] = associate_users[agent_id][c] + 1
            
                self.reward[agent_id] += (1 - self.epsilon) * ans
                data_rate[agent_id] = ans

        self.T += 1
        if self.T >= self.episode_limit: 
            self.terminated = True

        self.stepInfo = {
            #   'State of PN': self.get_state(), # 当前PN的state
            #   'State of DNT': est_loc, # 当前DNT的state
            #   'Action': actions, # 选择的action
            #   'Reward': self.reward, # 获得的reward
            #   'Terminated': self.terminated, # 这个episode是否结束
            #   'Allocation_list': self.setting.RbAllocation, # 本轮的RB分配结果
            # 'Ans_mat': ans_mat
            "DNT accuracy": acc, 
            "data rate mean": np.sum(data_rate) / self.setting.num_ue
        }


        if not self.terminated: 
            self.setting.reset_step()
            self.usr_dist = self.usr_distribution(self.setting.UserLocation)
    
        return self.reward.sum(), self.terminated, self.stepInfo # vdn
        # return self.reward, self.terminated, {} # iql
            # 依次返回下一时刻的状态，当前的奖励，任务是否结束（True or False），info(字典，可选的返回信息)
            # 第15个state是一条轨迹的结尾，所以self.state不改变
            # 其余的self.state返回用户移动后的位置
            # 返回值的格式时由gym规范的，如果gym版本更新有改变这个规范的话，也要相应调整。


    def reset(self,seed=None,options=None):
        # 环境状态的初始化，输出初始化的state, 每一次episode结束、新episode开始之前也会执行
        self.setting.reset_epi()
        self.usr_dist = self.usr_distribution(self.setting.UserLocation)
        self.dt_routine = np.array(self.setting.UserLocation.reshape(1,2*self.setting.num_ue))
        self.T = 0
        # self.reward = 0
        self.reward = np.zeros(self.n_agents)
        self.terminated = False
        self.stepInfo = {}
        return self.get_obs(), self.get_state()


    def render(self): # 这是一个可视化函数，如果不重写的话，训练过程就不会进行可视化。可以根据需要去设计
        print(self.stepInfo)
        return self.stepInfo

    def save_replay(self):
        """Save a replay."""
        replay_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".replay"
        logging.info("Replay saved at: %s" % replay_path)

    def close(self): 
        pass