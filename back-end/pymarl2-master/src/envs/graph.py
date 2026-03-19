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
from pathlib import Path

logger = logging.getLogger(__name__)

# MIN_X, MAX_X = -150, 150
# MIN_Y, MAX_Y = -50, 50
DIRECTIONS = ['up', 'down', 'left', 'right']
DELTA_L = 1
USER_MOV_PROB_LIST = [(0.85, 0, 0, 0.15), 
                      (0.85, 0, 0.15, 0), 
                      (0, 0.85, 0.15, 0), 
                      (0, 0.85, 0, 0.15), 
                      (0.15, 0, 0.85, 0), 
                      (0, 0.15, 0.85, 0), 
                      (0.15, 0, 0, 0.85), 
                      (0, 0.15, 0, 0.85)]
STEPS_PER_ADJUSTMENT = 1

class Setting():
    def __init__(self, num_bs=3, num_ue=10, num_rb=3, np_random=None):
        self.np_random = np_random or np.random.RandomState()
        self.num_bs = num_bs
        self.num_ue = num_ue
        self.num_rb = num_rb # Number of RBs of each BS
        
        self.radius = 50
        # TODO: 
        # self.BaseStationLocation - Generate the location of the BS based on # of the BSs.
        # MIN_X, MAX_X, MIN_Y, MAX_Y - Based on # of the BSs and their converage, the edge of the coverage area need to be determined. 

        # The channel parameters  
        # channel_gain = gain_b * gain_u * delta * o / ((distance + epsilon) ** beta)
        self.noise_level = 0.05
        self.gain_b = 1.0  # Antenna gain - BSs
        self.gain_u = 1.0  # Antenna gain - users
        self.delta = 1.0  # Shadowing effect from cell c to user u  # TODO: can be modeled
        self.o = [np.random.rayleigh(scale=1) for _ in range(self.num_ue)]  # Small scale fading effect from BS b to user u
        self.N0_density = 1e-4  # Gaussian noise power spectral density 
        self.epsilon = 1e-10  # Small constant to avoid division by zero
        self.beta = 2  # Path loss exponent
        self.bandwidth = 1  # Bandwidth of each RB

        # if os.path.exists(self.pos_file):
        #     self.init_UserLocation = np.load(self.pos_file)
        #     print("User positions loaded. ")
        # else:
        #     self.init_UserLocation = np.array([self.sample_point_in_circle()
        #                                   for _ in range(self.num_ue)])
        #     np.save(self.pos_file, self.init_UserLocation)
        #     print("User positions generated & saved. ")
        # self.UserLocation = self.init_UserLocation.copy()
        self.user_association = np.array([np.random.randint(0, self.num_bs) for _ in range(self.num_ue)]) 
        self.UserLocation = np.array([self.sample_point_in_circle(self.BaseStationLocation[self.user_association[u]]) for u in range(self.num_ue)])  # TODO: Generate the initial location of the users. Can be generated randomly within the coverage area of the BSs.

    def sample_point_in_circle(self, bs_loc):
        theta = self.np_random.uniform(0, 2 * np.pi) 
        r = self.radius * np.sqrt(self.np_random.uniform(0, 1)) 
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return bs_loc + np.array([x, y])

    def random_move(self): 
        # TODO: Modify the user movement model if we have sufficient time. 
        for u in range(self.num_ue): 
            x_current = self.UserLocation[u][0]
            y_current = self.UserLocation[u][1]
            move = random.choices(DIRECTIONS, weights=USER_MOV_PROB_LIST[u%len(USER_MOV_PROB_LIST)])[0]
        
            if move == 'up':
                new_pos = np.array([x_current, y_current+DELTA_L])
                dist = np.linalg.norm(self.BaseStationLocation-new_pos)
                if dist <= self.radius:  
                    y_current += DELTA_L
            elif move == 'down': 
                new_pos = np.array([x_current, y_current-DELTA_L])
                dist = np.linalg.norm(self.BaseStationLocation-new_pos)
                if dist <= self.radius:  
                    y_current -= DELTA_L
            elif move == 'left': 
                new_pos = np.array([x_current-DELTA_L, y_current])
                dist = np.linalg.norm(self.BaseStationLocation-new_pos)
                if dist <= self.radius:  
                    x_current -= DELTA_L
            elif move == 'right': 
                new_pos = np.array([x_current+DELTA_L, y_current])
                dist = np.linalg.norm(self.BaseStationLocation-new_pos)
                if dist <= self.radius: 
                    x_current += DELTA_L

            self.UserLocation[u] = np.array([x_current, y_current])    
        return self.UserLocation

    def reset_step(self): 
        for _ in range(STEPS_PER_ADJUSTMENT): 
            self.random_move() 

    def reset_epi(self): 
        self.user_association = np.array([np.random.randint(0, self.num_bs) for _ in range(self.num_ue)])
        self.UserLocation = np.array([self.sample_point_in_circle(self.BaseStationLocation[self.user_association[u]]) for u in range(self.num_ue)])  # TODO: Generate the initial location of the users. Can be generated randomly within the coverage area of the BSs.
        
# Environment class
class GraphEnv(MultiAgentEnv):

    def __init__(self, seed):
        self.setting = Setting()

        self.n_agents = self.setting.num_bs
        self.agent_id = [i for i in range(self.n_agents)]
        # TODO: To be determined based on the agorithm detail. 
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.setting.num_ue,2), dtype=np.float32) 
        # self.action_space = gym.spaces.MultiBinary(self.setting.num_ue+1) # Action: 1 - allocate an RB to the information trans, 0 - otherwise
        self.n_actions = self.get_total_actions()
        self.gamma = 1 

        self.reward = np.zeros(self.n_agents)
        self.episode_limit = 30
        self.T = 0
        self.terminated = False
        self.stepInfo = {}
    
    def get_obs(self): 
        # TODO: Implement the function of getting the global observation. 
        # NORMALIZATION: The observation is the input of the nueral network, so we need to normalize it. 
        return
        # obs = []
        # for bs_id in range(self.n_agents): 
        #     if_usr_in_range = self.usr_dist[bs_id]
        #     partial_obs = np.where(if_usr_in_range[:, np.newaxis], np.array(self.setting.UserLocation), np.array([0, 0]))
        #     partial_obs = partial_obs.reshape((2*self.setting.num_ue,))
        #     norm_obs = partial_obs / 10
        #     obs.append(norm_obs)
        # return obs
    
    def get_obs_agent(self, agent_id): 
        # TODO: Implement the function of getting the local observation for each agent.
        return 
        # if_usr_in_range = self.usr_dist[agent_id]
        # partial_obs = np.where(if_usr_in_range[:, np.newaxis], np.array(self.setting.UserLocation), np.array([0, 0]))
        # partial_obs = partial_obs.reshape((2*self.setting.num_ue,))
        # norm_obs = partial_obs / 10
        # return norm_obs
    
    def get_obs_size(self): 
        # TODO
        # Should be the size of the obs list, which should pass to initialize the controller
        return 
        # return 2*self.setting.num_ue
    
    def get_state(self): 
        # TODO: What's the difference between the state and the observation? 
        return 
        # return (self.setting.UserLocation/10).reshape((2*self.setting.num_ue,))

    def get_state_size(self): 
        # TODO
        return 
        # return 2*self.setting.num_ue

    def get_avail_actions(self): 
        # TODO
        # Return a bool np_array for the action which is available for each agent. 
        return 
        # avail_actions = []
        # for agent_id in range(self.n_agents): 
        #     avail_actions_agent = self.get_avail_agent_actions(agent_id)
        #     avail_actions.append(avail_actions_agent)
        # return avail_actions

    def get_avail_agent_actions(self, agent_id): 
        # TODO
        # Returns the available actions of agent_id 
        return
        # avail_usr_conn = np.array(self.usr_dist[agent_id])
        # invalid_usr = np.where(avail_usr_conn == False)[0]

        # avail_actions = np.ones(self.n_actions)
        # for action in range(self.n_actions): 
        #     action_bits = format(action, "011b")
        #     # Connect to the users which are not in the range
        #     if any(action_bits[i_u] == '1' for i_u in invalid_usr): 
        #         avail_actions[action] = 0
        #     # The number of the connections is more than the number of available RBs
        #     if (np.sum([int(b) for b in action_bits]) > self.n_rbs): 
        #         avail_actions[action] = 0

        # return avail_actions
    
    def get_total_actions(self):
        # Returns the total number of actions an agent could ever take 
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return 
        # n_actions= 2 ** np.prod(self.action_space.shape)
        # return n_actions
   
    def seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions): 
        # TODO

        return self.reward.sum(), self.terminated, self.stepInfo # vdn
        # return self.reward, self.terminated, {} # iql
            # 依次返回下一时刻的状态，当前的奖励，任务是否结束（True or False），info(字典，可选的返回信息)
            # 第15个state是一条轨迹的结尾，所以self.state不改变
            # 其余的self.state返回用户移动后的位置
            # 返回值的格式时由gym规范的，如果gym版本更新有改变这个规范的话，也要相应调整。


    def reset(self,seed=None,options=None):
        self.setting.reset_epi()
        self.T = 0
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