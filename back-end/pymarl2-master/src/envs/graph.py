import logging
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from envs.multiagentenv import MultiAgentEnv
import time
# from pathlib import Path

logger = logging.getLogger(__name__)

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
    def __init__(self, num_bs=3, num_ue=10, num_rb=3, random_generator=None):
        self.np_random = random_generator
        self.num_bs = num_bs
        self.num_ue = num_ue
        self.num_rb = num_rb # Number of RBs of each BS
        
        self.radius = 50
        # TODO: 
        # self.BaseStationLocation - Generate the location of the BS based on # of the BSs.
        self.BaseStationLocation = np.array([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]])

        # The channel parameters  
        # channel_gain = gain_b * gain_u * delta * o / ((distance + epsilon) ** beta)
        self.noise_level = 0.05
        self.gain_b = 1.0  # Antenna gain - BSs
        self.gain_u = 1.0  # Antenna gain - users
        self.delta = 1.0  # Shadowing effect from cell c to user u  # TODO: can be modeled
        self.o = self.np_random.rayleigh(scale=1.0, size=(self.num_bs, self.num_ue, self.num_rb))  # Small scale fading effect from BS b to user u
        self.N0_density = 1e-4  # Gaussian noise power spectral density 
        self.epsilon = 1e-10  # Small constant to avoid division by zero
        self.beta = 2  # Path loss exponent
        self.bandwidth = 1  # Bandwidth of each RB
        self.p_tx = 1  # Transmission power of each RB 

        # if os.path.exists(self.pos_file):
        #     self.init_UserLocation = np.load(self.pos_file)
        #     print("User positions loaded. ")
        # else:
        #     self.init_UserLocation = np.array([self.sample_point_in_circle()
        #                                   for _ in range(self.num_ue)])
        #     np.save(self.pos_file, self.init_UserLocation)
        #     print("User positions generated & saved. ")
        # self.UserLocation = self.init_UserLocation.copy()
        self.user_association = np.array([self.np_random.integers(0, self.num_bs) for _ in range(self.num_ue)]) 
        self.UserLocation = np.array([self.sample_point_in_circle(self.BaseStationLocation[self.user_association[u]]) for u in range(self.num_ue)])  # TODO: Generate the initial location of the users. Can be generated randomly within the coverage area of the BSs.

    def sample_point_in_circle(self, bs_loc):
        theta = self.np_random.uniform(0, 2 * np.pi) 
        r = self.radius * np.sqrt(self.np_random.uniform(0, 1)) 
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return bs_loc + np.array([x, y])
    
    def num_ue_in_bs_range(self): 
        num_ue_in_bs = np.zeros(self.num_bs, dtype=int)
        for bs_id in range(self.num_bs): 
            for u_in_bs in self.user_association: 
                if u_in_bs == bs_id: 
                    num_ue_in_bs[bs_id] += 1
        return num_ue_in_bs

    def random_move(self): 
        # TODO: Modify the user movement model if we have sufficient time. 
        for u in range(self.num_ue): 
            bs_loc = self.BaseStationLocation[self.user_association[u]]
            x_current = self.UserLocation[u][0]
            y_current = self.UserLocation[u][1]
            move = self.np_random.choice(DIRECTIONS, p=USER_MOV_PROB_LIST[u%len(USER_MOV_PROB_LIST)])
        
            if move == 'up':
                new_pos = np.array([x_current, y_current+DELTA_L])
                dist = np.linalg.norm(bs_loc - new_pos)
                if dist <= self.radius:  
                    y_current += DELTA_L
            elif move == 'down': 
                new_pos = np.array([x_current, y_current-DELTA_L])
                dist = np.linalg.norm(bs_loc - new_pos)
                if dist <= self.radius:  
                    y_current -= DELTA_L
            elif move == 'left': 
                new_pos = np.array([x_current-DELTA_L, y_current])
                dist = np.linalg.norm(bs_loc - new_pos)
                if dist <= self.radius:  
                    x_current -= DELTA_L
            elif move == 'right': 
                new_pos = np.array([x_current+DELTA_L, y_current])
                dist = np.linalg.norm(bs_loc - new_pos)
                if dist <= self.radius: 
                    x_current += DELTA_L

            self.UserLocation[u] = np.array([x_current, y_current])    
        return self.UserLocation

    def reset_step(self): 
        for _ in range(STEPS_PER_ADJUSTMENT): 
            self.random_move() 

    def reset_epi(self): 
        self.o = self.np_random.rayleigh(scale=1.0, size=(self.num_bs, self.num_ue, self.num_rb)) 
        self.user_association = np.array([self.np_random.integers(0, self.num_bs) for _ in range(self.num_ue)])
        self.UserLocation = np.array([self.sample_point_in_circle(self.BaseStationLocation[self.user_association[u]]) for u in range(self.num_ue)])  # TODO: Generate the initial location of the users. Can be generated randomly within the coverage area of the BSs.
        
# Environment class
class GraphEnv(MultiAgentEnv):

    def __init__(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        self.setting = Setting(random_generator=self.np_random)

        self.n_agents = self.setting.num_bs
        self.agent_id = [i for i in range(self.n_agents)]
        # TODO: To be determined based on the agorithm detail. 
        self.n_actions = self.get_total_actions()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2*self.setting.num_ue, ), dtype=np.float32) 
        self.action_space = gym.spaces.Discrete(self.n_actions)

        self.gamma = 1 

        self.reward = np.zeros(self.n_agents)
        self.episode_limit = 15
        self.T = 0
        self.terminated = False
        self.stepInfo = {}
    
    def get_obs(self): 
        # Implement the function of getting the global observation. 
        # NORMALIZATION: The observation is the input of the nueral network, so we need to normalize it. 
        return [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]
    
    def get_obs_agent(self, agent_id): 
        # Implement the function of getting the local observation for each agent.
        # NORMALIZATION: The observation is the input of the nueral network, so we need to normalize it. 
        user_assocated = (self.setting.user_association == agent_id)
        partial_obs = np.where(user_assocated[:, np.newaxis], np.array(self.setting.UserLocation), np.array([0, 0]))
        partial_obs = partial_obs.reshape((2*self.setting.num_ue,))
        norm_obs = partial_obs / 10
        return norm_obs
    
    def get_obs_size(self): 
        # Should be the size of the obs list, which should pass to initialize the controller
        return 2*self.setting.num_ue
    
    def get_state(self): 
        return (self.setting.UserLocation/10).reshape((2*self.setting.num_ue,))

    def get_state_size(self): 
        return 2*self.setting.num_ue

    def get_avail_actions(self): 
        # Return a bool np_array for the action which is available for each agent. 
        avail_actions = []
        for agent_id in range(self.n_agents): 
            avail_actions_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_actions_agent)
        return np.array(avail_actions)

    def get_avail_agent_actions(self, agent_id): 
        # Returns the available actions of agent_id 
        avail_actions = np.ones(self.n_actions, dtype=int)
        user_associated = np.where(self.setting.user_association == agent_id)[0]
        user_associated = user_associated + 1 # 0 means no user allocated, so we add 1 to make the user index start from 1. 
        for action in range(self.n_actions): 
            allocation_vector = self._decode_action(action)
            allocated_to_users = allocation_vector[allocation_vector > 0]
            if not np.all(np.isin(allocated_to_users, user_associated)): 
                avail_actions[action] = 0
            elif len(allocated_to_users) != len(np.unique(allocated_to_users)): 
                avail_actions[action] = 0
        return avail_actions
    
    def get_total_actions(self):
        # Returns the total number of actions an agent could ever take 
        # TODO: Current action is RB allocation
        n_actions = (self.setting.num_ue+1) ** self.setting.num_rb
        return n_actions
   
    def seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed)
        self.setting.np_random = self.np_random
        return [seed]

    def step(self, actions): 
        self.reward = np.zeros(self.n_agents, dtype=np.float32)
        # Action to RB allocation vector
        rb_allocation = np.array([self._decode_action(action) for action in actions])

        # Check the availability of the actions 
        for agent_id in range(self.n_agents): 
            rb_allocation_agent = rb_allocation[agent_id]
            for user in rb_allocation_agent: 
                if user > 0: 
                    user_index = user - 1
                    if self.setting.user_association[user_index] != agent_id: 
                        raise ValueError(f"Invalid action: Agent {agent_id} allocated RB to user {user_index} which is not associated with it.")
        
        # Calculate the reward for each agent based on the RB allocation and the channel conditions
        for agent_id in range(self.n_agents): 
            bs_loc = self.setting.BaseStationLocation[agent_id]
            rb_allocation_agent = rb_allocation[agent_id]
            reward_agent = 0
            for rb_index, user in enumerate(rb_allocation_agent): 
                if user > 0: 
                    user_index = user - 1
                    user_loc = self.setting.UserLocation[user_index]
                    distance = np.linalg.norm(bs_loc - user_loc)
                    channel_gain = self.setting.gain_b * self.setting.gain_u * self.setting.delta * self.setting.o[agent_id][user_index][rb_index] / ((distance + self.setting.epsilon) ** self.setting.beta)
                    # Noise from the same channel
                    interference = 0
                    for bs in range(self.n_agents): 
                        if bs == agent_id: 
                            continue
                        rb_allocation_other_agent = rb_allocation[bs]
                        if np.linalg.norm(self.setting.BaseStationLocation[bs] - user_loc) <= self.setting.radius and rb_allocation_other_agent[rb_index] > 0: 
                            interference += self.setting.p_tx * self.setting.gain_b * self.setting.gain_u * self.setting.delta * self.setting.o[bs][user_index][rb_index] / ((np.linalg.norm(self.setting.BaseStationLocation[bs] - user_loc) + self.setting.epsilon) ** self.setting.beta)
                    
                    sinr = self.setting.p_tx * channel_gain / (self.setting.noise_level + interference + self.setting.epsilon)
                    reward_agent += np.log2(1 + sinr) * self.setting.bandwidth

            self.reward[agent_id] = reward_agent

        # Next state
        self.T += 1
        if self.T >= self.episode_limit: 
            self.terminated = True
        else: 
            self.setting.reset_step()

        return self.reward.sum(), self.terminated, self.stepInfo # vdn
        # return self.reward, self.terminated, {} # iql
            # 依次返回下一时刻的状态，当前的奖励，任务是否结束（True or False），info(字典，可选的返回信息)
            # 第15个state是一条轨迹的结尾，所以self.state不改变
            # 其余的self.state返回用户移动后的位置
            # 返回值的格式时由gym规范的，如果gym版本更新有改变这个规范的话，也要相应调整。


    def reset(self,seed=None,options=None):
        if seed is not None: 
            self.seed(seed)
        self.setting.reset_epi()
        self.T = 0
        self.reward = np.zeros(self.n_agents)
        self.terminated = False
        self.stepInfo = {}
        return self.get_obs(), self.get_state()


    def render(self): # 这是一个可视化函数，如果不重写的话，训练过程就不会进行可视化。可以根据需要去设计
        pass

    def save_replay(self):
        """Save a replay."""
        replay_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".replay"
        logging.info("Replay saved at: %s" % replay_path)

    def close(self): 
        pass

    def _decode_action(self, action): 
        action_bits = np.zeros(self.setting.num_rb, dtype=int)
        for position in range(self.setting.num_rb - 1, -1, -1): 
            action_bits[position] = action % (self.setting.num_ue+1)
            action = action // (self.setting.num_ue+1)
        return action_bits