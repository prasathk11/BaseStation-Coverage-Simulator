from envs.multiagentenv import MultiAgentEnv
import gym
import numpy as np
from gym import spaces
from absl import logging
import time
import math
import copy
from utils.dict2namedtuple import convert
import torch
from gym.spaces import Box
from functools import partial
from gym.wrappers import TimeLimit

import matplotlib.pyplot as plt
from operator import attrgetter
from copy import deepcopy
import operator
from functools import reduce
import csv
import numpy as np
from scipy import interpolate
import heapq
import pylab as pl
import enum
import math
import time

import random
import math
import numpy
import time
import numpy as np
import csv





class UafOrient(MultiAgentEnv):
    '''
    action 动作空间
    水平角、竖直角度、速度、功率
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | yaw    | 0    | 360 |
    | 1   | pitch  | -90  | 90  |
    | 2   | speed  | 0    | 50  |
    | 3   | power  | 0    | 0.1 |

    observation 环境空间
    距离
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|

    | 0   |        x         | 0    |1000 |
    | 1   |        y         | 0    |1000 |
    | 2   |        z         | 0    |1000 |
    | 3   |    distance      | 0    |10000|
    '''
        
    def __init__(
        self,
        seed,
        n_agents = 6
    ):
        self.limit1_power = 500
        self.limit2_time = 1000
    
        self.fc = 9 * 10 ** 7
        self.aaa = 0.1 
        self.bbb = 21 




        self.limit = False
        self.pos_list_record = []
        self.pos_list_record_tdoa = []
        self._episode_count = 0                                
        self.n_agents=n_agents
        self.max_reward=0
        self._controller = None
        self.a = dict()
        t = 0

      
        for i in range(3):
            for j in range(2):
                for v in range(1):
                    tmp = [i,j,v]
                    self.a[t] = tmp
                    t =t +1


        self.time = 1 
        self.target_pos = [0,0,333.333]
        self.target_speed = 16
        self.target_yaw = 0
        self.target_pitch = 0
        self.speed = 3e8
        self.c = 3 * (10 ** 8)  
        self.ep_2 = 3.16 * 10 ** -13  
        self.f0 =6000000 
        self.w = 10 ** 3 
        self.UAFs = n_agents
        obs = []
        self.real_poses = []
        self.tdoa_poses = []

        sender_pos =  np.array([249,248,102])
        sender_pos_list = sender_pos.tolist()
        sender_pos_list.append(0) 
        obs.append(np.array(sender_pos_list))
        
        #TODO
        pos_tttt =[[0,0,332.333],[0,0,333.333],[-489.894,-282.842,233.333],[489.894,-282.842,433.333],[0,565.68,335.333]]
        power = 3
        for i in range(self.UAFs-1):
            pos = np.array(pos_tttt[i])
            dis,_,scale = self.calculate_r_distance(sender_pos, pos, self.target_pos, power)
            pos = pos.tolist()
            pos.append(dis)
            obs.append(np.array(pos))
        self._obs = obs
        self.episode_limit = 30
        self.n_actions = 6
    

    '''
    欧式距离算法
    '''
    def eucliDist(self,A, B):
        return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))
 
    '''
    计算sender uaf和target uaf距离 + receiver uaf和target uaf距离
    '''
    def calculate_r_distance(self, sender_pos, receiver_pos, target_pos,sender_power):
        dis_sender2target =  ((sender_pos[0] - target_pos[0]) ** 2 + (sender_pos[1] - target_pos[1])  ** 2 + (sender_pos[2] - target_pos[2]) ** 2)**0.5
        dis_receive2target = ((receiver_pos[0]-target_pos[0])**2 + (receiver_pos[1]-target_pos[1])**2 + (receiver_pos[2] - target_pos[2]) ** 2) ** 0.5
        tmp_a = (self.f0 + self.w/2) ** 3
        tmp_b = (self.f0 - self.w/2) ** 3
        scale = ((3* self.c**2 * self.ep_2)/(8 * math.pi *(tmp_a-tmp_b))) * (1 /(sender_power * ((dis_sender2target*dis_receive2target) ** -2)))
        self.scale = scale
        cc = math.sqrt(scale)
        e =  np.random.normal(loc=0.0, scale=cc, size=None) * 1
        r = dis_receive2target + dis_sender2target
        r_t = dis_receive2target + dis_sender2target + e
        self.r_t = r_t
        return r_t,e,scale


    def update_pos(self,pos,yaw,pitch,speed):
        yaw_agent, pitch_agent = math.pi * (yaw / 180), math.pi * (pitch / 180)
        move_x = speed * self.time * math.cos(yaw_agent) * math.cos(pitch_agent)
        move_y = speed * self.time * math.sin(yaw_agent) * math.cos(pitch_agent)
        move_z = speed * self.time * math.sin(pitch_agent)
        uaf_pos = []
        uaf_pos.append(move_x + pos[0])
        uaf_pos.append(move_y + pos[1])
        uaf_pos.append(move_z + pos[2])
        return uaf_pos

    '''
    """ Returns reward, terminated, info """
    '''
  
    
    def step(self, actions):
        start_time = time.time()
        self.pos_list_record.append(self.target_pos)
        actions = actions.tolist()
        action_list = []
   
        for tmp in actions:
            TTT = [0,0,11,3]
            tmp_ac = self.a[tmp]
          
            
            if tmp_ac[0] == 1:
                TTT[0] = 90
            elif tmp_ac[0] == 2:
                TTT[0] = 180
            elif tmp_ac[0] == 3:
                TTT[0] = 270

            if tmp_ac[1] == 1:
                TTT[1] = -90
            elif tmp_ac[1] == 2:
                TTT[1] = 90

            action_list.append(TTT)

        uafs_new_pos = []
        power = action_list[0][3]
        sender_pos = []
        


     
        for i in range(self.UAFs):
            yaw_agent, pitch_agent, speed, power = math.pi*(action_list[i][0]/180),math.pi*(action_list[i][1]/180),action_list[i][2],action_list[i][3]
            move_x = speed * self.time * math.cos(yaw_agent) * math.cos(pitch_agent)
            move_y = speed * self.time * math.sin(yaw_agent) * math.cos(pitch_agent)
            move_z = speed * self.time * math.sin(pitch_agent)
            uaf_pos = []
            uaf_pos.append(move_x + self._obs[i][0])
            uaf_pos.append(move_y + self._obs[i][1])
            uaf_pos.append(move_z + self._obs[i][2])
            uafs_new_pos.append(uaf_pos[0:3])

        reveivers_pos = uafs_new_pos[1:]
        sender_pos = uafs_new_pos[0]
        reveivers_diss = []
        obs = []
        tmp_obs = copy.deepcopy(uafs_new_pos[0])
        tmp_obs.append(0)
        obs.append(tmp_obs)
        R = 0
        scale_list = []
        _,_,scale = self.calculate_r_distance(sender_pos,uafs_new_pos[0],self.target_pos,power)
       
        for i in range(self.UAFs-1):
            tmp_obs = copy.deepcopy(uafs_new_pos[i+1])
            tmp_dis,rr,scale = self.calculate_r_distance(sender_pos,uafs_new_pos[i+1],self.target_pos,power)
            scale_list.append(scale)
            tmp_obs.append(tmp_dis)
            obs.append(tmp_obs)
            reveivers_diss.append(tmp_dis)
            
            if rr > 0:
                rr = -rr
            R = R + rr

        self._obs = obs
        
        dis_4 = []
        for i in range(3):
            dis_4.append(obs[i+2][3]-obs[1][3])
        reveivers_pos = torch.tensor(reveivers_pos, device = 'cpu')
        A = np.mat(reveivers_pos).T

        dis_bs = []
    
        for i in range(len(reveivers_diss)-1):
            dis_bs.append([reveivers_diss[1+i]-reveivers_diss[0]])
       

        tmpTarget = self.target_pos

        tmp = [scale_list[0]]*(self.UAFs -2)
        Q = []
        for i in range(self.UAFs -2):
            tmp = [scale_list[0]]*(self.UAFs -2)
            tmp[i] = tmp[i] + scale_list[i+1]
          
            Q.append(tmp)
            
       
        reward = self.tdoa_location(dis_bs,reveivers_pos.tolist(),Q,self.speed)
       
       
      
        if self.time_step == 0:
            self.target_yaw = 2
        elif self.time_step == 1:
            self.target_yaw = 4
        elif self.time_step == 2:
            self.target_yaw = 6
        elif self.time_step == 3:
            self.target_yaw = 8
        elif self.time_step == 4:
            self.target_yaw = 10
        elif self.time_step == 5:
            self.target_yaw = 12
        elif self.time_step== 6:
            self.target_yaw = 14
        elif self.time_step == 7:
            self.target_yaw = 16
        elif self.time_step == 8:
            self.target_yaw =14
        elif self.time_step == 9:
            self.target_yaw = 12
        elif self.time_step == 10:
            self.target_yaw = 10
        elif self.time_step == 11:
            self.target_yaw = 8
        elif self.time_step == 12:
            self.target_yaw = 6
        elif self.time_step == 13:
            self.target_yaw = 4
        elif self.time_step == 14:
            self.target_yaw = 2
        elif self.time_step == 15:
            self.target_yaw = 0
        elif self.time_step == 16:
            self.target_yaw = -2
        elif self.time_step == 17:
            self.target_yaw = -4
        elif self.time_step == 18:
            self.target_yaw = -6
        elif self.time_step == 19:
            self.target_yaw = -8
        elif self.time_step == 20:
            self.target_yaw = -10
        elif self.time_step== 21:
            self.target_yaw = -12
        elif self.time_step == 22:
            self.target_yaw = -14
        elif self.time_step == 23:
            self.target_yaw =-16
        elif self.time_step == 24:
            self.target_yaw = -14
        elif self.time_step == 25:
            self.target_yaw = -12
        elif self.time_step == 26:
            self.target_yaw = -10
        elif self.time_step == 27:
            self.target_yaw = -8
        elif self.time_step == 28:
            self.target_yaw = -6
        elif self.time_step == 29:
            self.target_yaw = -4
        else:
            self.target_yaw = -2
      
        
        if self.time_step%10 == 0:
            self.target_pitch = 3
        elif self.time_step%3 == 1:
            self.target_pitch = 3
        else:
            self.target_pitch = 3
        
        self.target_pos = self.update_pos(self.target_pos, self.target_yaw, self.target_pitch, self.target_speed)
        self.time_step += 1
        t = reward
        
        if self.time_step >= self.episode_limit:
           
            end_time = time.time()   
            execution_time = end_time - start_time 
          
            file=open(r"./data.txt","a",encoding="UTF-8")
            file.write(str(self.pos_list_record)+"\n");
            file.write(str(self.pos_list_record_tdoa)+"\n\n");
            file.close()    
            terminated = True
            
        else:
         
            end_time = time.time() 
            execution_time = end_time - start_time
           
            terminated = False
        
        
        return reward, terminated, {}
    
        


    
    
    def reset(self):
        self.pos_list_record = []
        self.pos_list_record_tdoa = []
        self.time_step = 0
        self.target_pos = [0,0,333.333]
        obs = []
        sender_pos = np.array([249,248,102])
        sender_pos_list = sender_pos.tolist()
        sender_pos_list.append(0) 
        obs.append(np.array(sender_pos_list))
        #TODO
        pos_ttt =[[0,0,332.333],[0,0,333.333],[-489.894,-282.842,233.333],[489.894,-282.842,433.333],[0,565.68,335.333]]
        power = 3
        for i in range(self.UAFs-1):
            pos = np.array(pos_ttt[i])
            dis,_ ,_= self.calculate_r_distance(sender_pos, pos, self.target_pos, power)
            pos = pos.tolist()
            pos.append(dis)
            obs.append(np.array(pos))
        self._obs = obs
        return  self.get_obs(), self.get_state



    def get_avail_actions(self):
        if self.limit:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return np.ones(shape=(self.n_agents, self.n_actions,))

    def limit_1(self, phi, v, limit_value):
        c1 = 4929
        c2 = 0.002
        v_h = 9.43
        m = 4
        g = 9.8
        rst = c1/(100*math.cos(phi)**2+((v*math.cos(phi))**4+4*v_h**4)**0.5)**0.5 + m*g*v*math.sin(phi) + c2*math.cos(phi)**3
        if rst <= limit_value:
            return True
        else:
            return False

    def limit_2(self, agent_id,i, phi, v):
        if agent_id == 0:
            return True
        else:
            yaw = i * 90
            uav_pos = self._obs[agent_id][1:]
            tmp_uav_pos = self.update_pos(uav_pos,yaw, phi, v)
        
            uav2bs_dis = ((tmp_uav_pos[0] - 249) ** 2 + (tmp_uav_pos[1] - 248)  ** 2 + (tmp_uav_pos[2] - 0) ** 2)**0.5
            tmp_l =  20 * math.log(self.fc * 4 * math.pi / self.speed) + 20 * math.log(uav2bs_dis)
            l1 = tmp_l + self.aaa
            l2 = tmp_l + self.bbb
            p1 = (1 + 11.9* math.exp(-0.13 * (math.asin(tmp_uav_pos[2]/uav2bs_dis) - 11.9 ))) ** (-1)
            rst_l = p1 * l1 + (1-p1)*l2
            snr = 3*10 ** (13-rst_l/10)
            t = 2/(10 **6 * math.log2(1+snr))
         
            if t <= self.limit2_time:
                return True
            else:
                return False


    def get_avail_agent_actions(self, agent_id):
        tmp_dict = {
            0:[0,3,6,9],
            1:[1,4,7,10],
            2:[2,5,8,11]
        }
        avail_actions = np.zeros(self.n_actions)
        if self.limit_1(0, self.target_speed, self.limit1_power):
            for i in range(4):
                if self.limit_2(agent_id, i, 0, self.target_speed):      
                    avail_actions[tmp_dict[0][i]] = 1
        if self.limit_1(-90, self.target_speed, self.limit1_power):
            for i in range(4):
                if self.limit_2(agent_id, i, -90, self.target_speed):
                    avail_actions[tmp_dict[1][i]] = 1
        if self.limit_1(90, self.target_speed, self.limit1_power):
            for i in range(4):
                if self.limit_2(agent_id, i, 90, self.target_speed):
                    avail_actions[tmp_dict[2][i]] = 1
        if np.all(avail_actions == 0): 
            avail_actions[0] = 1
            return avail_actions
        else:
            return avail_actions

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix 
        replay_dir = self.replay_dir or ""
        replay_path = replay_dir+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".replay"
        logging.info("Replay saved at: %s" % replay_path)
        
    def get_obs_agent(self, agent_id):
       
        """ Returns observation for agent_id """
        return self._obs[agent_id]



    def get_stats(self):
        return  {}

    def get_obs(self):
        t = self._obs
        return t


    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if True:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

    def get_state_size(self):
        """Returns the size of the global state."""
        if True:
            return self.get_obs_size() * self.n_agents



    def get_obs_size(self):
        """Returns the size of the observation, 4个数字"""
        return 4



    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions


    def close(self):
        """Close StarCraft II."""
    
    def tdoa_location(self, r, bs, Q, c):
        Q = np.array(Q)
        r = np.array(r)
        WLS2_near_out = np.zeros((3,1))
        WLS2_far_out = np.zeros((3,1))
        WLS2_near = np.zeros((3, 8))
        N = len(bs)
        K = np.zeros((N,1))
        for i in range(N):
            K[i] = bs[i][0]**2 + bs[i][1]**2 + bs[i][2]**2

        ha = 0.5*(r**2 - K[1:]+ K[0])
        bs_array = np.array(bs)
        tmps = bs_array[0][0]
        tmp = bs_array[1:][:,0] - bs_array[0][0]
        a =-1 * np.array([(bs_array[1:][:,0] - bs_array[0][0]).tolist(), (bs_array[1:][:,1] - bs_array[0][1]).tolist(), (bs_array[1:][:,2] - bs_array[0][2]).tolist(),r[:,0].tolist()])
        
        Ga = a.T
       
        Za1 =  np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Ga.T,np.linalg.inv(Q)),Ga)),Ga.T),np.linalg.inv(Q)),ha)
        WLS1_far = np.array(Za1[:3,:])
       
        W1tem = np.ones((N-1,1))
        W1 = Za1[0] * W1tem
        W2 = Za1[1] * W1tem
        W3 = Za1[2] * W1tem
       
        Batem = np.transpose(np.sqrt((W1[0:][:,0] - bs_array[1:][:,0])**2 + (W2[0:][:,0] - bs_array[1:][:,1])**2 + (W3[0:][:,0] - bs_array[1:][:,2])**2))
        Ba = np.diag(Batem)
        c2 = c**2
        Fa = c2*np.dot(np.dot(Ba,Q),Ba)
        Za2 = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Ga.T, np.linalg.inv(Fa)), Ga)), Ga.T), np.linalg.inv(Fa)), ha)
        WLS1_near = np.array(Za2[:3,:])
      
        W1 = Za2[0] * W1tem
        W2 = Za2[1] * W1tem
        W3 = Za2[2] * W1tem
        Batem = np.transpose(np.sqrt((W1[0:][:,0] - bs_array[1:][:,0])**2 + (W2[0:][:,0] - bs_array[1:][:,1])**2 + (W3[0:][:,0] - bs_array[1:][:,2])**2))
        Ba = np.diag(Batem)
        c2 = c**2
        Fa = c2*np.dot(np.dot(Ba,Q),Ba)

        Gb = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
        Bb_temp = [Za2[0][0] - bs_array[0][0],Za2[1][0] - bs_array[0][1], Za2[2][0] - bs_array[0][2],np.linalg.norm(Za2[0:3][:,0]-np.transpose(bs_array[0,:][0:3]))]
        Bb = np.diag(Bb_temp)
        cov_Za = np.linalg.inv(np.dot(np.dot(Ga.T,np.linalg.inv(Fa)),Ga))
        Fb = 4*np.dot(np.dot(Bb,cov_Za),Bb)
        h = np.array([[(Za2[0][0] - bs_array[0][0])**2],[(Za2[1][0] - bs_array[0][1])**2],[(Za2[2][0] - bs_array[0][2])**2],[(Za2[3][0])**2]])
        Zb1 = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(Gb.T, np.linalg.inv(Fb)), Gb)), Gb.T), np.linalg.inv(Fb)), h)
        WLS2_near[0:3][:,0] = np.array([(Zb1[0][0])**0.5+bs_array[0][0],(Zb1[1][0])**0.5+bs_array[0][1],np.sqrt(Zb1[2][0])**0.5+bs_array[0][2]]).T
        WLS2_near[0:3][:,1] = np.array([(Zb1[0][0])**0.5 + bs_array[0][0], -(Zb1[1][0])**0.5 + bs_array[0][1],(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,2] = np.array([(Zb1[0][0])**0.5 + bs_array[0][0], (Zb1[1][0])**0.5 + bs_array[0][1],-(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,3] = np.array([(Zb1[0][0])**0.5 + bs_array[0][0], -(Zb1[1][0])**0.5 + bs_array[0][1],-(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,4] = np.array([-(Zb1[0][0])**0.5 + bs_array[0][0], (Zb1[1][0])**0.5 + bs_array[0][1],(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,5] = np.array([-(Zb1[0][0])**0.5 + bs_array[0][0], -(Zb1[1][0])**0.5 + bs_array[0][1],(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,6] = np.array([-(Zb1[0][0])**0.5 + bs_array[0][0], (Zb1[1][0])**0.5 + bs_array[0][1],-(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_near[0:3][:,7] = np.array([-(Zb1[0][0])**0.5 + bs_array[0][0], -(Zb1[1][0])**0.5 + bs_array[0][1],-(Zb1[2][0])**0.5 + bs_array[0][2]]).T
        p=len(r)+1
        Rems=np.zeros((p,p-1))
        for i in range(p-1):
            for j in range(1,1,p-1):
                Rems[j-1][i] = np.sqrt((bs_array[j][0]-WLS2_near[0][i])**2+(bs_array[j][1]-WLS2_near[1][i])**2+(bs_array[j][2]-WLS2_near[2][i])**2)-np.sqrt((bs_array[0][0]-WLS2_near[0][i])**2+(bs_array[0][1]-WLS2_near[1][i])**2+(bs_array[0][3]-WLS2_near[2][i])**2)
        for i in range(p-1):
            num = 0
            for j in range(1,1,p-1):
                if np.sign(Rems[j-1][i])*np.sign(r[j-1][i])==1:
                    num = num + 1
            if num >= (p-1):
                WLS2_near_out = WLS2_near[0:3][:,i]

      
        Bb_temp = [Za1[0][0] - bs_array[0][0],Za1[1][0] - bs_array[0][1], Za1[2][0] - bs_array[0][2],np.linalg.norm(Za2[0:3][:,0]-np.transpose(bs_array[0,:][0:3]))]
        Bb = np.diag(Bb_temp)
        Zb2 = np.linalg.inv((Gb.T) @ np.linalg.inv(Bb) @ (Ga.T) @ np.linalg.inv(Q) @ Ga @ np.linalg.inv(Bb) @ Gb) @ (Gb.T) @ np.linalg.inv(Bb) @ (Ga.T) @ np.linalg.inv(Q) @ Ga @ np.linalg.inv(Bb) @ h
        WLS2_far = np.zeros((3,8))
       
        WLS2_far[0:3][:, 0] = np.array([(Zb2[0][0])**0.5 + bs_array[0][0], (Zb2[1][0])**0.5 + bs_array[0][1],(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 1] = np.array([(Zb2[0][0])**0.5 + bs_array[0][0], -(Zb2[1][0])**0.5 + bs_array[0][1],(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 2] = np.array([(Zb2[0][0])**0.5 + bs_array[0][0], (Zb2[1][0])**0.5 + bs_array[0][1],-(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 3] = np.array([(Zb2[0][0])**0.5 + bs_array[0][0], -(Zb2[1][0])**0.5 + bs_array[0][1],-(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 4] = np.array([-(Zb2[0][0])**0.5 + bs_array[0][0], (Zb2[1][0])**0.5 + bs_array[0][1],(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 5] = np.array([-(Zb2[0][0])**0.5 + bs_array[0][0], -(Zb2[1][0])**0.5 + bs_array[0][1],(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 6] = np.array([-(Zb2[0][0])**0.5 + bs_array[0][0], (Zb2[1][0])**0.5 + bs_array[0][1],-(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        WLS2_far[0:3][:, 7] = np.array([-(Zb2[0][0])**0.5 + bs_array[0][0], -(Zb2[1][0])**0.5 + bs_array[0][1],-(Zb2[2][0])**0.5 + bs_array[0][2]]).T
        for i in range(p - 1):
            for j in range(1, 1, p - 1):
                Rems[j - 1][i] = np.sqrt((bs_array[j][0] - WLS2_far[0][i]) ** 2 + (bs_array[j][1] - WLS2_far[1][i]) ** 2 + (bs_array[j][2] - WLS2_far[2][i]) ** 2) - np.sqrt((bs_array[0][0] - WLS2_far[0][i]) ** 2 + (bs_array[0][1] - WLS2_far[1][i]) ** 2 + (bs_array[0][3] - WLS2_far[2][i]) ** 2)


        for i in range(p - 1):
            num = 0
            for j in range(1, 1, p - 1):
                if np.sign(Rems[j - 1][i]) * np.sign(r[j - 1][i]) == 1:
                    num = num + 1
            if num >= (p-1):
                WLS2_far_out = WLS2_far[0:3][:,i]
        S_list = []
        S_list.append(WLS1_near.T.tolist()[0])
        S_list.append(WLS1_far.T.tolist()[0])
        S_list.append(WLS2_near_out.T.tolist()[0])
        S_list.append(WLS2_far_out.T.tolist()[0])
        tmpSs = [self.eucliDist(c, self.target_pos) for c in S_list]
        
        tmp_dict = {}
        for j in range(0,len(S_list)):
            
            tmp_dict[tmpSs[j]] = S_list[j]
        


        tmpSs.sort()
        sss = tmpSs[0]
        toda_pos_record = tmp_dict[sss]
       
        self.pos_list_record_tdoa.append(toda_pos_record)
        
        self.real_poses.append(self.target_pos)
        self.tdoa_poses.append(toda_pos_record)
       
        reward = -tmpSs[0]
        
        return reward
