from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
import random

# This multi-agent controller shares parameters between agents
class SDMAC:
    def __init__(self, scheme, groups, args):
        pass
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, agent_id=None): 
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions = []
        for batch in avail_actions: 
            batch_actions = []
            for agent_actions in batch: 
                avail_action = th.nonzero(agent_actions)[0]
                if len(avail_action) == 0:
                    batch_actions.append(0)
                else:
                    batch_actions.append(random.choice(avail_action))
            chosen_actions.append(batch_actions)
        
        chosen_actions = np.array(chosen_actions)
        return th.tensor(chosen_actions)
        