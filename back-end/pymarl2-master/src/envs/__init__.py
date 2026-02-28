from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .uav import UafOrient
# from .dt_marl_env import DT_MARL
from .dt_marl_env_10u_smallrange import DT_MARL_10u_org
# from .dt_marl_env_10u_fixstart import DT_MARL_10u_fix
try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["uav"] = partial(env_fn, env=UafOrient)
# REGISTRY["dt_rb_allocation"] = partial(env_fn, env=DT_MARL)
REGISTRY["dt_rb_allocation_10u_org"] = partial(env_fn, env=DT_MARL_10u_org)
# REGISTRY["dt_rb_allocation_10u_fix"] = partial(env_fn, env=DT_MARL_10u_fix)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
