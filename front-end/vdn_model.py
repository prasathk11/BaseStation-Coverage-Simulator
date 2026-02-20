import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/Users/prasath/Project/Personal Project/BaseStation-Coverage-Simulator/back-end/models/vdn_env=dt_rb_allocation_10u_rb_9_eps_0.6__2026-02-18_12-10-58/30"
NUM_AGENTS  = 3
NUM_USERS   = 10
NUM_RBS     = 4          # n_rbs = len(RbInfo) = 4
INPUT_DIM   = 20         # 2 * num_ue
HIDDEN_DIM  = 128
N_ACTIONS   = 2048       # 2^11

# Real BS locations from the env (normalised coord space: MIN_X=-150, MAX_X=150)
BS_LOCS = [[-100, 0], [0, 0], [100, 0]]   # in env units
BS_COVERAGE = 60                            # env units


# ── Exact PyMARL2 RNNAgent ────────────────────────────────────────────────────
class RNNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.rnn = nn.GRUCell(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, N_ACTIONS)

    def forward(self, x, h):
        x = F.relu(self.fc1(x), inplace=True)
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h

    def init_hidden(self):
        return self.fc1.weight.new(1, HIDDEN_DIM).zero_()


# ── VDNModel wrapper ──────────────────────────────────────────────────────────
class VDNModel:
    """Loads trained VDN agents and runs real GRU inference."""

    def __init__(self):
        self.num_agents       = NUM_AGENTS
        self.num_rbs          = NUM_RBS
        self.agents           = []
        self.hidden_states    = []
        self.model_loaded     = False
        self.using_real_model = False
        self.inference_count  = 0
        self._load_model()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            agent_path = os.path.join(MODEL_PATH, "agent.th")
            if not os.path.exists(agent_path):
                print(f"✗ agent.th not found: {agent_path}")
                return

            state_dict = torch.load(agent_path, map_location="cpu")

            for i in range(NUM_AGENTS):
                agent = RNNAgent()
                agent.load_state_dict(state_dict)
                agent.eval()
                self.agents.append(agent)

            self._reset_hidden()
            self.model_loaded     = True
            self.using_real_model = True
            print(f"✓ agent.th loaded  |  input={INPUT_DIM}  hidden={HIDDEN_DIM}  actions={N_ACTIONS}")

        except Exception as e:
            print(f"✗ Model load failed: {e}")
            import traceback; traceback.print_exc()

    def _reset_hidden(self):
        self.hidden_states = [agent.init_hidden() for agent in self.agents]

    # ── Observation builder (mirrors get_obs_agent exactly) ───────────────────
    def _build_obs(self, agent_id, mobiles, canvas_w=1200, canvas_h=700):
        """
        Mirrors DT_MARL_10u_fix.get_obs_agent():
          - For each of 10 users: if within BS coverage → (x,y) else (0,0)
          - Normalise by /10
          - Returns tensor shape [1, 20]

        Coordinate mapping:
          Canvas (0..1200, 0..700) → Env (-150..150, -50..50)
        """
        bs_env = BS_LOCS[agent_id]   # e.g. [-100, 0]

        obs = []
        for i, mobile in enumerate(mobiles[:NUM_USERS]):
            # Map canvas coords → env coords
            env_x = (mobile.x / canvas_w) * 300 - 150   # [-150, 150]
            env_y = (mobile.y / canvas_h) * 100 - 50    # [-50,   50]

            # Check if user is within BS coverage (Chebyshev distance, same as env)
            dx = abs(env_x - bs_env[0])
            dy = abs(env_y - bs_env[1])
            in_range = (dx <= BS_COVERAGE) and (dy <= BS_COVERAGE)

            if in_range:
                obs.extend([env_x / 10.0, env_y / 10.0])
            else:
                obs.extend([0.0, 0.0])

        # Pad to INPUT_DIM if fewer than NUM_USERS mobiles
        while len(obs) < INPUT_DIM:
            obs.append(0.0)
        obs = obs[:INPUT_DIM]

        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 20]

    # ── Inference ─────────────────────────────────────────────────────────────
    def allocate_resources(self, mobiles, base_stations,
                           canvas_w=1200, canvas_h=700):
        """
        Run real GRU inference → decode 11-bit actions → build allocation dict.
        Returns: {mobile_idx: (rb_index, agent_id)}
        """
        self.inference_count += 1

        if not self.using_real_model:
            return self._heuristic_allocation(mobiles, base_stations)

        try:
            chosen_actions = []
            new_hidden     = []

            for i, agent in enumerate(self.agents):
                obs = self._build_obs(i, mobiles, canvas_w, canvas_h)
                with torch.no_grad():
                    q, h = agent(obs, self.hidden_states[i])

                # Mask unavailable actions (too many users selected)
                avail = self._get_avail_actions(i, mobiles, canvas_w, canvas_h)
                avail_t = torch.tensor(avail, dtype=torch.float32)
                q_flat = q.squeeze(0)          # [1, 2048] → [2048]
                q_masked = q_flat.clone()
                q_masked[avail_t == 0] = -1e9

                action = int(q_masked.argmax().item())
                chosen_actions.append(action)
                new_hidden.append(h)

            self.hidden_states = new_hidden

            allocation = self._decode_actions(chosen_actions, mobiles,
                                              base_stations, canvas_w, canvas_h)

            if self.inference_count % 30 == 0:
                bits = [format(a, "011b") for a in chosen_actions]
                print(f"MODEL #{self.inference_count}  "
                      f"actions={chosen_actions}  bits={bits}")

            return allocation

        except Exception as e:
            print(f"⚠ Inference error: {e}")
            import traceback; traceback.print_exc()
            return self._heuristic_allocation(mobiles, base_stations)

    # ── Available actions (mirrors get_avail_agent_actions) ───────────────────
    def _get_avail_actions(self, agent_id, mobiles, canvas_w, canvas_h):
        bs_env = BS_LOCS[agent_id]
        in_range = []
        for i, mobile in enumerate(mobiles[:NUM_USERS]):
            env_x = (mobile.x / canvas_w) * 300 - 150
            env_y = (mobile.y / canvas_h) * 100 - 50
            dx = abs(env_x - bs_env[0])
            dy = abs(env_y - bs_env[1])
            in_range.append((dx <= BS_COVERAGE) and (dy <= BS_COVERAGE))

        avail = np.ones(N_ACTIONS)
        for action in range(N_ACTIONS):
            bits = format(action, "011b")
            # Users not in range cannot be served
            for u in range(NUM_USERS):
                if bits[u] == "1" and not in_range[u]:
                    avail[action] = 0
                    break
            # Cannot serve more users than available RBs
            if avail[action] and sum(int(b) for b in bits) > NUM_RBS:
                avail[action] = 0
        return avail

    # ── Decode 11-bit actions → allocation dict ───────────────────────────────
    def _decode_actions(self, actions, mobiles, base_stations, canvas_w, canvas_h):
        """
        Each action is an 11-bit integer:
          bits[0..9] = users served by this BS
          bits[10]   = sync bit (ignored for visualisation)

        We assign RBs 0..3 per agent sequentially to served users.
        Global RB index = agent_id * NUM_RBS + local_rb
        """
        allocation   = {}
        user_claimed = {}   # user_idx → agent_id (first-come wins)

        for agent_id, action in enumerate(actions):
            bits = format(action, "011b")   # 11 chars, MSB first
            local_rb = 0
            for user_idx in range(NUM_USERS):
                if bits[user_idx] == "1":
                    if user_idx not in user_claimed and local_rb < NUM_RBS:
                        user_claimed[user_idx] = agent_id
                        global_rb = agent_id * NUM_RBS + local_rb
                        allocation[user_idx] = (global_rb % (NUM_AGENTS * NUM_RBS),
                                                agent_id)
                        local_rb += 1

        return allocation

    # ── Fallback heuristic ────────────────────────────────────────────────────
    def _heuristic_allocation(self, mobiles, base_stations):
        allocation     = {}
        agent_rb_count = {i: 0 for i in range(NUM_AGENTS)}

        for idx, mobile in enumerate(mobiles[:NUM_USERS]):
            best_agent = 0
            best_dist  = float("inf")
            for bs in base_stations:
                d = ((mobile.x - bs["x"])**2 + (mobile.y - bs["y"])**2)**0.5
                if d < best_dist:
                    best_dist  = d
                    best_agent = bs["agent_id"]

            if agent_rb_count[best_agent] < NUM_RBS:
                rb = best_agent * NUM_RBS + agent_rb_count[best_agent]
                agent_rb_count[best_agent] += 1
                allocation[idx] = (rb % (NUM_AGENTS * NUM_RBS), best_agent)

        return allocation

    # ── Status helpers ────────────────────────────────────────────────────────
    def get_status_string(self):
        if self.using_real_model:
            return f"GRU-VDN  |  Inferences: {self.inference_count}"
        return f"⚠ HEURISTIC  |  Calls: {self.inference_count}"

    def get_metrics(self):
        return {
            "model_loaded":     self.model_loaded,
            "using_real_model": self.using_real_model,
            "num_agents":       self.num_agents,
            "num_rbs":          self.num_rbs,
            "inference_count":  self.inference_count,
        }