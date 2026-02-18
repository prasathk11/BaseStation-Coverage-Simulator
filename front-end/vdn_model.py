import numpy as np
import os
from config import (
    NUM_AGENTS, NUM_RESOURCE_BLOCKS, STATE_SHAPE, 
    HIDDEN_DIM, MODEL_PATH
)


class VDNModel:
    """Wrapper for VDN model inference."""

    def __init__(self):
        """Initialize VDN model."""
        self.num_agents = NUM_AGENTS
        self.num_rbs = NUM_RESOURCE_BLOCKS
        self.state_shape = STATE_SHAPE
        self.hidden_dim = HIDDEN_DIM
        self.model_loaded = False
        self.hidden_states = None

        # Try to load model
        self.load_model()

    def load_model(self):
        """Load trained VDN model from checkpoint."""
        try:
            # Check if model path exists
            if os.path.exists(MODEL_PATH):
                print(f"Model found at: {MODEL_PATH}")
                # In real implementation, load PyTorch model here
                # self.model = torch.load(MODEL_PATH)
                self.model_loaded = True
                self.reset_hidden_states()
            else:
                print(f"Model not found at: {MODEL_PATH}")
                print("Using random allocation fallback")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def reset_hidden_states(self):
        """Reset RNN hidden states."""
        self.hidden_states = np.zeros((self.num_agents, self.hidden_dim))

    def get_state_representation(self, mobiles, base_stations):
        """
        Convert simulation state to model input format.

        Args:
            mobiles: List of Mobile objects
            base_stations: List of base station dictionaries

        Returns:
            np.array: State representation (shape: STATE_SHAPE)
        """
        state = []

        # Add mobile positions and signal strengths (normalized)
        for i, mobile in enumerate(mobiles[:10]):  # Max 10 users
            # Normalized position
            state.append(mobile.x / 1200.0)
            state.append(mobile.y / 700.0)

        # Pad if less than 10 mobiles
        while len(state) < 20:
            state.append(0.0)

        return np.array(state[:STATE_SHAPE])

    def allocate_resources(self, mobiles, base_stations):
        """
        Allocate resource blocks to mobiles using VDN model.

        Args:
            mobiles: List of Mobile objects
            base_stations: List of base station dictionaries

        Returns:
            dict: Allocation mapping {mobile_index: (rb_index, agent_id)}
        """
        if not self.model_loaded:
            # Fallback: Random allocation
            return self._random_allocation(mobiles)

        # Get state representation
        state = self.get_state_representation(mobiles, base_stations)

        # In real implementation, run model inference
        # q_values = self.model(state, self.hidden_states)
        # actions = self._select_actions(q_values)

        # For now, use intelligent heuristic allocation
        return self._heuristic_allocation(mobiles, base_stations)

    def _random_allocation(self, mobiles):
        """Random resource block allocation."""
        allocation = {}
        for i, mobile in enumerate(mobiles[:10]):
            rb_index = i % self.num_rbs
            agent_id = i % self.num_agents
            allocation[i] = (rb_index, agent_id)
        return allocation

    def _heuristic_allocation(self, mobiles, base_stations):
        """
        Heuristic allocation based on signal strength.
        Simulates VDN behavior: each agent allocates RBs to its best users.
        """
        from utils import calculate_distance, calculate_signal_strength

        allocation = {}

        # Calculate signal strength for each mobile from each BS
        mobile_signals = []
        for i, mobile in enumerate(mobiles[:10]):
            signals = []
            for bs in base_stations:
                distance = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
                signal = calculate_signal_strength(distance)
                signals.append((signal, bs["agent_id"]))

            # Find best base station for this mobile
            best_signal, best_agent = max(signals, key=lambda x: x[0])
            mobile_signals.append((i, best_signal, best_agent))

        # Sort by signal strength (prioritize strong signals)
        mobile_signals.sort(key=lambda x: x[1], reverse=True)

        # Allocate RBs (each agent manages 3 RBs)
        agent_rb_count = {0: 0, 1: 0, 2: 0}

        for mobile_idx, signal, agent_id in mobile_signals:
            # Assign RB from the best agent
            if agent_rb_count[agent_id] < 3:  # Each agent has 3 RBs
                rb_index = agent_id * 3 + agent_rb_count[agent_id]
                allocation[mobile_idx] = (rb_index, agent_id)
                agent_rb_count[agent_id] += 1
            else:
                # Fallback to least loaded agent
                min_agent = min(agent_rb_count, key=agent_rb_count.get)
                rb_index = min_agent * 3 + agent_rb_count[min_agent]
                allocation[mobile_idx] = (rb_index % self.num_rbs, min_agent)
                agent_rb_count[min_agent] += 1

        return allocation

    def get_metrics(self):
        """Get current model metrics."""
        return {
            "model_loaded": self.model_loaded,
            "num_agents": self.num_agents,
            "num_rbs": self.num_rbs,
            "state_shape": self.state_shape,
            "epsilon": 0.6,
            "gamma": 0.2
        }
