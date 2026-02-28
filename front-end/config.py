# Canvas dimensions
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 700

# Colors
BACKGROUND_COLOR = "#1a1a2e"
GRID_COLOR = "#16213e"
BASE_STATION_COLOR = "#e94560"
COVERAGE_CIRCLE_COLOR = "#0f3460"

# Replace hardcoded BASE_STATIONS with ratio-based positions
BASE_STATIONS = [
    {"rx": 0.25, "ry": 0.35, "name": "BS1", "agent_id": 0},
    {"rx": 0.75, "ry": 0.35, "name": "BS2", "agent_id": 1},
    {"rx": 0.50, "ry": 0.65, "name": "BS3", "agent_id": 2}
]

BASE_STATION_RADIUS = 12
BASE_STATION_RANGE = 380

# Mobile device colors
MOBILE_COLORS = ["#00d4ff", "#ff6b9d", "#c0e218", "#ffa500", "#9d4edd", 
                 "#ff006e", "#8338ec", "#3a86ff", "#06ffa5", "#ffbe0b"]

# Resource Block colors
RB_COLORS = ["#ff0054", "#ff6b35", "#f7b801", "#6a994e", "#0077b6",
             "#7209b7", "#f72585", "#4361ee", "#06d6a0"]

# Simulation settings
DEFAULT_MOBILE_COUNT = 10  # 10 users as per dt_rb_allocation_10u
MAX_MOBILE_COUNT = 15
DEFAULT_SPEED = 1.5
MAX_SPEED = 10
UPDATE_INTERVAL = 100  # milliseconds

# PyMARL2 VDN Model Parameters
NUM_AGENTS = 3
NUM_RESOURCE_BLOCKS = 9
STATE_SHAPE = 20
HIDDEN_DIM = 128
LEARNING_RATE = 0.0001
GAMMA = 0.2
EPSILON = 0.6

# Model path
MODEL_PATH = "/Users/prasath/Project/Personal Project/BaseStation-Coverage-Simulator/back-end/pymarl2-master/results/models/vdn_env=dt_rb_allocation_10u_rb_9_eps_0.6__2026-02-18_12-10-58"

# Physics settings
MOBILE_RADIUS = 6
BOUNCE_DAMPING = 1.0

# Visual settings
GRID_SIZE = 50
MIN_LINE_WIDTH = 0.5
MAX_LINE_WIDTH = 4.0
SIGNAL_LINE_ALPHA = 0.7

# Resource Block Allocation Display
RB_DISPLAY_WIDTH = 250
RB_DISPLAY_PADDING = 20
RB_BAR_HEIGHT = 25
RB_BAR_SPACING = 8