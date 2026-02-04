# Canvas dimensions
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# Colors
BACKGROUND_COLOR = "#1a1a2e"
GRID_COLOR = "#16213e"
BASE_STATION_COLOR = "#e94560"
COVERAGE_CIRCLE_COLOR = "#0f3460"

# Base station parameters
BASE_STATIONS = [
    {"x": 200, "y": 200, "name": "BS1"},
    {"x": 600, "y": 200, "name": "BS2"},
    {"x": 400, "y": 450, "name": "BS3"}
]
BASE_STATION_RADIUS = 10
BASE_STATION_RANGE = 250

# Mobile device colors
MOBILE_COLORS = ["#00d4ff", "#ff6b9d", "#c0e218", "#ffa500", "#9d4edd"]

# Simulation settings
DEFAULT_MOBILE_COUNT = 5
MAX_MOBILE_COUNT = 20
DEFAULT_SPEED = 2
MAX_SPEED = 10
UPDATE_INTERVAL = 50  

# Physics settings
MOBILE_RADIUS = 5
BOUNCE_DAMPING = 1.0

# Visual settings
GRID_SIZE = 50
MIN_LINE_WIDTH = 0.5
MAX_LINE_WIDTH = 3.0
SIGNAL_LINE_ALPHA = 0.6
