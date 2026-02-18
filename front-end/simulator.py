import tkinter as tk
from config import (
    CANVAS_WIDTH, CANVAS_HEIGHT, BACKGROUND_COLOR,
    BASE_STATIONS, DEFAULT_MOBILE_COUNT, DEFAULT_SPEED, UPDATE_INTERVAL
)
from mobile import Mobile
from renderer import Renderer
from controls import ControlPanel
from vdn_model import VDNModel


class BaseStationSimulator:
    """Main simulator with VDN resource allocation."""

    def __init__(self, root):
        """Initialize the simulator."""
        self.root = root
        self.root.title("Base Station Signal Strength Simulator - VDN Resource Allocation")
        self.root.configure(bg="#16213e")

        # Create canvas
        self.canvas = tk.Canvas(
            root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg=BACKGROUND_COLOR, highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Initialize VDN model
        self.vdn_model = VDNModel()

        # Initialize components
        self.renderer = Renderer(self.canvas)
        self.control_panel = ControlPanel(root, self)

        # Simulation state
        self.base_stations = BASE_STATIONS
        self.mobiles = []
        self.speed = DEFAULT_SPEED
        self.paused = False
        self.allocation = {}  # Resource block allocation
        self.frame_count = 0

        # Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # Wait for canvas to be drawn before initializing
        self.root.after(100, self.initialize_simulation)

    def initialize_simulation(self):
        """Initialize the simulation after canvas is ready."""
        self.initialize_mobiles(DEFAULT_MOBILE_COUNT)
        self.update_allocation()
        self.animate()

    def on_canvas_resize(self, event):
        """Handle canvas resize events."""
        canvas_width = event.width
        canvas_height = event.height

        for mobile in self.mobiles:
            mobile.update_canvas_size(canvas_width, canvas_height)

    def get_canvas_dimensions(self):
        """Get current canvas dimensions."""
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def initialize_mobiles(self, count):
        """Initialize mobile devices."""
        width, height = self.get_canvas_dimensions()
        self.mobiles = [Mobile.create_random(self.speed, width, height) for _ in range(count)]

    def update_allocation(self):
        """Update resource block allocation using VDN model."""
        self.allocation = self.vdn_model.allocate_resources(self.mobiles, self.base_stations)

    def update_speed(self, value):
        """Update simulation speed."""
        self.speed = float(value)
        for mobile in self.mobiles:
            current_speed = (mobile.vx ** 2 + mobile.vy ** 2) ** 0.5
            if current_speed > 0:
                scale = self.speed / current_speed
                mobile.vx *= scale
                mobile.vy *= scale

    def update_mobile_count(self, value):
        """Update number of mobile devices."""
        count = int(value)
        current_count = len(self.mobiles)

        if count > current_count:
            width, height = self.get_canvas_dimensions()
            for _ in range(count - current_count):
                self.mobiles.append(Mobile.create_random(self.speed, width, height))
        elif count < current_count:
            self.mobiles = self.mobiles[:count]

        # Update allocation when mobile count changes
        self.update_allocation()

    def toggle_pause(self):
        """Toggle simulation pause state."""
        self.paused = not self.paused
        self.control_panel.pause_button.config(
            text="Resume" if self.paused else "Pause"
        )

    def update_positions(self):
        """Update positions and reallocate resources."""
        if not self.paused:
            for mobile in self.mobiles:
                mobile.update_position()

            # Reallocate resources every 10 frames (adaptive allocation)
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.update_allocation()

    def draw(self):
        """Draw the entire scene with RB allocation."""
        self.renderer.clear()

        # Draw grid if enabled
        if self.control_panel.grid_var.get():
            self.renderer.draw_grid()

        # Draw coverage circles if enabled
        if self.control_panel.coverage_var.get():
            self.renderer.draw_coverage_circles(self.base_stations)

        # Draw signal lines with allocation colors
        self.renderer.draw_signal_lines(self.base_stations, self.mobiles, self.allocation)

        # Draw base stations
        self.renderer.draw_base_stations(self.base_stations)

        # Draw mobiles with allocation info
        self.renderer.draw_mobiles(self.mobiles, self.base_stations, self.allocation)

        # Draw RB allocation panel
        if self.control_panel.rb_panel_var.get():
            self.renderer.draw_rb_allocation_panel(self.allocation, self.mobiles)

    def animate(self):
        """Main animation loop."""
        self.update_positions()
        self.draw()
        self.root.after(UPDATE_INTERVAL, self.animate)