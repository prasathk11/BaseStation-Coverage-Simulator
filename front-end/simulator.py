import tkinter as tk
from config import (
    CANVAS_WIDTH, CANVAS_HEIGHT, BACKGROUND_COLOR,
    BASE_STATIONS, DEFAULT_MOBILE_COUNT, DEFAULT_SPEED, UPDATE_INTERVAL
)
from mobile import Mobile
from renderer import Renderer
from controls import ControlPanel


class BaseStationSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Base Station Signal Strength Simulator")
        self.root.configure(bg="#16213e")

        # Create canvas
        self.canvas = tk.Canvas(
            root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg=BACKGROUND_COLOR, highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Initialize components
        self.renderer = Renderer(self.canvas)
        self.control_panel = ControlPanel(root, self)

        # Simulation state
        self.base_stations = BASE_STATIONS
        self.mobiles = []
        self.speed = DEFAULT_SPEED
        self.paused = False
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        self.root.after(100, self.initialize_simulation)

    def initialize_simulation(self):
        self.initialize_mobiles(DEFAULT_MOBILE_COUNT)
        self.animate()

    def on_canvas_resize(self, event):
        canvas_width = event.width
        canvas_height = event.height

        for mobile in self.mobiles:
            mobile.update_canvas_size(canvas_width, canvas_height)

    def get_canvas_dimensions(self):
        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def initialize_mobiles(self, count):
        width, height = self.get_canvas_dimensions()
        self.mobiles = [Mobile.create_random(self.speed, width, height) for _ in range(count)]

    def update_speed(self, value):
        self.speed = float(value)
        for mobile in self.mobiles:
            # Scale velocity
            current_speed = (mobile.vx ** 2 + mobile.vy ** 2) ** 0.5
            if current_speed > 0:
                scale = self.speed / current_speed
                mobile.vx *= scale
                mobile.vy *= scale

    def update_mobile_count(self, value):
        count = int(value)
        current_count = len(self.mobiles)

        if count > current_count:
            # Add new mobiles
            width, height = self.get_canvas_dimensions()
            for _ in range(count - current_count):
                self.mobiles.append(Mobile.create_random(self.speed, width, height))
        elif count < current_count:
            # Remove excess mobiles
            self.mobiles = self.mobiles[:count]

    def toggle_pause(self):
        self.paused = not self.paused
        self.control_panel.pause_button.config(
            text="Resume" if self.paused else "Pause"
        )

    def update_positions(self):
        if not self.paused:
            for mobile in self.mobiles:
                mobile.update_position()

    def draw(self):
        self.renderer.clear()

        if self.control_panel.grid_var.get():
            self.renderer.draw_grid()

        if self.control_panel.coverage_var.get():
            self.renderer.draw_coverage_circles(self.base_stations)

        self.renderer.draw_signal_lines(self.base_stations, self.mobiles)

        self.renderer.draw_base_stations(self.base_stations)

        self.renderer.draw_mobiles(self.mobiles, self.base_stations)

    def animate(self):
        """Main animation loop."""
        self.update_positions()
        self.draw()
        self.root.after(UPDATE_INTERVAL, self.animate)
