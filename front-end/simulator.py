import tkinter as tk
from config import (
    CANVAS_WIDTH, CANVAS_HEIGHT, BACKGROUND_COLOR,
    BASE_STATIONS, DEFAULT_MOBILE_COUNT, DEFAULT_SPEED, UPDATE_INTERVAL,
    NUM_RESOURCE_BLOCKS, RB_COLORS, NUM_AGENTS
)
from mobile import Mobile
from renderer import Renderer
from controls import ControlPanel
from vdn_model import VDNModel


class BaseStationSimulator:
    """Main simulator with VDN resource allocation."""

    # Base station positions as ratios of canvas size (so they scale on resize)
    BS_RATIOS = [
        {"rx": 0.25, "ry": 0.38, "name": "BS1", "agent_id": 0},
        {"rx": 0.75, "ry": 0.38, "name": "BS2", "agent_id": 1},
        {"rx": 0.50, "ry": 0.68, "name": "BS3", "agent_id": 2},
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Base Station Signal Strength Simulator - VDN Resource Allocation")
        self.root.configure(bg="#16213e")

        # ── Outer frame holds canvas + sidebar ──────────────────────────────
        self.outer_frame = tk.Frame(root, bg="#16213e")
        self.outer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        # ── Simulation canvas (left, expands) ───────────────────────────────
        self.canvas = tk.Canvas(
            self.outer_frame,
            bg=BACKGROUND_COLOR,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── RB Sidebar (right, fixed 260 px) ────────────────────────────────
        self.sidebar = tk.Frame(
            self.outer_frame,
            bg="#0f3460",
            width=260,
            highlightbackground="#e94560",
            highlightthickness=2
        )
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.sidebar.pack_propagate(False)   # keep fixed width

        self._build_sidebar()

        # ── VDN model ────────────────────────────────────────────────────────
        self.vdn_model = VDNModel()

        # ── Renderer (canvas only, no RB panel drawing) ──────────────────────
        self.renderer = Renderer(self.canvas)

        # ── Control panel (bottom bar) ───────────────────────────────────────
        self.control_panel = ControlPanel(root, self)

        # ── Simulation state ─────────────────────────────────────────────────
        self.mobiles = []
        self.speed = DEFAULT_SPEED
        self.paused = False
        self.allocation = {}
        self.frame_count = 0

        # Bind resize
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Kick off after layout is ready
        self.root.after(150, self.initialize_simulation)

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        """Create static sidebar widgets; labels updated each frame."""
        tk.Label(
            self.sidebar, text="Resource Block Allocation",
            bg="#0f3460", fg="white", font=("Arial", 11, "bold")
        ).pack(pady=(12, 8))

        self.rb_labels = []
        for i in range(NUM_RESOURCE_BLOCKS):
            lbl = tk.Label(
                self.sidebar,
                text=f"RB{i}: Unassigned",
                bg=RB_COLORS[i % len(RB_COLORS)],
                fg="white",
                font=("Arial", 9, "bold"),
                width=28,
                relief=tk.FLAT,
                pady=5
            )
            lbl.pack(pady=3, padx=10)
            self.rb_labels.append(lbl)

        tk.Frame(self.sidebar, bg="#0f3460", height=10).pack()

        self.model_info_label = tk.Label(
            self.sidebar,
            text="VDN Model Active\n3 Agents | 9 RBs | 10 Users",
            bg="#0f3460", fg="#00d4ff",
            font=("Arial", 9),
            justify=tk.CENTER
        )
        self.model_info_label.pack(pady=6)

    def update_sidebar(self):
        """Refresh RB label text each animation frame."""
        rb_assignments = {}
        for mobile_idx, (rb_idx, agent_id) in self.allocation.items():
            rb_assignments.setdefault(rb_idx, []).append((mobile_idx, agent_id))

        for rb_idx, lbl in enumerate(self.rb_labels):
            if rb_idx in rb_assignments:
                users = [f"U{m}" for m, _ in rb_assignments[rb_idx]]
                agent = rb_assignments[rb_idx][0][1]
                lbl.config(text=f"RB{rb_idx} (A{agent}): {', '.join(users)}")
            else:
                lbl.config(text=f"RB{rb_idx}: Unassigned")

        n_users = len(self.mobiles)
        self.model_info_label.config(
            text=f"VDN Model Active\n{NUM_AGENTS} Agents | {NUM_RESOURCE_BLOCKS} RBs | {n_users} Users"
        )

    # ── BS position helpers ───────────────────────────────────────────────────

    def get_base_stations(self):
        """Return BS positions scaled to current canvas dimensions."""
        w, h = self.get_canvas_dimensions()
        result = []
        for bs in self.BS_RATIOS:
            result.append({
                "x": int(bs["rx"] * w),
                "y": int(bs["ry"] * h),
                "name": bs["name"],
                "agent_id": bs["agent_id"]
            })
        return result

    def get_canvas_dimensions(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        # Fallback to config defaults before first Configure event
        if w < 2:
            w = CANVAS_WIDTH
        if h < 2:
            h = CANVAS_HEIGHT
        return w, h

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize_simulation(self):
        self.initialize_mobiles(DEFAULT_MOBILE_COUNT)
        self.update_allocation()
        self.animate()

    def on_canvas_resize(self, event):
        for mobile in self.mobiles:
            mobile.update_canvas_size(event.width, event.height)

    def initialize_mobiles(self, count):
        w, h = self.get_canvas_dimensions()
        self.mobiles = [Mobile.create_random(self.speed, w, h) for _ in range(count)]

    def update_allocation(self):
        self.allocation = self.vdn_model.allocate_resources(
            self.mobiles, self.get_base_stations()
        )

    # ── Controls ──────────────────────────────────────────────────────────────

    def update_speed(self, value):
        self.speed = float(value)
        for mobile in self.mobiles:
            spd = (mobile.vx ** 2 + mobile.vy ** 2) ** 0.5
            if spd > 0:
                scale = self.speed / spd
                mobile.vx *= scale
                mobile.vy *= scale

    def update_mobile_count(self, value):
        count = int(value)
        current = len(self.mobiles)
        if count > current:
            w, h = self.get_canvas_dimensions()
            for _ in range(count - current):
                self.mobiles.append(Mobile.create_random(self.speed, w, h))
        elif count < current:
            self.mobiles = self.mobiles[:count]
        self.update_allocation()

    def toggle_pause(self):
        self.paused = not self.paused
        self.control_panel.pause_button.config(
            text="Resume" if self.paused else "Pause"
        )

    # ── Animation loop ────────────────────────────────────────────────────────

    def update_positions(self):
        if not self.paused:
            for mobile in self.mobiles:
                mobile.update_position()
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.update_allocation()

    def draw(self):
        bs = self.get_base_stations()
        self.renderer.clear()

        if self.control_panel.grid_var.get():
            self.renderer.draw_grid()

        if self.control_panel.coverage_var.get():
            self.renderer.draw_coverage_circles(bs)

        self.renderer.draw_signal_lines(bs, self.mobiles, self.allocation)
        self.renderer.draw_base_stations(bs)
        self.renderer.draw_mobiles(self.mobiles, bs, self.allocation)

        # Sidebar is a real tkinter widget — just refresh its labels
        self.update_sidebar()

    def animate(self):
        self.update_positions()
        self.draw()
        self.root.after(UPDATE_INTERVAL, self.animate)
