import tkinter as tk
from config import DEFAULT_MOBILE_COUNT, MAX_MOBILE_COUNT, DEFAULT_SPEED, MAX_SPEED


class ControlPanel:
    def __init__(self, parent, simulator):
        self.simulator = simulator
        self.frame = tk.Frame(parent, bg="#16213e", padx=10, pady=10)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._create_controls()

    def _create_controls(self):
        tk.Label(self.frame, text="Speed:", bg="#16213e", fg="white", 
                font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.speed_slider = tk.Scale(
            self.frame, from_=1, to=MAX_SPEED, orient=tk.HORIZONTAL,
            bg="#0f3460", fg="white", highlightthickness=0,
            command=self.simulator.update_speed, length=150
        )
        self.speed_slider.set(DEFAULT_SPEED)
        self.speed_slider.grid(row=0, column=1, padx=5)

        tk.Label(self.frame, text="Users:", bg="#16213e", fg="white",
                font=("Arial", 10)).grid(row=0, column=2, padx=5)
        self.mobile_slider = tk.Scale(
            self.frame, from_=1, to=MAX_MOBILE_COUNT, orient=tk.HORIZONTAL,
            bg="#0f3460", fg="white", highlightthickness=0,
            command=self.simulator.update_mobile_count, length=150
        )
        self.mobile_slider.set(DEFAULT_MOBILE_COUNT)
        self.mobile_slider.grid(row=0, column=3, padx=5)

        self.pause_button = tk.Button(
            self.frame, text="Pause", command=self.simulator.toggle_pause,
            bg="#e94560", fg="white", padx=15, font=("Arial", 10, "bold")
        )
        self.pause_button.grid(row=0, column=4, padx=10)

        self.coverage_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.frame, text="Coverage", variable=self.coverage_var,
            bg="#16213e", fg="white", selectcolor="#0f3460",
            font=("Arial", 9), command=self.simulator.draw
        ).grid(row=0, column=5, padx=5)

        self.grid_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.frame, text="Grid", variable=self.grid_var,
            bg="#16213e", fg="white", selectcolor="#0f3460",
            font=("Arial", 9), command=self.simulator.draw
        ).grid(row=0, column=6, padx=5)

        self.rb_panel_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.frame, text="RB Panel", variable=self.rb_panel_var,
            bg="#16213e", fg="#00d4ff", selectcolor="#0f3460",
            font=("Arial", 9, "bold"), command=self.simulator.draw
        ).grid(row=0, column=7, padx=5)

        tk.Label(
            self.frame, text="VDN Model | 3 Agents | 9 RBs",
            bg="#16213e", fg="#c0e218", font=("Arial", 9, "bold")
        ).grid(row=0, column=8, padx=15)