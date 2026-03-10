import tkinter as tk
import math
from config import (
    BACKGROUND_COLOR, GRID_COLOR, BASE_STATION_COLOR, COVERAGE_CIRCLE_COLOR,
    BASE_STATION_RADIUS, BASE_STATION_RANGE, GRID_SIZE, SIGNAL_LINE_ALPHA,
    RB_COLORS
)
from utils import calculate_distance, calculate_signal_strength, get_line_width, blend_color

class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

    def clear(self):
        self.canvas.delete("all")

    def draw_grid(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        for x in range(0, w, GRID_SIZE):
            self.canvas.create_line(x, 0, x, h, fill=GRID_COLOR, width=1)
        for y in range(0, h, GRID_SIZE):
            self.canvas.create_line(0, y, w, y, fill=GRID_COLOR, width=1)


    def draw_coverage_circles(self, base_stations):
        for bs in base_stations:
            x, y = bs["x"], bs["y"]
            r = BASE_STATION_RANGE
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline=COVERAGE_CIRCLE_COLOR, width=2, dash=(5, 5)
            )


    def draw_signal_lines(self, base_stations, mobiles, allocation):
        import math
        OFFSET = 4  # perpendicular offset between the two parallel edges

        for i, mobile in enumerate(mobiles):
            if i not in allocation:
                continue
            rb_index, agent_id = allocation[i]
            line_color = RB_COLORS[rb_index % len(RB_COLORS)]
            bs = base_stations[agent_id]
            distance = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
            signal = calculate_signal_strength(distance)
            if signal <= 0:
                continue

            lw = get_line_width(signal)
            color = blend_color(line_color, SIGNAL_LINE_ALPHA * signal / 100)

            x1, y1 = bs["x"], bs["y"]
            x2, y2 = mobile.x, mobile.y

            # Perpendicular unit vector for offset
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1
            px, py = -dy / length * OFFSET, dx / length * OFFSET

            # ── Edge 1: BS → Mobile (arrow at mobile end) ──────────────────
            self.canvas.create_line(
                x1 + px, y1 + py, x2 + px, y2 + py,
                fill=color, width=lw,
                arrow=tk.LAST, arrowshape=(10, 12, 4)
            )

            # ── Edge 2: Mobile → BS (arrow at BS end) ──────────────────────
            self.canvas.create_line(
                x2 - px, y2 - py, x1 - px, y1 - py,
                fill=color, width=lw,
                arrow=tk.LAST, arrowshape=(10, 12, 4)
            )


    def draw_base_stations(self, base_stations):
        for bs in base_stations:
            x, y = bs["x"], bs["y"]
            r = BASE_STATION_RADIUS
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=BASE_STATION_COLOR, outline="white", width=2
            )
            self.canvas.create_text(
                x, y - r - 15,
                text=f"{bs['name']}\nAgent {bs['agent_id']}",
                fill="white", font=("Arial", 9, "bold")
            )


    def draw_mobiles(self, mobiles, base_stations, allocation):
        for i, mobile in enumerate(mobiles):
            if i in allocation:
                rb_index, agent_id = allocation[i]
                color = RB_COLORS[rb_index % len(RB_COLORS)]
            else:
                color = mobile.color

            r = mobile.radius
            self.canvas.create_oval(
                mobile.x - r, mobile.y - r,
                mobile.x + r, mobile.y + r,
                fill=color, outline="white", width=2
            )

            if i in allocation:
                rb_index, agent_id = allocation[i]
                bs = base_stations[agent_id]
                dist = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
                sig = calculate_signal_strength(dist)
                self.canvas.create_text(
                    mobile.x, mobile.y + r + 12,
                    text=f"U{i}\nRB{rb_index}\n{int(sig)}%",
                    fill="white", font=("Arial", 7)
                )