from config import (
    BACKGROUND_COLOR, GRID_COLOR,
    BASE_STATION_COLOR, COVERAGE_CIRCLE_COLOR, BASE_STATION_RADIUS,
    BASE_STATION_RANGE, GRID_SIZE, SIGNAL_LINE_ALPHA
)
from utils import calculate_distance, calculate_signal_strength, get_line_width, blend_color


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

    def clear(self):
        self.canvas.delete("all")

    def draw_grid(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for x in range(0, canvas_width, GRID_SIZE):
            self.canvas.create_line(x, 0, x, canvas_height, fill=GRID_COLOR, width=1)
        for y in range(0, canvas_height, GRID_SIZE):
            self.canvas.create_line(0, y, canvas_width, y, fill=GRID_COLOR, width=1)

    def draw_coverage_circles(self, base_stations):
        for bs in base_stations:
            x, y = bs["x"], bs["y"]
            self.canvas.create_oval(
                x - BASE_STATION_RANGE, y - BASE_STATION_RANGE,
                x + BASE_STATION_RANGE, y + BASE_STATION_RANGE,
                outline=COVERAGE_CIRCLE_COLOR, width=2, dash=(5, 5)
            )

    def draw_signal_lines(self, base_stations, mobiles):
        for mobile in mobiles:
            for bs in base_stations:
                distance = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
                signal_strength = calculate_signal_strength(distance)

                if signal_strength > 0:
                    line_width = get_line_width(signal_strength)
                    color = blend_color(mobile.color, SIGNAL_LINE_ALPHA * signal_strength / 100)

                    self.canvas.create_line(
                        bs["x"], bs["y"], mobile.x, mobile.y,
                        fill=color, width=line_width
                    )

    def draw_base_stations(self, base_stations):
        for bs in base_stations:
            x, y, name = bs["x"], bs["y"], bs["name"]

            # Draw base station circle
            self.canvas.create_oval(
                x - BASE_STATION_RADIUS, y - BASE_STATION_RADIUS,
                x + BASE_STATION_RADIUS, y + BASE_STATION_RADIUS,
                fill=BASE_STATION_COLOR, outline="white", width=2
            )

            # Draw base station label
            self.canvas.create_text(
                x, y - BASE_STATION_RADIUS - 10,
                text=name, fill="white", font=("Arial", 10, "bold")
            )

    def draw_mobiles(self, mobiles, base_stations):
        for mobile in mobiles:
            self.canvas.create_oval(
                mobile.x - mobile.radius, mobile.y - mobile.radius,
                mobile.x + mobile.radius, mobile.y + mobile.radius,
                fill=mobile.color, outline="white", width=1
            )

            max_signal = 0
            best_bs = None

            for bs in base_stations:
                distance = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
                signal = calculate_signal_strength(distance)
                if signal > max_signal:
                    max_signal = signal
                    best_bs = bs["name"]

            if max_signal > 0:
                self.canvas.create_text(
                    mobile.x, mobile.y + mobile.radius + 10,
                    text=f"{int(max_signal)}%\n{best_bs}",
                    fill="white", font=("Arial", 8)
                )
