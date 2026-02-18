from config import (
    BACKGROUND_COLOR, GRID_COLOR, BASE_STATION_COLOR, COVERAGE_CIRCLE_COLOR,
    BASE_STATION_RADIUS, BASE_STATION_RANGE, GRID_SIZE, SIGNAL_LINE_ALPHA,
    RB_COLORS, RB_DISPLAY_WIDTH, RB_DISPLAY_PADDING, RB_BAR_HEIGHT, RB_BAR_SPACING,
    NUM_RESOURCE_BLOCKS
)
from utils import calculate_distance, calculate_signal_strength, get_line_width, blend_color


class Renderer:
    """Handles all drawing operations including RB allocation visualization."""

    def __init__(self, canvas):
        """Initialize the renderer."""
        self.canvas = canvas

    def clear(self):
        """Clear the canvas."""
        self.canvas.delete("all")

    def draw_grid(self):
        """Draw background grid that adjusts to canvas size."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for x in range(0, canvas_width, GRID_SIZE):
            self.canvas.create_line(x, 0, x, canvas_height, fill=GRID_COLOR, width=1)

        for y in range(0, canvas_height, GRID_SIZE):
            self.canvas.create_line(0, y, canvas_width, y, fill=GRID_COLOR, width=1)

    def draw_coverage_circles(self, base_stations):
        """Draw coverage circles for base stations."""
        for bs in base_stations:
            x, y = bs["x"], bs["y"]
            self.canvas.create_oval(
                x - BASE_STATION_RANGE, y - BASE_STATION_RANGE,
                x + BASE_STATION_RANGE, y + BASE_STATION_RANGE,
                outline=COVERAGE_CIRCLE_COLOR, width=2, dash=(5, 5)
            )

    def draw_signal_lines(self, base_stations, mobiles, allocation):
        """Draw signal lines with RB allocation colors."""
        for i, mobile in enumerate(mobiles):
            if i in allocation:
                rb_index, agent_id = allocation[i]
                line_color = RB_COLORS[rb_index % len(RB_COLORS)]

                # Draw line to assigned base station
                bs = base_stations[agent_id]
                distance = calculate_distance(mobile.x, mobile.y, bs["x"], bs["y"])
                signal_strength = calculate_signal_strength(distance)

                if signal_strength > 0:
                    line_width = get_line_width(signal_strength)
                    color = blend_color(line_color, SIGNAL_LINE_ALPHA * signal_strength / 100)

                    self.canvas.create_line(
                        bs["x"], bs["y"], mobile.x, mobile.y,
                        fill=color, width=line_width
                    )

    def draw_base_stations(self, base_stations):
        """Draw base stations with agent IDs."""
        for bs in base_stations:
            x, y, name = bs["x"], bs["y"], bs["name"]
            agent_id = bs["agent_id"]

            # Draw base station circle
            self.canvas.create_oval(
                x - BASE_STATION_RADIUS, y - BASE_STATION_RADIUS,
                x + BASE_STATION_RADIUS, y + BASE_STATION_RADIUS,
                fill=BASE_STATION_COLOR, outline="white", width=2
            )

            # Draw label with agent ID
            self.canvas.create_text(
                x, y - BASE_STATION_RADIUS - 15,
                text=f"{name}\nAgent {agent_id}",
                fill="white", font=("Arial", 9, "bold")
            )

    def draw_mobiles(self, mobiles, base_stations, allocation):
        """Draw mobile devices with RB allocation info."""
        for i, mobile in enumerate(mobiles):
            # Get allocation info
            if i in allocation:
                rb_index, agent_id = allocation[i]
                mobile_color = RB_COLORS[rb_index % len(RB_COLORS)]
            else:
                mobile_color = mobile.color

            # Draw mobile circle
            self.canvas.create_oval(
                mobile.x - mobile.radius, mobile.y - mobile.radius,
                mobile.x + mobile.radius, mobile.y + mobile.radius,
                fill=mobile_color, outline="white", width=2
            )

            # Display allocation info
            if i in allocation:
                rb_index, agent_id = allocation[i]
                distance = calculate_distance(mobile.x, mobile.y, 
                                             base_stations[agent_id]["x"], 
                                             base_stations[agent_id]["y"])
                signal = calculate_signal_strength(distance)

                self.canvas.create_text(
                    mobile.x, mobile.y + mobile.radius + 12,
                    text=f"U{i}\nRB{rb_index}\n{int(signal)}%",
                    fill="white", font=("Arial", 7)
                )

    def draw_rb_allocation_panel(self, allocation, mobiles):
        """Draw resource block allocation panel on the right side."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        panel_x = canvas_width - RB_DISPLAY_WIDTH - RB_DISPLAY_PADDING
        panel_y = RB_DISPLAY_PADDING

        # Draw panel background
        self.canvas.create_rectangle(
            panel_x, panel_y,
            panel_x + RB_DISPLAY_WIDTH, canvas_height - RB_DISPLAY_PADDING,
            fill="#0f3460", outline="#e94560", width=2
        )

        # Title
        self.canvas.create_text(
            panel_x + RB_DISPLAY_WIDTH // 2, panel_y + 20,
            text="Resource Block Allocation",
            fill="white", font=("Arial", 11, "bold")
        )

        # Draw RB bars
        y_offset = panel_y + 50

        # Group by RB
        rb_assignments = {}
        for mobile_idx, (rb_idx, agent_id) in allocation.items():
            if rb_idx not in rb_assignments:
                rb_assignments[rb_idx] = []
            rb_assignments[rb_idx].append((mobile_idx, agent_id))

        for rb_idx in range(NUM_RESOURCE_BLOCKS):
            bar_y = y_offset + rb_idx * (RB_BAR_HEIGHT + RB_BAR_SPACING)

            # RB bar
            self.canvas.create_rectangle(
                panel_x + 10, bar_y,
                panel_x + RB_DISPLAY_WIDTH - 10, bar_y + RB_BAR_HEIGHT,
                fill=RB_COLORS[rb_idx], outline="white", width=1
            )

            # RB label and assignments
            if rb_idx in rb_assignments:
                users = [f"U{m}" for m, _ in rb_assignments[rb_idx]]
                agent = rb_assignments[rb_idx][0][1]
                text = f"RB{rb_idx} (A{agent}): {', '.join(users)}"
            else:
                text = f"RB{rb_idx}: Unassigned"

            self.canvas.create_text(
                panel_x + RB_DISPLAY_WIDTH // 2, bar_y + RB_BAR_HEIGHT // 2,
                text=text, fill="white", font=("Arial", 8, "bold")
            )

        # Model info
        info_y = y_offset + NUM_RESOURCE_BLOCKS * (RB_BAR_HEIGHT + RB_BAR_SPACING) + 30
        self.canvas.create_text(
            panel_x + RB_DISPLAY_WIDTH // 2, info_y,
            text="VDN Model Active\n3 Agents | 9 RBs | 10 Users",
            fill="#00d4ff", font=("Arial", 9)
        )