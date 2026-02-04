import math
from config import BASE_STATION_RANGE, MIN_LINE_WIDTH, MAX_LINE_WIDTH


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_signal_strength(distance, max_range=BASE_STATION_RANGE):
    if distance > max_range:
        return 0
    return max(0, 100 * (1 - (distance / max_range) ** 2))


def get_line_width(signal_strength):
    return MIN_LINE_WIDTH + (signal_strength / 100) * (MAX_LINE_WIDTH - MIN_LINE_WIDTH)


def blend_color(color, alpha):
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

    bg_r, bg_g, bg_b = 26, 26, 46 

    r = int(r * alpha + bg_r * (1 - alpha))
    g = int(g * alpha + bg_g * (1 - alpha))
    b = int(b * alpha + bg_b * (1 - alpha))

    return f'#{r:02x}{g:02x}{b:02x}'
