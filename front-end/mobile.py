import random
from config import MOBILE_RADIUS, MOBILE_COLORS, BOUNCE_DAMPING


class Mobile:
    def __init__(self, x, y, vx, vy, color, canvas_width, canvas_height):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = MOBILE_RADIUS
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def update_canvas_size(self, width, height):
        self.canvas_width = width
        self.canvas_height = height

    def update_position(self):
        self.x += self.vx
        self.y += self.vy

        if self.x - self.radius <= 0 or self.x + self.radius >= self.canvas_width:
            self.vx *= -BOUNCE_DAMPING
            self.x = max(self.radius, min(self.canvas_width - self.radius, self.x))

        if self.y - self.radius <= 0 or self.y + self.radius >= self.canvas_height:
            self.vy *= -BOUNCE_DAMPING
            self.y = max(self.radius, min(self.canvas_height - self.radius, self.y))

    @staticmethod
    def create_random(speed, canvas_width, canvas_height):
        x = random.randint(MOBILE_RADIUS, canvas_width - MOBILE_RADIUS)
        y = random.randint(MOBILE_RADIUS, canvas_height - MOBILE_RADIUS)
        vx = random.uniform(-speed, speed)
        vy = random.uniform(-speed, speed)
        color = random.choice(MOBILE_COLORS)
        return Mobile(x, y, vx, vy, color, canvas_width, canvas_height)
