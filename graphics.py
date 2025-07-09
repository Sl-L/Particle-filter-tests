# Example file showing a circle moving on screen
import pygame
import scipy
import random
import math

# pygame setup
pygame.init()
pygame.display.set_caption("Multilateration simulation")
screen = pygame.display.set_mode((1880, 900))

base = pygame.Surface((1880, 900))
map = base.subsurface((0, 100, 1880, 800))

clock = pygame.time.Clock()
running = True
dt = 0

class Node:
    def __init__(self,
                 pos: pygame.Vector2,
                 color: str,
                 radius: int,
                 surface: pygame.Surface,
                 ):
        
        self.pos = pos
        self.color = color
        self.radius = radius
        self.surface = surface
        self.surface_offset = surface.get_offset()
        self.pressed = False
        

    def check_press(self):
        mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
        local_mouse_pos = mouse_pos - self.surface_offset
        if self.pos.distance_to(local_mouse_pos) < self.radius * 1.5:
            self.pressed = True
            self.pos = local_mouse_pos
        
    def drag(self):
        if self.pressed:
            mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
            local_mouse_pos = mouse_pos - self.surface_offset
            self.pos = local_mouse_pos
    
    def drop(self):
        self.pressed = False

    def draw(self):
        pygame.draw.circle(self.surface, self.color, self.pos, self.radius)

class Beacon(Node):
    def __init__(self,
                 pos: pygame.Vector2,
                 color: str,
                 radius: int,
                 surface: pygame.Surface,
                 target: Node,
                 add_noise: bool,
                 noise: float = 1.0,
                 tx_power: float = 0.0,
                 path_loss_exponent: float = 2.0  # Free space path loss exponent
                 ):
        
        super().__init__(pos, color, radius, surface)
        self.target = target
        self.add_noise = add_noise
        self.noise = noise
        self.tx_power = tx_power
        self.path_loss_exponent = path_loss_exponent
        self.sigma_ln = ( math.log(10) / ( 10 * path_loss_exponent )) * self.noise  # Standard deviation for log-normal noise
    
    def _calculate_rssi(self) -> float:
        return self.tx_power - 10 * self.path_loss_exponent * math.log10(self.get_true_distance())

    def get_rssi(self):
        if self.add_noise:
            return round(self._calculate_rssi())
        else:
            return self._calculate_rssi()
    
    def get_true_distance(self) -> float:
        return self.pos.distance_to(self.target.pos)

    def get_distance(self) -> float:
        if not self.add_noise:
            return self.get_true_distance()
        else:
            # Calculate the distance with noise
            rssi = self.get_rssi()
            distance = 10 ** ((self.tx_power - rssi) / (10 * self.path_loss_exponent)) + random.lognormvariate(0, self.noise)
            return distance


class UI:
    def __init__(self, base: pygame.Surface, map: pygame.Surface):
        self.base = base
        self.map = map
        self.font = pygame.font.SysFont("Arial", 36)

    def draw_text(self, text: str, pos: pygame.Vector2, color: str = "black"):
        text_surface = self.font.render(text, True, color)
        self.base.blit(text_surface, (0, 0))
    
    def draw_circles_op(self, robot, beacons: list[Beacon], color: str = "black"):
        # RGBA color: (R, G, B, Alpha)
        if isinstance(color, str):
            rgb = pygame.Color(color)
        else:
            rgb = pygame.Color(*color)
        rgba = (rgb.r, rgb.g, rgb.b, 32)
        
        for beacon in beacons:
            if beacon.add_noise:
                for i in range(3):
                    i1 = scipy.stats.lognorm.interval(0.95 - 0.1*i, s=beacon.sigma_ln, scale=beacon.get_true_distance())
                    r1 = int(i1[0])
                    r2 = int(i1[1])

                    radius = ( r1 + r2 ) // 2
                    thickness = r2 - r1

                    diameter = radius * 2 + thickness * 2 + 4 # + 4 for padding
                    circle_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
                    center = (diameter // 2 , diameter // 2 )

                    pygame.draw.circle(circle_surface, rgba, center, radius + thickness // 2, width=thickness)

                    # Blit the circle surface to the main UI surface
                    blit_pos = beacon.pos - pygame.Vector2(diameter // 2, diameter // 2)
                    self.map.blit(circle_surface, blit_pos)
            
            else:
                # Calculate distance for circle radius
                thickness = int(beacon.noise)
                radius = int(beacon.get_true_distance())

                # Create a transparent surface large enough for the circle
                diameter = radius * 2 + thickness * 2 + 4 # + 4 for padding
                circle_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)

                # Circle center on the temp surface
                center = (diameter // 2 , diameter // 2 )

                # Draw the circle outline with transparency and thickness
                pygame.draw.circle(circle_surface, rgba, center, radius + thickness // 2, width=thickness)

                # Blit the circle surface to the main UI surface
                blit_pos = beacon.pos - pygame.Vector2(diameter // 2, diameter // 2)
                self.map.blit(circle_surface, blit_pos)

robot = Node(pygame.Vector2(map.get_width() / 2, map.get_height() / 2), "red", 20, map)

beacon1 = Beacon(pygame.Vector2(100, 300), "blue", 10, map, robot, add_noise=True, tx_power=0, path_loss_exponent=2.0)
beacon2 = Beacon(pygame.Vector2(1000, 300), "blue", 10, map, robot, add_noise=True, tx_power=0, path_loss_exponent=2.0)
beacon3 = Beacon(pygame.Vector2(1000, 700), "blue", 10, map, robot, add_noise=True, tx_power=0, path_loss_exponent=2.0)
beacon4 = Beacon(pygame.Vector2(100, 700), "blue", 10, map, robot, add_noise=True, tx_power=0, path_loss_exponent=2.0)
beacon5 = Beacon(pygame.Vector2(500, 500), "blue", 10, map, robot, add_noise=True, tx_power=0, path_loss_exponent=2.0)

UI = UI(base, map)

draw_circles = True

must_update = True

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                draw_circles = not draw_circles
                must_update = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                robot.check_press()
                beacon1.check_press()
                beacon2.check_press()
                beacon3.check_press()
                beacon4.check_press()
                beacon5.check_press()
                must_update = True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                robot.drop()
                beacon1.drop()
                beacon2.drop()
                beacon3.drop()
                beacon4.drop()
                beacon5.drop()
                must_update = True

        if event.type == pygame.QUIT:
            running = False

    if must_update:
        robot.drag()
        beacon1.drag()
        beacon2.drag()
        beacon3.drag()
        beacon4.drag()
        beacon5.drag()

        screen.fill("purple")

        base.fill("brown")
        map.fill("white")

        robot.draw()
        beacon1.draw()
        beacon2.draw()
        beacon3.draw()
        beacon4.draw()
        beacon5.draw()

        if draw_circles: UI.draw_circles_op(robot, [beacon1, beacon2, beacon3, beacon4, beacon5], "black")

        UI.draw_text("Press C to show/hide RSSI circles", pygame.Vector2(10, 10))

        screen.blit(base, (0, 0))

        pygame.display.flip()

    

    if not pygame.mouse.get_pressed()[0]:
        must_update = False

    dt = clock.tick(60) / 1000

pygame.quit()