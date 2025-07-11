import pygame
import scipy
import random
import math
import numpy as np

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
            return round(self._calculate_rssi() + np.random.normal(0, self.noise))
        else:
            return self._calculate_rssi()
    
    def get_true_distance(self) -> float:
        return self.pos.distance_to(self.target.pos)

    def get_distance(self) -> float:
        if not self.add_noise:
            return self.get_true_distance()
        else:
            # Calculate the distance with noise
            rssi = self._calculate_rssi()
            distance = 10 ** ((self.tx_power - rssi) / (10 * self.path_loss_exponent)) + random.lognormvariate(0, self.noise)
            return distance

class ParticleFilter:
    def __init__(self, num_particles, map_width, map_height, beacons):
        self.num_particles = num_particles
        self.map_width = map_width
        self.map_height = map_height
        self.beacons = beacons
        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles
        self.estimated_pos = pygame.Vector2(map_width/2, map_height/2)
        
    def initialize_particles(self):
        # Initialize particles randomly across the map
        particles = []
        for _ in range(self.num_particles):
            x = random.uniform(0, self.map_width)
            y = random.uniform(0, self.map_height)
            particles.append(pygame.Vector2(x, y))
        return particles
    
    def predict(self, motion_std=1.0):
        # Add small random motion to particles (simulates robot movement)
        for i in range(self.num_particles):
            self.particles[i].x += random.gauss(0, motion_std)
            self.particles[i].y += random.gauss(0, motion_std)
            
            # Keep particles within map bounds
            self.particles[i].x = max(0, min(self.map_width, self.particles[i].x))
            self.particles[i].y = max(0, min(self.map_height, self.particles[i].y))
    
    def update_weights(self):
        # Update weights based on RSSI measurements
        for i, particle in enumerate(self.particles):
            total_log_likelihood = 0.0
            
            for beacon in self.beacons:
                # Calculate expected RSSI at particle position
                dist = particle.distance_to(beacon.pos)
                expected_rssi = beacon.tx_power - 10 * beacon.path_loss_exponent * math.log10(dist) if dist > 0 else -100
                
                # Get actual RSSI measurement
                measured_rssi = beacon.get_rssi()
                
                # Calculate likelihood (probability of getting this measurement given particle position)
                # Using Gaussian noise model
                likelihood = scipy.stats.norm(expected_rssi, beacon.noise).pdf(measured_rssi)
                
                # Avoid zero likelihood
                likelihood = max(likelihood, 1e-10)
                
                total_log_likelihood += math.log(likelihood)
            
            # Update weight (using log to avoid numerical underflow)
            self.weights[i] = math.exp(total_log_likelihood)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
    
    def resample(self):
        # Resample particles based on weights
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        new_particles = [pygame.Vector2(self.particles[i].x, self.particles[i].y) for i in indices]
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights
    
    def estimate_position(self):
        # Estimate position as weighted average of particles
        sum_x = 0.0
        sum_y = 0.0
        for i in range(self.num_particles):
            sum_x += self.particles[i].x * self.weights[i]
            sum_y += self.particles[i].y * self.weights[i]
        
        self.estimated_pos = pygame.Vector2(sum_x, sum_y)
        return self.estimated_pos
    
    def update(self):
        self.predict()
        self.update_weights()
        self.resample()
        return self.estimate_position()
    
    def draw(self, surface):
        # Draw particles
        for particle in self.particles:
            pygame.draw.circle(surface, (100, 100, 255, 50), (int(particle.x), int(particle.y)), 2)
        
        # Draw estimated position
        pygame.draw.circle(surface, (255, 0, 0), (int(self.estimated_pos.x), int(self.estimated_pos.y)), 10, 2)

class UI:
    def __init__(self, base: pygame.Surface, map: pygame.Surface):
        self.base = base
        self.map = map
        self.font = pygame.font.SysFont("Arial", 36)

    def draw_text(self, text: str, pos: pygame.Vector2, color: str = "black"):
        text_surface = self.font.render(text, True, color)
        self.base.blit(text_surface, pos)
    
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

beacon1 = Beacon(pygame.Vector2( 100, 300), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon2 = Beacon(pygame.Vector2(1000, 300), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon3 = Beacon(pygame.Vector2(1000, 700), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon4 = Beacon(pygame.Vector2( 100, 700), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon5 = Beacon(pygame.Vector2( 500, 500), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)

beacons = [beacon1, beacon2, beacon3, beacon4, beacon5]

# Initialize particle filter
particle_filter = ParticleFilter(num_particles=500, 
                               map_width=map.get_width(), 
                               map_height=map.get_height(),
                               beacons=beacons)

UI = UI(base, map)

draw_circles = True
use_particle_filter = True
show_particles = True

must_update = True

while running:
    dt = clock.tick(60) / 1000

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                draw_circles = not draw_circles
                must_update = True
            if event.key == pygame.K_p:
                use_particle_filter = not use_particle_filter
                must_update = True
            if event.key == pygame.K_s:
                show_particles = not show_particles
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

        if draw_circles: 
            UI.draw_circles_op(robot, beacons, "black")
        
        if use_particle_filter:
            estimated_pos = particle_filter.update()
            error = estimated_pos.distance_to(robot.pos)

            if show_particles:
                particle_filter.draw(map)
            
            # Draw estimated position
            pygame.draw.circle(map, (255, 0, 255), (int(estimated_pos.x), int(estimated_pos.y)), 15, 2)
            # Draw line from estimated to true position
            pygame.draw.line(map, (255, 0, 0), robot.pos, estimated_pos, 2)

        UI.draw_text("Press C to show/hide RSSI circles", pygame.Vector2(10, 10))
        UI.draw_text("Press P to toggle particle filter", pygame.Vector2(610, 10))
        UI.draw_text("Press S to show/hide particles", pygame.Vector2(1210, 10))
        if use_particle_filter:
            error = estimated_pos.distance_to(robot.pos)
            UI.draw_text(f"Estimation error: {error:.1f} pixels", pygame.Vector2(10, 50))

        screen.blit(base, (0, 0))

    pygame.display.flip()

    if not pygame.mouse.get_pressed()[0]:
        must_update = False
pygame.quit()