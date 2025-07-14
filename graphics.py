import pygame
import scipy
import random
import math
import numpy as np
import numpy.typing as npt
import itertools

# pygame setup
pygame.init()
pygame.display.set_caption("Multilateration simulation")
screen = pygame.display.set_mode((1880, 900))

base = pygame.Surface((1880, 900))
map = base.subsurface((0, 100, 1880, 800))

clock = pygame.time.Clock()
running = True
dt = 0

# Scale factor (1 pixel = scale meters)
SCALE = 0.1  # 1px = 0.1 meters (10 cm per pixel)

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
        distance = self.get_true_distance() * SCALE  # Convert to meters
        if distance <= 0:
            return -100  # Avoid log(0)
        return self.tx_power - 10 * self.path_loss_exponent * math.log10(distance)

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
            return distance / SCALE  # Convert back to pixels

class UI:
    def __init__(self, base: pygame.Surface, map: pygame.Surface, robot: Node):
        self.base = base
        self.map = map
        self.robot = robot
        self.font = pygame.font.SysFont("Arial", 24)
        self.big_font = pygame.font.SysFont("Arial", 36)

    def draw_text(self, text: str, pos: pygame.Vector2, color: str = "black", big=False):
        font = self.big_font if big else self.font
        text_surface = font.render(text, True, color)
        self.base.blit(text_surface, pos)
    
    def draw_distance_lines(self, beacons: list[Beacon], color: str = "black"):
        for beacon in beacons:
            # Draw line between robot and beacon
            pygame.draw.line(self.map, color, self.robot.pos, beacon.pos, 1)
            
            # Calculate distance in meters
            distance_px = self.robot.pos.distance_to(beacon.pos)
            distance_m = distance_px * SCALE
            
            # Position the text in the middle of the line
            mid_point = (self.robot.pos + beacon.pos) / 2
            text = f"{distance_m:.1f}m"
            text_surface = self.font.render(text, True, color)
            
            # Draw a background for better readability
            text_rect = text_surface.get_rect(center=(mid_point.x, mid_point.y))
            pygame.draw.rect(self.map, "black", text_rect.inflate(4, 4))
            self.map.blit(text_surface, text_rect)
    
    def draw_circles_op(self, beacons: list[Beacon], color: str = "black"):
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

    def draw_estimation(self, estimated_pos: pygame.Vector2, covariance: np.array, color: str):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        
        # Confidence scaling
        k = np.sqrt(-2 * np.log(1 - 0.95))  # For 2D (chi-squared)
        a = k * np.sqrt(eigenvalues[0])
        b = k * np.sqrt(eigenvalues[1])
        angle = np.degrees(np.arctan2(eigenvectors[1,0], eigenvectors[0,0]))

        try:
            ellipse_surf = pygame.Surface((a, b), pygame.SRCALPHA)
            pygame.draw.ellipse(ellipse_surf, color, ellipse_surf.get_rect(), 2)
            rotated_surf = pygame.transform.rotate(ellipse_surf, -angle)
            rotated_rect = rotated_surf.get_rect(center=estimated_pos)
            self.map.blit(rotated_surf, rotated_rect.topleft)

        except pygame.error:
            print("Error: confidence ellipse couldn't draw")
        
        pygame.draw.circle(self.map, color, estimated_pos, 4)
        pygame.draw.line(self.map, color, self.robot.pos, estimated_pos, 2)

class ParticleFilter:
    def __init__(self, num_particles: int, map_width: int, map_height: int, beacons: list[Beacon]):
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
    
    def predict(self, motion_std: float = 5.0):
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
                dist = particle.distance_to(beacon.pos) * SCALE  # Convert to meters
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
        mean = np.average(self.particles, axis=0, weights=self.weights)
        
        self.estimated_pos = pygame.Vector2(mean[0], mean[1])
        return self.estimated_pos
    
    def update(self):
        self.predict()
        self.update_weights()
        self.resample()
        return self.estimate_position()
    
    def draw_particles(self, surface: pygame.Surface):
        # Draw particles
        for particle in self.particles:
            pygame.draw.circle(surface, (100, 100, 255, 50), (int(particle.x), int(particle.y)), 2)

    def draw_estimate(self, surface: pygame.Surface):
        particles = np.array([list(i) for i in self.particles]) # Convert particles from List[pygame.Vector2] to np.array[[x, y]]

        cov = np.cov(particles.T, aweights=self.weights)

        UI.draw_estimation(self.estimated_pos, cov, "red")

def trilaterate(beacons: tuple[Beacon, Beacon, Beacon]) -> tuple[float, float]:
    """
    Find power center, used to get the initial guess for the optimizer
    """
    r1, r2, r3 = [i.get_distance() for i in beacons]
    ((x1, y1), (x2, y2), (x3, y3)) = [[i.pos[0], i.pos[1]] for i in beacons]

    a = (x2 - x1)
    b = (y2 - y1)

    r = (x3 - x1)
    s = (y3 - y1)

    # Check that the three beacons aren't colinear
    if ( b*r - a*s ) == 0.0:
        return (None, None)

    c = (r1**2 - r2**2 + y2**2 + x2**2 - y1**2 - x1**2 ) / 2
    t = (r1**2 - r3**2 + y3**2 + x3**2 - y1**2 - x1**2 ) / 2

    # Use Cramer's rule
    y = ( r*c - a*t ) / ( b*r - a*s )

    if a == 0.0:
        x = ( t - s*y ) / r
    else:
        x = ( c - b*y ) / a

    return (x, y)

def multilaterate(robot: Node, beacons: list[Beacon], discard_threeshold: float) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Trilaterates with all combinations and then returns average and covariance
    """
    points = np.empty((0, 2))

    for i in itertools.combinations(beacons, 3):
        new_point = trilaterate(i)

        if new_point[0] == None:
            print(f"The beacons {i} are colinear")

        else:
            points = np.append(points, np.array([[new_point[0], new_point[1]]]), axis=0)

    estimated_pos = np.mean(points, axis=0)

    cov = np.cov(points.T)

    return (estimated_pos, cov)

def multilateration(map: pygame.Surface, robot: Node, beacons: list[Beacon], iterations: int, discard_threeshold: float) -> float: # Return real estimation error
    """
    Iterates a defined amount of times over the combinations of beacons for trilateration, then calculates the average position and draws it in the map
    """
    
    points = np.empty((0, 2))

    if iterations > 1:
        for i in range(iterations):
            estimate = multilaterate(robot, beacons, discard_threeshold)[0]
            points = np.append(points, np.array([[estimate[0], estimate[1]]]), axis=0)
    elif iterations == 1:
        points = multilaterate(robot, beacons, discard_threeshold)
    
    else:
        print("Invalid number of multilateration iterations")

    if len(points) == 0:
        print("All on this lateration batch was discarted, measurements had too much noise")
    elif iterations == 1:
        estimated_pos = pygame.Vector2(points[0][0], points[0][1])
        UI.draw_estimation(estimated_pos, points[1], "green")
        return estimated_pos.distance_to(robot.pos)
    else:
        estimated_pos = np.mean(points, axis=0)
        estimated_pos = pygame.Vector2(estimated_pos[0], estimated_pos[1])
        cov = np.cov(points.T)

        UI.draw_estimation(estimated_pos, cov, "green")

        # Draw point indicating estimate and a line from estimated to true position
        pygame.draw.circle(map, "green", estimated_pos, 4)
        pygame.draw.line(map, "green", robot.pos, estimated_pos, 2)

        return estimated_pos.distance_to(robot.pos)

robot = Node(pygame.Vector2(map.get_width() / 2, map.get_height() / 2), "red", 20, map)

beacon1 = Beacon(pygame.Vector2(750, 60), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon2 = Beacon(pygame.Vector2(120, 420), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon3 = Beacon(pygame.Vector2(540, 735), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon4 = Beacon(pygame.Vector2(1500, 720), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)
beacon5 = Beacon(pygame.Vector2(1600, 161), "blue", 10, map, robot, add_noise=True, tx_power=-50, path_loss_exponent=2.0)

beacons = [beacon1, beacon2, beacon3, beacon4, beacon5]

UI = UI(base, map, robot)

draw_circles = True
use_particle_filter = True
show_particles = True
show_distance_lines = True

must_redraw = True

nodes = [
    robot,
    beacon1,
    beacon2,
    beacon3,
    beacon4,
    beacon5
]

recalculate_particles = True

use_multilateration = False
use_single_sample_multilateration = True

multilateration_error = None

while running:
    dt = clock.tick(60) / 1000

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                draw_circles = not draw_circles
                must_redraw = True
            if event.key == pygame.K_p:
                use_particle_filter = not use_particle_filter
                must_redraw = True
                if use_particle_filter:
                    recalculate_particles = True
                else:
                    recalculate_particles = True
            if event.key == pygame.K_s:
                show_particles = not show_particles
                must_redraw = True
            if event.key == pygame.K_l:
                show_distance_lines = not show_distance_lines
                must_redraw = True
            if event.key == pygame.K_t:
                use_multilateration = True
                must_redraw = True
            if event.key == pygame.K_y:
                use_single_sample_multilateration = not use_single_sample_multilateration
                must_redraw = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for i in nodes:
                    i.check_press()
                    if i.pressed:
                        must_redraw = True
                        recalculate_particles = False
                        break

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                for i in nodes:
                    if i.pressed:
                        i.drop()
                        must_redraw = True
                        recalculate_particles = True
                        break

        if event.type == pygame.QUIT:
            running = False

    if must_redraw:
        for i in nodes:
            i.drag()

        screen.fill("purple")

        base.fill("brown")
        map.fill("white")

        for i in nodes:
            i.draw()

        if draw_circles:
            UI.draw_circles_op(beacons, "black")
        
        if show_distance_lines:
            UI.draw_distance_lines(beacons, "green")
    
            
        if use_particle_filter:
            if recalculate_particles:
                # Initialize particle filter
                particle_filter = ParticleFilter(num_particles=1500,
                                            map_width=map.get_width(),
                                            map_height=map.get_height(),
                                            beacons=beacons)
                estimated_pos = particle_filter.update()
                error = estimated_pos.distance_to(robot.pos)
            
            if show_particles:
                particle_filter.draw_particles(map)
            
            particle_filter.draw_estimate(map)

            UI.draw_text(f"Estimation error: {error*SCALE:.1f} meters", pygame.Vector2(800, 60))

        if use_multilateration:
            if use_single_sample_multilateration:
                multilateration_error = multilateration(map, robot, beacons, 1, 100)
            else:
                multilateration_error = multilateration(map, robot, beacons, 5, 100)

        if use_multilateration:
            if multilateration_error == None:
                UI.draw_text(f"Estimation error: Error", pygame.Vector2(1550, 60))
            else:
                UI.draw_text(f"Estimation error: {multilateration_error*SCALE:.1f}", pygame.Vector2(1550, 60))

        UI.draw_text("Press C to show/hide RSSI confidence circles", pygame.Vector2(10, 15))
        UI.draw_text("Press L to show/hide distance lines", pygame.Vector2(10, 45))
        UI.draw_text("Press P to toggle particle filter", pygame.Vector2(450, 15))
        UI.draw_text("Press S to show/hide particles", pygame.Vector2(450, 45))
        UI.draw_text("Press T to toggle multilateration", pygame.Vector2(1250, 60))
        UI.draw_text(f"Multiple sampling multilateration {"OFF" if use_single_sample_multilateration else "ON"} press Y to toggle", pygame.Vector2(1250, 30))


    mouse_pos = pygame.mouse.get_pos()

    text = f"[{mouse_pos[0]:0>4}, {mouse_pos[1]:0>4}]"
    text_surface = UI.font.render(text, True, "black")
            
    # Draw a background for better readability
    text_rect = text_surface.get_rect(center=(1820, 15))
    pygame.draw.rect(base, "brown", text_rect.inflate(4, 4))
    base.blit(text_surface, text_rect)

    screen.blit(base, (0, 0))

    pygame.display.flip()

    if not pygame.mouse.get_pressed()[0]:
        must_redraw = False
        recalculate_particles = False

pygame.quit()