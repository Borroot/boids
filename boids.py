import matplotlib.pyplot as plt
import numpy as np
import pygame



class Scene:

    def __init__(
        self, w, h, N, cohesion, seperation, alignment, speed, interaction_radius,
        gui=True, fps=30, draw_size=5,
    ):
        self.w = w  # width  of the window
        self.h = h  # height of the window
        self.N = N  # number of particles

        # set the simulation parameters
        self.cohesion = cohesion
        self.seperation = seperation
        self.alignment = alignment
        self.speed = speed
        self.interaction_radius_squared = interaction_radius**2

        # initialize gui related info
        self.gui = gui
        if gui:
            pygame.init()

            self.screen = pygame.display.set_mode([w, h])
            self.draw_size = draw_size

            self.fps = fps
            self.fps_limiter = pygame.time.Clock()

    def neighbours(self, p1):
        return [
            p2 for p2 in self.swarm
            if (p1.pos.x - p2.pos.x) ** 2 + (p1.pos.y - p2.pos.y) ** 2 <= self.interaction_radius_squared
        ]

    def step(self):
        for particle in self.swarm:
            particle.step(self.neighbours(particle), self.cohesion, self.seperation, self.alignment, self.speed)

    def draw(self):
        self.screen.fill((255, 255, 255))

        for particle in self.swarm:
            pygame.draw.circle(self.screen, (0, 0, 0), particle.pos, self.draw_size)

        pygame.display.flip()
        self.fps_limiter.tick(self.fps)

    def check_interrupt(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return True
        return False

    def order(self):
        return sum(
            [particle.dir.normalize() for particle in self.swarm], pygame.Vector2(0, 0)
        ).magnitude() / self.N

    def run(self, num_steps=300):
        # initialize swarm
        self.swarm = np.array([Particle(self.w, self.h) for _ in range(self.N)])

        interrupt = False
        step = 0

        orders = []

        while step < num_steps and not interrupt:
            # the user interrupted the run
            if self.gui and self.check_interrupt(): interrupt = True

            # update the scene
            self.step()
            orders.append(self.order())

            # draw the scene
            if self.gui: self.draw()

            step += 1

        if self.gui: pygame.quit()

        return orders


class Particle:

    def __init__(self, w, h):
        self.pos = pygame.Vector2(np.random.randint(w), np.random.randint(h))
        self.dir = pygame.Vector2(*(np.random.rand(2) * 2 - 1)).normalize()

        self.w = w
        self.h = h

    def wrap(self):
        # wrap the particle around to the other side of the scene
        if self.pos.x < 0 or self.pos.x > self.w: self.pos.x %= self.w
        if self.pos.y < 0 or self.pos.y > self.h: self.pos.y %= self.h

    def step(self, neighbours, cohesion, seperation, alignment, speed):
        avg_pos  = pygame.Vector2(0, 0)  # average position within interaction radius
        avg_away = pygame.Vector2(0, 0)  # average direction to move away from others

        # the average direction within the interaction radius
        avg_sin = 0
        avg_cos = 0

        for neighbour in neighbours:
            avg_sin += np.sin(np.deg2rad(neighbour.dir.as_polar()[1]))
            avg_cos += np.cos(np.deg2rad(neighbour.dir.as_polar()[1]))

            avg_pos += neighbour.pos

            if neighbour != self:
                away = self.pos - neighbour.pos
                away /= away.magnitude_squared()  # normalize
                avg_away += away

        # take the mean
        avg_pos  /= len(neighbours)
        avg_away /= len(neighbours)

        # alignment: move towards the average heading of the neighbours
        avg_angle = np.arctan2(avg_sin, avg_cos)  # note first y then x
        avg_dir = pygame.Vector2.from_polar((1, np.rad2deg(avg_angle)))
        self.dir = avg_dir * alignment

        # cohesion: move towards the average position of the neighbours
        self.dir += (avg_pos - self.pos) / cohesion

        # seperation: move away from the neighbours to avoid crowding
        self.dir += avg_away * seperation

        # update the position with a constant speed
        self.pos += self.dir.normalize() * speed
        self.wrap()


def plot_order(orders):
    for order in orders:
        plt.plot(list(range(len(order))), order)

    plt.xlabel('step')
    plt.ylabel('average normalized velocity')

    plt.savefig('results/test.png')
    plt.show()


def main():
    np.random.seed(0)  # for reproducability

    scene = Scene(
        w=600, h=600, N=200, cohesion=100, seperation=30, alignment=1, speed=5, interaction_radius=100,
        gui=False, fps=30
    )

    repetitions = 10
    orders = [scene.run() for _ in range(10)]
    plot_order(orders)


if __name__ == "__main__":
    main()
