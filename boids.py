import matplotlib.pyplot as plt
import numpy as np
import pygame
import multiprocessing as mp


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

        # initialize swarm
        self.swarm = np.array([Particle(self.w, self.h) for _ in range(self.N)])

        # initialize gui related info
        self.gui = gui
        if gui:
            pygame.init()

            self.screen = pygame.display.set_mode([w, h])
            self.draw_size = draw_size

            self.fps = fps
            self.fps_limiter = pygame.time.Clock()

    def neighbours_of(self, p1):
        neighbours = []
        distances = []

        for p2 in self.swarm:
            distance = (p1.pos.x - p2.pos.x) ** 2 + (p1.pos.y - p2.pos.y) ** 2
            if distance <= self.interaction_radius_squared:
                neighbours.append(p2)
                distances.append(distance)

        return neighbours, distances

    def step(self):
        neighbour_distances = []

        for particle in self.swarm:
            neighbours, distances = self.neighbours_of(particle)
            particle.step(neighbours, self.cohesion, self.seperation, self.alignment, self.speed)
            neighbour_distances.append(distances)

        return sum(neighbour_distances, [])

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

    def run(self, num_steps=300, process_id=None, results=None):
        interrupt = False
        step = 0

        neighbours = []
        orders = []

        while step < num_steps and not interrupt:
            # the user interrupted the run
            if self.gui and self.check_interrupt(): interrupt = True

            # update the scene
            neighbours.append(self.step())
            orders.append(self.order())

            # draw the scene
            if self.gui: self.draw()

            step += 1

        if self.gui: pygame.quit()

        if process_id is not None:
            results['orders'][process_id] = orders
            results['neighbours'][process_id] = neighbours
            return

        return orders, neighbours


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
                try: away /= away.magnitude_squared()  # normalize
                except ZeroDivisionError: away = pygame.Vector2(0, 0)
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
    mean = np.mean(orders, axis=0)
    std = np.std(orders, axis=0)

    # plot the individual runs
    for order in orders:
        plt.plot(order, color='grey', alpha=0.30)

    # plot the mean and std
    plt.plot(mean, label='mean', color='tab:red')
    plt.fill_between(
        np.arange(0, len(mean)), mean - std, mean + std,
        alpha=0.3, label='std', color='tab:red'
    )

    plt.xlabel('step')
    plt.ylabel('average normalized velocity')

    plt.legend()

    plt.savefig('results/test_orders.png')
    plt.show()


def plot_neighbours(neighbours):
    # note we do a sqrt because the distances are squared
    mean = np.array([np.mean(np.sqrt(sum(neighbours[:, t], []))) for t in range(neighbours.shape[1])])
    std = np.array([np.std(np.sqrt(sum(neighbours[:, t], []))) for t in range(neighbours.shape[1])])

    plt.plot(mean, label='mean', color='tab:red')
    plt.fill_between(
        np.arange(0, len(mean)), mean - std, mean + std,
        alpha=0.3, label='std', color='tab:red'
    )

    plt.xlabel('step')
    plt.ylabel('nearest neighbour distance')

    plt.legend()

    plt.savefig('results/test_neighbours_gaussian.png')
    plt.show()

    step = 30
    plt.boxplot([np.sqrt(sum(neighbours[:, t], [])) for t in range(0, neighbours.shape[1], step)])
    # plt.xticks(range(0, neighbours.shape[1], step)) # FIXME xlabels

    plt.xlabel(f'step x {step}')
    plt.ylabel('nearest neighbour distance')

    plt.legend()

    plt.savefig('results/test_neighbours_boxplot.png')
    plt.show()


def run_repeated(
    w, h, N, cohesion, seperation, alignment,speed, interaction_radius,
    num_steps=300, repetitions=mp.cpu_count(),
):
    manager = mp.Manager()
    results = manager.dict()

    results['neighbours'] = manager.dict()
    results['orders'] = manager.dict()

    jobs = []

    scenes = [
        Scene(
            w=w, h=h, N=N, cohesion=cohesion, seperation=seperation, alignment=alignment,
            speed=speed, interaction_radius=interaction_radius, gui=False
        ) for _ in range(repetitions)
    ]

    for process_id in range(repetitions):
        jobs.append(mp.Process(
            target=scenes[process_id].run,
            args=(num_steps, process_id, results)
        ))
        jobs[-1].start()

    for process in jobs:
        process.join()

    return (
        np.array(results['orders'].values()),
        np.array(results['neighbours'].values(), dtype=object),  # this array is inhomogeneous
    )


def mutate(variables, samples):
    mutated_variables = {}
    for variable_name, variable_props in variables.items():
        mutated = np.random.normal(samples[variable_name], variable_props['sigma'])
        while not variable_props['valid'](mutated):
            mutated = np.random.normal(samples[variable_name], variable_props['sigma'])
        mutated_variables[variable_name] = mutated
    return mutated_variables


def sample_priors(variables):
    return {
        variable_name: variable_function['prior'](1)[0]
        for variable_name, variable_function in variables.items()
    }


def accept(orders, epsilon, n=50):
    # check the mean of the last n values over all runs > epsilon
    return np.mean(orders[:, -n:]) > epsilon


def create_population(
    epsilon, w, h, N, abc_N, variables, speed, interaction_radius, num_steps, repetitions
):
    accepted = []
    while len(accepted) < abc_N:
        variable_samples = sample_priors(variables)
        orders, _ = run_repeated(
            w, h, N, speed=speed, interaction_radius=interaction_radius, **variable_samples
        )

        if accept(orders, epsilon):
            accepted.append(variable_samples)

    return accepted


def mutate_population(
    population, epsilon, w, h, N, abc_N, variables, speed, interaction_radius, num_steps, repetitions
):
    accepted = []
    while len(accepted) < abc_N:
        variable_samples = mutate(variables, np.random.choice(population))
        orders, _ = run_repeated(
            w, h, N, speed=speed, interaction_radius=interaction_radius, **variable_samples
        )

        if accept(orders, epsilon):
            accepted.append(variable_samples)

    return accepted


def abc(
    w, h, N, abc_N, epsilons, variables, speed, interaction_radius,
    num_steps=300, repetitions=mp.cpu_count(),
):
    print(f'abc: creating the initial population with e={epsilons[0]}')
    populations = [create_population(
        epsilons[0], w, h, N, abc_N, variables,speed,
        interaction_radius, num_steps, repetitions
    )]

    for index, epsilon in enumerate(epsilons[1:]):
        print(f'abc: mutating population with e={epsilon:.3f} ({index + 1}/{len(epsilons) - 1})')
        populations.append(mutate_population(
            populations[-1], epsilon, w, h, N, abc_N, variables, speed,
            interaction_radius, num_steps, repetitions
        ))

    return populations


def main():
    np.random.seed(0)  # for reproducability

    # run multiple experiments
    # orders, neighbours = run_repeated(
    #     w=600, h=600, N=200, cohesion=100, seperation=30, alignment=1,
    #     speed=5, interaction_radius=100,
    # )

    # plot_order(orders)
    # plot_neighbours(neighbours)

    # run a single trial with gui
    scene = Scene(
        w=600, h=600, N=200, cohesion=100, seperation=30, alignment=1,
        speed=5, interaction_radius=100, gui=True, fps=30
    )
    scene.run(num_steps=np.inf)

    # run abc
    # populations = abc(
    #     w=600, h=600, abc_N=20, N=15, speed=5, interaction_radius=100,
    #     epsilons=np.linspace(0.5, 0.95, 10),
    #     variables={
    #         'cohesion': {
    #             'prior': lambda n: np.random.uniform(1, 100, n),
    #             'valid': lambda x: 1 < x < 100,
    #             'sigma': 10,
    #         },
    #         'seperation': {
    #             'prior': lambda n: np.random.uniform(10, 50, n),
    #             'valid': lambda x: 10 < x < 50,
    #             'sigma': 10,
    #         },
    #         'alignment': {
    #             'prior': lambda n: np.random.uniform(0.1, 1.5, n),
    #             'valid': lambda x: 0.1 < x < 1.5,
    #             'sigma': 0.4,
    #         },
    #     }
    # )
    # for population in populations[-1:]:
    #     print(*population, sep='\n')
    #     print()

    #     for accepted in population[:3]:
    #         # run a single trial with gui
    #         scene = Scene(
    #             w=600, h=600, N=15, **accepted, speed=5, interaction_radius=100, gui=True, fps=30
    #         )
    #         orders, _ = scene.run(num_steps=300)
    #         plot_order(np.array([orders]))


if __name__ == "__main__":
    main()