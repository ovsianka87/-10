import numpy as np

# Функция Экле
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(sum([xi**2 for xi in x]) / d))
    sum2 = -np.exp(sum([np.cos(c * xi) for xi in x]) / d)
    return sum1 + sum2 + a + np.exp(1)

# Алгоритм роя частиц
class Particle:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.current_value = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        inertia = w * self.velocity
        cognitive = c1 * np.random.random() * (self.best_position - self.position)
        social = c2 * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

class ParticleSwarmOptimizer:
    def __init__(self, function, dimensions, bounds, num_particles, iterations):
        self.function = function
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_value = float('inf')

    def optimize(self):
        for iteration in range(self.iterations):
            for particle in self.particles:
                particle.current_value = self.function(particle.position)

                if particle.current_value < particle.best_value:
                    particle.best_value = particle.current_value
                    particle.best_position = particle.position.copy()

                if particle.current_value < self.global_best_value:
                    self.global_best_value = particle.current_value
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)

            print(f"Iteration {iteration + 1}/{self.iterations}, Global Best Value: {self.global_best_value}")

        return self.global_best_position, self.global_best_value

# Параметры
dimensions = 2
bounds = (-5, 5)
num_particles = 30
iterations = 100

# Оптимизация
pso = ParticleSwarmOptimizer(ackley_function, dimensions, bounds, num_particles, iterations)
best_position, best_value = pso.optimize()

print(f"Best Position: {best_position}, Best Value: {best_value}")
