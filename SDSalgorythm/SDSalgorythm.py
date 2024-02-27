import numpy as np

# Целевая функция
def objective_function(x):
    return -x**2 + 4*x

# Функция тестирования агента
def test_agent(agent, objective_function, bounds):
    position = np.random.uniform(bounds[0], bounds[1])
    agent['position'] = position
    agent['fitness'] = objective_function(position)
    agent['active'] = True if np.random.rand() < agent['fitness'] / max_fitness else False

# Фаза диффузии
def diffuse_agents(agents):
    for i, agent in enumerate(agents):
        if not agent['active']:
            selected_agent = np.random.choice(agents)
            agent['position'] = selected_agent['position']
            agent['fitness'] = selected_agent['fitness']
            agent['active'] = selected_agent['active']

# Определение максимального значения приспособленности
bounds = [-2, 5]
max_fitness = objective_function(2)  # Предполагаем, что максимальное значение известно

# Инициализация агентов
num_agents = 50
agents = [{'position': None, 'fitness': None, 'active': False} for _ in range(num_agents)]

# Алгоритм
iterations = 100
for iteration in range(iterations):
    # Фаза тестирования
    for agent in agents:
        test_agent(agent, objective_function, bounds)
    
    # Фаза диффузии
    diffuse_agents(agents)

    # Оценка результата
    best_agent = max(agents, key=lambda x: x['fitness'])
    print(f"Iteration {iteration}: Best Position = {best_agent['position']}, Fitness = {best_agent['fitness']}")

# Вывод лучшего решения
print(f"Best solution: Position = {best_agent['position']}, Fitness = {best_agent['fitness']}")

