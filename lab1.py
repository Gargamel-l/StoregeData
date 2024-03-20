# Ген алгоритм коммивояжера 
import numpy as np
import random
import matplotlib.pyplot as plt

# Функция для создания матрицы расстояний между городами
def create_distance_matrix(n):
    return np.random.randint(1, 100, size=(n, n))

# Функция для расчета длины маршрута
def calculate_route_length(route, distance_matrix):
    return sum([distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)]) + distance_matrix[route[-1], route[0]]

# Создание начальной популяции
def create_initial_population(size, n_cities):
    return [random.sample(range(n_cities), n_cities) for _ in range(size)]

# Отбор родителей для скрещивания
def select_parents(population, distance_matrix, k=5):
    selection = random.sample(population, k)
    selection.sort(key=lambda x: calculate_route_length(x, distance_matrix))
    return selection[0], selection[1]

# Функция скрещивания (кроссовера)
def crossover(parent1, parent2):
    size = len(parent1)
    child = [-1]*size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1
    return child

# Функция мутации
def mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]
    return route

# Функция для генерации координат городов
def generate_city_coordinates(n_cities):
    return np.random.rand(n_cities, 2) * 100

# Функция для визуализации лучшего маршрута
def plot_best_route(best_route, city_coordinates):
    plt.figure(figsize=(10, 5))
    for i in range(len(best_route)):
        plt.plot([city_coordinates[best_route[i-1], 0], city_coordinates[best_route[i], 0]],
                 [city_coordinates[best_route[i-1], 1], city_coordinates[best_route[i], 1]], 'ro-')
    plt.title("Best Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# Основная функция генетического алгоритма
def genetic_algorithm(n_cities, population_size, n_generations):
    distance_matrix = create_distance_matrix(n_cities)
    population = create_initial_population(population_size, n_cities)
    city_coordinates = generate_city_coordinates(n_cities)  # Генерация координат городов

    for _ in range(n_generations):
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = select_parents(population, distance_matrix)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population

    best_route = min(population, key=lambda x: calculate_route_length(x, distance_matrix))
    best_length = calculate_route_length(best_route, distance_matrix)
    plot_best_route(best_route, city_coordinates)  # Визуализация лучшего маршрута
    return best_route, best_length, city_coordinates

n_cities = 10 # кол-во городов  
population_size = 100 # размер популяции
n_generations = 1000 # кол-во поколений
# Запуск ген алгоритма
best_route, best_length, city_coordinates = genetic_algorithm(n_cities, population_size, n_generations)
# Вывод лучшего маршрута 
print(f"Best route: {best_route}\nLength: {best_length}")
