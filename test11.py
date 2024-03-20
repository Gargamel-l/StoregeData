# использовали алгоритм дифференциальной эволюции для решения задачи оптимизации.
# Этот алгоритм является популярным методом глобальной оптимизации, основанным на популяционном подходе. 
# Он эффективно исследует пространство поиска за счет механизмов мутации, кроссовера и отбора, 
# что позволяет находить оптимальные или близкие к оптимальным решения сложных оптимизационных задач.
import numpy as np

def fitness(x):
    """Вычисляет приспособленность особи.
    Функция приспособленности вычисляет значение функции, которую мы хотим оптимизировать.
    Здесь используется простая функция: f(x) = x * sin(10*pi*x) + 1. Эта функция имеет несколько локальных максимумов и минимумов."""
    return x * np.sin(10 * np.pi * x) + 1

def initialize_population(size, dim):
    """Инициализирует популяцию случайными значениями.
    Эта функция создает популяцию из `size` особей, каждая из которых является вектором размерности `dim`.
    В данном случае dim установлено в 1, что означает, что мы имеем дело с одномерной задачей."""
    return np.random.uniform(-1, 2, (size, dim))

def mutate(population, F):
    """Применяет мутацию к популяции.
    Для каждой особи в популяции создается мутантный вектор путем добавления взвешенной разницы между двумя случайно
    выбранными векторами к третьему вектору.
    Масштабирующий фактор F контролирует степень мутации."""
    mutant_population = np.empty_like(population)
    for i in range(population.shape[0]):
        indices = [idx for idx in range(population.shape[0]) if idx != i]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + F * (b - c)
        mutant_population[i] = np.clip(mutant, -1, 2)  # Обеспечиваем, чтобы мутанты оставались в пределах границ
    return mutant_population

def crossover(mutant_population, population, CR):
    """Применяет кроссовер между мутантной и исходной популяцией.
    Для каждого вектора создается пробный вектор. Каждый элемент пробного вектора либо берется из мутантного вектора (с вероятностью CR), 
    либо из целевого вектора."""
    trial_population = np.empty_like(population)
    for i in range(population.shape[0]):
        cross_points = np.random.rand(population.shape[1]) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, population.shape[1])] = True  # Гарантируем, что хотя бы один элемент берется из мутанта
        trial_population[i] = np.where(cross_points, mutant_population[i], population[i])
    return trial_population

def selection(trial_population, population):
    """Выбирает особи для следующего поколения.
    Для каждой пары пробного и целевого векторов выбирается тот, у которого выше приспособленность, для включения в следующее поколение."""
    new_population = np.empty_like(population)
    for i in range(population.shape[0]):
        if fitness(trial_population[i]) > fitness(population[i]):
            new_population[i] = trial_population[i]
        else:
            new_population[i] = population[i]
    return new_population

def differential_evolution(num_generations, population_size, dim, F, CR):
    """Основная функция алгоритма дифференциальной эволюции.
    Эволюционирует популяцию на протяжении нескольких поколений для нахождения решения, которое максимизирует функцию приспособленности."""
    population = initialize_population(population_size, dim)
    for generation in range(num_generations):
        mutant_population = mutate(population, F)
        trial_population = crossover(mutant_population, population, CR)
        population = selection(trial_population, population)
        best_fitness = np.max([fitness(ind) for ind in population])
        print(f"Поколение {generation + 1}, лучшая приспособленность {best_fitness}")
    return population[np.argmax([fitness(ind) for ind in population])]

# Параметры алгоритма
num_generations = 10
population_size = 50
dim = 1  # Размерность задачи
F = 0.8  # Фактор мутации
CR = 0.9  # Вероятность кроссовера

# Запуск алгоритма
best_solution = differential_evolution(num_generations, population_size, dim, F, CR)
print(f"Лучшее решение: {best_solution}, с приспособленностью: {fitness(best_solution)}")