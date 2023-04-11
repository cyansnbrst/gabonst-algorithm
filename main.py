import numpy as np
import random
from numba import njit
import matplotlib.pyplot as plt
import os
import ctypes
import time


# Функция генерации начальной популяции хромосом (параметры (отрезок) от а до б)
def generate_initial_population(population_number, chromosome_length, a, b):
    population = []
    for i in range(population_number):
        chromosome = []
        for j in range(chromosome_length):
            chromosome.append(random.uniform(a, b))
        population.append(chromosome)
    return np.array(population)


# Алгоритм мутации гена
@njit
def uniform_mutation(chromosome, a, b):
    mutated_gene = random.randint(0, len(chromosome) - 1)
    chromosome[mutated_gene] = random.uniform(a, b)
    return chromosome


# Алгоритм кроссинговера
@njit
def arithmetic_crossover(chromosome, top_five_chromosomes, gamma, chromosome_length, a, b):
    randind = random.randint(0, len(top_five_chromosomes) - 1)
    rschromosome = top_five_chromosomes[randind]
    alpha = [random.uniform(-gamma, gamma + 1) for j in range(chromosome_length)]
    offspring = []
    for i in range(chromosome_length):
        new_gene = alpha[i] * chromosome[i] + (1 - alpha[i]) * rschromosome[i]
        if new_gene < a:
            new_gene = a
        if new_gene > b:
            new_gene = b
        offspring.append(new_gene)
    return np.array(offspring)


def genetic_algorithm(chromosome_length, population_number, iteration_number, gamma, a, b, test_function):
    # 1. Начало работы алгоритма
    results = []

    # 3. Генерируем популяцию случайным образом
    population = generate_initial_population(population_number, chromosome_length, a, b)
    new_generation = []
    best = population[0]

    # 6. Сравниваем каждый элемент со средним значением
    iteration = 1
    while iteration <= iteration_number:

        # 4. Вычисляем значение пригодности каждой хромосомы
        fitness = np.array([test_function(chromosome) for chromosome in population])
        best_index = np.argmin(fitness)
        cbest = population[best_index]
        if test_function(cbest) < test_function(best):
            best = np.copy(cbest)

        # 5. Вычисляем среднее значение пригодности
        mean = sum(fitness) / population_number

        for i in range(population_number):

            # 7. Если меньше, то мутируем
            if fitness[i] <= mean:
                mutated_chromosome = uniform_mutation(population[i], a, b)
                new_generation.append(mutated_chromosome)

            # 8. Первый шанс на улучшение (скрещивание с одной из лучших хромосом)
            else:
                sorted_indices = np.argsort(np.array(fitness))
                top_five_indices = sorted_indices[:5]
                top_five_chromosomes = population[top_five_indices]
                offspring = arithmetic_crossover(population[i], top_five_chromosomes, gamma, chromosome_length, a, b)
                if test_function(offspring) <= mean:
                    new_generation.append(offspring)

                # 9. Второй шанс на улучшение - мутация
                else:
                    mutated_chromosome = uniform_mutation(population[i], a, b)
                    if test_function(mutated_chromosome) <= mean:
                        new_generation.append(mutated_chromosome)
                    else:
                        new_generation.append([random.uniform(a, b) for _ in range(chromosome_length)])

        population = np.copy(np.array(new_generation))
        new_generation = []
        iteration += 1

        results.append(test_function(best))

    # 10. Получаем лучшую хромосому поколения
    return results, best


def test_results(chromosome_length, population_number, iteration_number, gamma, a, b, test_function):
    x = np.array([i for i in range(1, iteration_number + 1)])
    results, best = genetic_algorithm(chromosome_length, population_number, iteration_number, gamma, a, b,
                                      test_function)
    plt.xlabel("Iteration number")
    plt.ylabel("Value")
    plt.plot(x, results)
    plt.grid()
    plt.show()
    print("x: ", best)
    print("f(x): ", test_function(best))


# Тест-функции
@njit
def f1(x):
    s = sum([np.sin(5 * np.pi * xi) ** 6 for xi in x])
    return -1 / len(x) * s


@njit
def f3(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


@njit
def f4(x):
    return 1 / 2 * sum([xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x])


@njit
def f5(x):
    return np.prod(np.array([np.sqrt(xi) * np.sin(xi) for xi in x]))


@njit
def f6(x):
    return sum([xi ** 2 for xi in x])


@njit
def f7(x):
    s = 0
    for i in range(len(x) - 1):
        s += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return s


@njit
def f8(x):
    s = 0
    for j in range(len(x)):
        s += j * x[j] ** 4 + random.uniform(0, 1)
    return s


@njit
def f9(x):
    return sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])


@njit
def f10(x):
    a = sum([xi ** 2 for xi in x])
    b = sum([np.cos(2 * np.pi * xi) for xi in x])
    return -20 * np.exp(-0.2 * np.sqrt(a / len(x))) - np.exp(b / len(x)) + 20 + np.e


@njit
def f11(x):
    a = sum([xi ** 2 for xi in x])
    b = 1
    for j in range(len(x)):
        b *= np.cos(x[j] / np.sqrt(j + 1))
    return 1 / 4000 * a - b + 1


@njit
def f12(x):
    return (1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
            30 + (2 * x[0] - 3 * x[1]) ** 2 * (
            18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


@njit
def f14(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + 1 / 3 * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


@njit
def f15(x):
    return (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


# CHROMOSOME_LENGTH, POPULATION_NUMBER, ITERATION_NUMBER, GAMMA, A, B, TEST_FUNCTION

def c(file, name, types, result):
    path = os.path.abspath(file)
    module = ctypes.cdll.LoadLibrary(path)
    func = module[name]
    func.argtypes = types
    func.restype = result
    return func


function_type = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(ctypes.c_float))

c_genetic_algorithm = c("C:/Users/PC/CLionProjects/gabonst/libcode.dll", "genetic_algorithm",
                        (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float),
                        function_type)


def c10(x):
    a = sum([xi ** 2 for xi in x])
    b = sum([np.cos(2 * np.pi * xi) for xi in x])
    return -20 * np.exp(-0.2 * np.sqrt(a / len(x))) - np.exp(b / len(x)) + 20 + np.e


def c11(x):
    a = sum([xi ** 2 for xi in x])
    b = 1
    for j in range(len(x)):
        b *= np.cos(x[j] / np.sqrt(j + 1))
    return 1 / 4000 * a - b + 1


def c12(x):
    return (1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
            30 + (2 * x[0] - 3 * x[1]) ** 2 * (
            18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


def c14(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + 1 / 3 * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


def c15(x):
    return (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


c_times = []
times = []

start_time = time.time()
c_genetic_algorithm(2, 50, 100, 0.4, -2, 2, function_type(c12))
end_time = time.time()

c_times.append(end_time - start_time)

start_time = time.time()
c_genetic_algorithm(2, 50, 100, 0.4, -5, 5, function_type(c14))
end_time = time.time()

c_times.append(end_time - start_time)

start_time = time.time()
c_genetic_algorithm(2, 50, 100, 0.4, -5, 5, function_type(c15))
end_time = time.time()

c_times.append(end_time - start_time)

start_time = time.time()
genetic_algorithm(2, 50, 100, 0.4, -2, 2, f12)
end_time = time.time()

times.append(end_time - start_time)

start_time = time.time()
genetic_algorithm(2, 50, 100, 0.4, -5, 5, f14)
end_time = time.time()

times.append(end_time - start_time)

start_time = time.time()
genetic_algorithm(2, 50, 100, 0.4, -5, 5, f15)
end_time = time.time()

times.append(end_time - start_time)

x1 = np.arange(3) - 0.2
x2 = np.arange(3) + 0.2
y1 = c_times
y2 = times

fig, ax = plt.subplots()

ax.bar(x1, y1, width=0.4)
ax.bar(x2, y2, width=0.4)

ax.set_facecolor('seashell')
fig.set_figwidth(12)
fig.set_figheight(6)
fig.set_facecolor('floralwhite')

ax.legend(['С', 'Python + Numba'])

plt.show()



