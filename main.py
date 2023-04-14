from test_functions import *
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

