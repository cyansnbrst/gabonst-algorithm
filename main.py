import numpy as np
import random
from numba import njit


# Функция генерации начальной популяции хромосом (параметры (отрезок) от а до б)
def generate_initial_population(POPULATION_NUMBER):
    population = []
    for i in range(POPULATION_NUMBER):
        chromosome = []
        for j in range(CHROMOSOME_LENGTH):
            chromosome.append(random.uniform(A, B))
        population.append(chromosome)
    return np.array(population)


# Алгоритм мутации гена
@njit
def uniform_mutation(chromosome):
    mutated_gene = random.randint(0, len(chromosome) - 1)
    chromosome[mutated_gene] = random.uniform(A, B)
    return chromosome


# Алгоритм кроссинговера
@njit
def arithmetic_crossover(chromosome, top_five_chromosomes):
    randind = random.randint(0, len(top_five_chromosomes) - 1)
    rschromosome = top_five_chromosomes[randind]
    alpha = [random.uniform(-GAMMA, GAMMA + 1) for j in range(CHROMOSOME_LENGTH)]
    offspring = []
    for i in range(CHROMOSOME_LENGTH):
        new_gene = alpha[i] * chromosome[i] + (1 - alpha[i]) * rschromosome[i]
        if new_gene < A:
            new_gene = A
        if new_gene > B:
            new_gene = B
        offspring.append(new_gene)
    return np.array(offspring)


# Тест-функции
@njit
def f1(x):
    s = sum([np.sin(5 * np.pi * xi) ** 6 for xi in x])
    return -1 / CHROMOSOME_LENGTH * s


@njit
def f3(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2



if __name__ == "__main__":
    # 1. Начало работы алгоритма
    print("Genetic Algorithm")

    # 2. Задаем n - число популяций и число итераций NumIter
    CHROMOSOME_LENGTH = 10
    POPULATION_NUMBER = 400
    ITERATION_NUMBER = 100
    GAMMA = 0.4
    A = -1
    B = 1
    TEST_FUNCTION = f1

    # 3. Генерируем популяцию случайным образом
    population = generate_initial_population(POPULATION_NUMBER)
    new_generation = []
    best = population[0]

    # 6. Сравниваем каждый элемент со средним значением
    iter = 1
    while (iter <= ITERATION_NUMBER):

        # 4. Вычисляем значение пригодности каждой хромосомы
        fitness = np.array([TEST_FUNCTION(chromosome) for chromosome in population])
        best_index = np.where(fitness == min(fitness))[0][0]
        cbest = population[best_index]
        if TEST_FUNCTION(cbest) < TEST_FUNCTION(best):
            best = cbest[::]

        # 5. Вычисляем среднее значение пригодности
        mean = sum(fitness) / POPULATION_NUMBER

        for i in range(POPULATION_NUMBER):

            # 7. Если меньше, то мутируем
            if fitness[i] <= mean:
                mutated_chromosome = uniform_mutation(population[i])
                new_generation.append(mutated_chromosome)

            # 8. Первый шанс на улучшение (скрещивание с одной из лучших хромосом)
            else:
                sorted_indices = np.argsort(np.array(fitness))
                top_five_indices = sorted_indices[:5]
                top_five_chromosomes = population[top_five_indices]
                offspring = arithmetic_crossover(population[i], top_five_chromosomes)
                if TEST_FUNCTION(offspring) <= mean:
                    new_generation.append(offspring)

                # 9. Второй шанс на улучшение - мутация
                else:
                    mutated_chromosome = uniform_mutation(population[i])
                    if TEST_FUNCTION(mutated_chromosome) <= mean:
                        new_generation.append(mutated_chromosome)
                    else:
                        new_generation.append([random.uniform(A, B) for j in range(CHROMOSOME_LENGTH)])

        population = np.array(new_generation[::])
        new_generation = []
        iter += 1

    # 10. Получаем лучшую хромосому поколения
    print(TEST_FUNCTION(best))
