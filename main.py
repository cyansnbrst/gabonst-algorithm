import numpy as np
import random
from numba import njit


# Функция генерации начальной популяции хромосом (параметры (отрезок) от а до б)
def generate_initial_population(POPULATION_NUMBER):
    population = []
    for i in range(POPULATION_NUMBER):
        chromosome = [random.uniform(A, B) for j in range(CHROMOSOME_LENGTH)]
        population.append(chromosome)
    return population


# Алгоритм мутации гена
def uniform_mutation(chromosome):
    mutated_gene = random.randint(0, len(chromosome) - 1)
    chromosome[mutated_gene] = random.uniform(A, B)
    return chromosome


# Алгоритм кроссинговера
def arithmetic_crossover(chromosome, top_five_chromosomes):
    rschromosome = random.choice(top_five_chromosomes)
    alpha = [random.uniform(-GAMMA, GAMMA + 1) for j in range(CHROMOSOME_LENGTH)]
    offspring = []
    for i in range(CHROMOSOME_LENGTH):
        new_gene = alpha[i] * chromosome[i] + (1 - alpha[i]) * rschromosome[i]
        if new_gene < A:
            new_gene = A
        if new_gene > B:
            new_gene = B
        offspring.append(new_gene)
    return offspring


# Тест-функции

def f1(x):
    d = CHROMOSOME_LENGTH
    s = sum([np.sin(5 * np.pi * xi) ** 6 for xi in x])
    return -1 / d * s

def f3(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

if __name__ == "__main__":
    # 1. Начало работы алгоритма
    print("Genetic Algorithm")

    # 2. Задаем n - число популяций и число итераций NumIter
    CHROMOSOME_LENGTH = 2
    POPULATION_NUMBER = 5
    ITERATION_NUMBER = 10
    GAMMA = 0.4
    A = -10
    B = 10
    TEST_FUNCTION = f3

    # 3. Генерируем популяцию случайным образом
    population = generate_initial_population(POPULATION_NUMBER)
    new_generation = []

    # 6. Сравниваем каждый элемент со средним значением
    iter = 1
    while (iter <= ITERATION_NUMBER):

        # 4. Вычисляем значение пригодности каждой хромосомы
        fitness = [TEST_FUNCTION(chromosome) for chromosome in population]

        # 5. Вычисляем среднее значение пригодности
        mean = sum(fitness) / POPULATION_NUMBER

        for i in range(POPULATION_NUMBER):

            # 7. Если меньше, то мутируем
            if fitness[i] <= mean:
                mutated_chromosome = uniform_mutation(population[i])
                new_generation.append(mutated_chromosome)

            # 8. Первый шанс на улучшение (скрещивание с одной из лучших хромосом)
            else:

                top_five_chromosomes = sorted(population, key=lambda x: TEST_FUNCTION(x))[:5]
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

        population = new_generation
        new_generation = []
        print(mean)
        iter += 1

    # 10. Получаем лучшую хромосому поколения
    print(TEST_FUNCTION(sorted(population)[0]))

