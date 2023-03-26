#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define POPULATION_NUMBER 50
#define CHROMOSOME_LENGTH 2
#define ITERATION_NUMBER 400
#define A -5.0
#define B 5.0
#define GAMMA 0.4

double rand_double(double a, double b) {
    return a + ((double) rand() / RAND_MAX) * (b - a);
}

double *generate_chromosome() {
    double *chromosome = (double *) malloc(sizeof(double) * CHROMOSOME_LENGTH);
    for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
        chromosome[i] = rand_double(A, B);
    }
    return chromosome;
}

double **generate_initial_population() {
    double **population = (double **) malloc(sizeof(double *) * POPULATION_NUMBER);
    for (int i = 0; i < POPULATION_NUMBER; i++) {
        population[i] = generate_chromosome();
    }
    return population;
}

double *uniform_mutation(double *chromosome) {
    int mutated_gene = rand() % CHROMOSOME_LENGTH;
    chromosome[mutated_gene] = rand_double(A, B);
    return chromosome;
}

double *arithmetic_crossover(double *chromosome, double top_five_chromosomes[5][CHROMOSOME_LENGTH]) {
    int randind = rand() % 5;
    double *rchromosome = top_five_chromosomes[randind];
    double alpha[CHROMOSOME_LENGTH];
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        alpha[i] = rand_double(-GAMMA, GAMMA + 1);
    }
    double *offspring = (double *) malloc(sizeof(double) * CHROMOSOME_LENGTH);
    double new_gene;
    for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
        new_gene = alpha[i] * chromosome[i] + (1 - alpha[i]) * rchromosome[i];
        if (new_gene < A)
            new_gene = A;
        if (new_gene > B)
            new_gene = B;
        offspring[i] = new_gene;
    }
    return offspring;
}


double f3(double *x) {
    return pow((x[0] + 2 * x[1] - 7), 2) + pow((2 * x[0] + x[1] - 5), 2);
}

double f14(double *x) {
    double term1, term2, term3, term4, term5;
    term1 = 4 * pow(x[0], 2);
    term2 = -2.1 * pow(x[0], 4);
    term3 = 1 / 3.0 * pow(x[0], 6);
    term4 = x[0] * x[1];
    term5 = -4 * pow(x[1], 2) + 4 * pow(x[1], 4);
    return term1 + term2 + term3 + term4 + term5;
}

int main() {
    printf("Genetical algorithm\n");
    srand(time(NULL));

    double **population = generate_initial_population();
    double **new_generation = (double **) malloc(sizeof(double *) * POPULATION_NUMBER);

    double *best = population[0];

    int iter = 1;
    while (iter <= ITERATION_NUMBER) {

        int cc = 0;

        double fitness[POPULATION_NUMBER];
        for (int i = 0; i < POPULATION_NUMBER; ++i) {
            fitness[i] = f14(population[i]);
        }

        double *cbest;
        for (int i = 0; i < POPULATION_NUMBER; ++i) {
            if (fitness[i] < f14(best)) {
                cbest = population[i];
            }
        }

        if (f14(cbest) < f14(best)) {
            memcpy(best, cbest, sizeof(double) * CHROMOSOME_LENGTH);
        }

        double mean = 0;
        for (int i = 0; i < POPULATION_NUMBER; ++i) {
            mean += fitness[i];
        }
        mean /= POPULATION_NUMBER;

        for (int i = 0; i < POPULATION_NUMBER; ++i) {
            if (fitness[i] < mean) {
                double *mutated_chromosome = uniform_mutation(population[i]); // явное копирование
                new_generation[cc] = mutated_chromosome;
                ++cc;

            } else {
                double *temp;
                double **sorted_array = (double **) malloc(sizeof(double *) * POPULATION_NUMBER);
                memcpy(sorted_array, population, sizeof(double *) * POPULATION_NUMBER);
                _Bool no_swap;
                for (int j = POPULATION_NUMBER - 1; j >= 0; --j) {
                    no_swap = 1;
                    for (int k = 0; k < i; ++k) {
                        if (f14(sorted_array[k]) > f14(sorted_array[k + 1])) {
                            temp = sorted_array[k];
                            sorted_array[k] = sorted_array[k + 1];
                            sorted_array[k + 1] = temp;
                            no_swap = 0;
                        }
                    }
                    if (no_swap == 1)
                        break;
                }

                double top_five_chromosomes[5][CHROMOSOME_LENGTH] = {0};
                for (int j = 0; j < 5; ++j) {
                    memcpy(top_five_chromosomes[j], sorted_array[j], sizeof(double) * CHROMOSOME_LENGTH);
                }

                double *offspring = arithmetic_crossover(population[i], top_five_chromosomes);
                if (f14(offspring) <= mean) {
                    new_generation[cc] = offspring;
                    ++cc;
                } else {
                    double *mutated_chromosome = uniform_mutation(population[i]);
                    if (f14(mutated_chromosome) <= mean) {
                        new_generation[cc] = mutated_chromosome;
                        ++cc;
                    } else {
                        new_generation[cc] = generate_chromosome();
                        ++cc;
                    }
                }

            }
        }
        memcpy(population, new_generation, sizeof(double *) * POPULATION_NUMBER);
        memset(new_generation, 0, POPULATION_NUMBER);
        ++iter;

    }
    printf("%f", f14(best));
    return 0;

}