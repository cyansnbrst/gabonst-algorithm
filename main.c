#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define MAX_CHROMOSOME_LENGTH 20

float rand_double(float a, float b) {
    return a + ((float) rand() / RAND_MAX) * (b - a);
}

float *generate_chromosome(int chromosome_length, float a, float b) {
    float *chromosome = (float *) malloc(sizeof(float) * chromosome_length);
    for (int i = 0; i < chromosome_length; i++) {
        chromosome[i] = rand_double(a, b);
    }
    return chromosome;
}

float **generate_initial_population(int population_number, int chromosome_length, float a, float b) {
    float **population = (float **) malloc(sizeof(float *) * population_number);
    for (int i = 0; i < population_number; i++) {
        population[i] = generate_chromosome(chromosome_length, a, b);
    }
    return population;
}

float *uniform_mutation(float *chromosome, int chromosome_length, float a, float b) {
    int mutated_gene = rand() % chromosome_length;
    chromosome[mutated_gene] = rand_double(a, b);
    return chromosome;
}

float *
arithmetic_crossover(float *chromosome, float top_five_chromosomes[5][MAX_CHROMOSOME_LENGTH], int chromosome_length,
                     float gamma, float a, float b) {
    int randind = rand() % 5;
    float *rchromosome = top_five_chromosomes[randind];
    float alpha[chromosome_length];
    for (int i = 0; i < chromosome_length; ++i) {
        alpha[i] = rand_double(-gamma, gamma + 1);
    }
    float *offspring = (float *) malloc(sizeof(float) * chromosome_length);
    float new_gene;
    for (int i = 0; i < chromosome_length; ++i) {
        new_gene = alpha[i] * chromosome[i] + (1 - alpha[i]) * rchromosome[i];
        if (new_gene < a)
            new_gene = a;
        if (new_gene > b)
            new_gene = b;
        offspring[i] = new_gene;
    }
    return offspring;
}


float f3(float *x) {
    return pow((x[0] + 2 * x[1] - 7), 2) + pow((2 * x[0] + x[1] - 5), 2);
}

float f14(float *x) {
    float term1, term2, term3, term4, term5;
    term1 = 4 * pow(x[0], 2);
    term2 = -2.1 * pow(x[0], 4);
    term3 = 1 / 3.0 * pow(x[0], 6);
    term4 = x[0] * x[1];
    term5 = -4 * pow(x[1], 2) + 4 * pow(x[1], 4);
    return term1 + term2 + term3 + term4 + term5;
}

float
genetic_algorithm(int chromosome_length, int population_number, int iteration_number, float gamma, float a, float b,
                  float (*test_function)(float *)) {
    srand(time(NULL));

    float **population = generate_initial_population(population_number, chromosome_length, a, b);
    float **new_generation = (float **) malloc(sizeof(float *) * population_number);

    float *best = population[0];

    int iter = 1;
    while (iter <= iteration_number) {

        int cc = 0;

        float fitness[population_number];
        for (int i = 0; i < population_number; ++i) {
            fitness[i] = test_function(population[i]);
        }

        float *cbest;
        for (int i = 0; i < population_number; ++i) {
            if (fitness[i] < test_function(best)) {
                cbest = population[i];
            }
        }

        if (test_function(cbest) < test_function(best)) {
            memcpy(best, cbest, sizeof(float) * chromosome_length);
        }

        float mean = 0;
        for (int i = 0; i < population_number; ++i) {
            mean += fitness[i];
        }
        mean /= population_number;

        for (int i = 0; i < population_number; ++i) {
            if (fitness[i] < mean) {
                float *mutated_chromosome = uniform_mutation(population[i], chromosome_length, a, b); // явное копирование
                new_generation[cc] = mutated_chromosome;
                ++cc;

            } else {
                float *temp;
                float **sorted_array = (float **) malloc(sizeof(float *) * population_number);
                memcpy(sorted_array, population, sizeof(float *) * population_number);
                _Bool no_swap;
                for (int j = population_number - 1; j >= 0; --j) {
                    no_swap = 1;
                    for (int k = 0; k < i; ++k) {
                        if (test_function(sorted_array[k]) > test_function(sorted_array[k + 1])) {
                            temp = sorted_array[k];
                            sorted_array[k] = sorted_array[k + 1];
                            sorted_array[k + 1] = temp;
                            no_swap = 0;
                        }
                    }
                    if (no_swap == 1)
                        break;
                }

                float top_five_chromosomes[5][MAX_CHROMOSOME_LENGTH] = {0};
                for (int j = 0; j < 5; ++j) {
                    memcpy(top_five_chromosomes[j], sorted_array[j], sizeof(float) * chromosome_length);
                }

                float *offspring = arithmetic_crossover(population[i], top_five_chromosomes, chromosome_length, gamma, a, b);
                if (test_function(offspring) <= mean) {
                    new_generation[cc] = offspring;
                    ++cc;
                } else {
                    float *mutated_chromosome = uniform_mutation(population[i], chromosome_length, a, b);
                    if (test_function(mutated_chromosome) <= mean) {
                        new_generation[cc] = mutated_chromosome;
                        ++cc;
                    } else {
                        new_generation[cc] = generate_chromosome(chromosome_length, a, b);
                        ++cc;
                    }
                }

            }
        }
        memcpy(population, new_generation, sizeof(float *) * population_number);
        memset(new_generation, 0, population_number);
        ++iter;

    }
    printf("%f", test_function(best));
}

int main() {
    genetic_algorithm(2, 50, 400, 0.4f, -5.f, 5.0f, f14);
    return 0;

}