from main import *
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams


def algo_comparing(chromosome_length, population_number, iteration_number, gamma, a, b, test_function):
    x = np.array([i for i in range(1, iteration_number + 1)])
    results, best = genetic_algorithm(chromosome_length, population_number, iteration_number, gamma, a, b,
                                      test_function)
    var_bound = [(a, b) for _ in range(chromosome_length)]
    model = ga(test_function, dimension=chromosome_length,
               variable_type='real',
               variable_boundaries=var_bound,
               algorithm_parameters=AlgorithmParams(max_num_iteration=iteration_number))
    model.run()
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Value")
    ax.plot(x, results, label="default")
    ax.plot(x, model.report, label="pypi ga2")
    ax.legend()
    ax.grid()
    plt.show()


algo_comparing(10, 500, 100, 0.4, -1, 1, f1)
