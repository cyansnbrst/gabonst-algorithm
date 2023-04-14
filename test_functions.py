import numpy as np
import random
from numba import njit


# Тест-функции
@njit
def f1(x):
    s = sum([np.sin(5 * np.pi * xi) ** 6 for xi in x])
    return -1 / len(x) * s


def c1(x):
    s = sum([np.sin(5 * np.pi * xi) ** 6 for xi in x])
    return -1 / len(x) * s


@njit
def f3(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def c3(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


@njit
def f4(x):
    return 1 / 2 * sum([xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x])


def c4(x):
    return 1 / 2 * sum([xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x])


@njit
def f5(x):
    return np.prod(np.array([np.sqrt(xi) * np.sin(xi) for xi in x]))


def c5(x):
    return np.prod(np.array([np.sqrt(xi) * np.sin(xi) for xi in x]))


@njit
def f6(x):
    return sum([xi ** 2 for xi in x])


def c6(x):
    return sum([xi ** 2 for xi in x])


@njit
def f7(x):
    s = 0
    for i in range(len(x) - 1):
        s += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return s


def c7(x):
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


def c8(x):
    s = 0
    for j in range(len(x)):
        s += j * x[j] ** 4 + random.uniform(0, 1)
    return s


@njit
def f9(x):
    return sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])


def c9(x):
    return sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x])


@njit
def f10(x):
    a = sum([xi ** 2 for xi in x])
    b = sum([np.cos(2 * np.pi * xi) for xi in x])
    return -20 * np.exp(-0.2 * np.sqrt(a / len(x))) - np.exp(b / len(x)) + 20 + np.e


def c10(x):
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


def c11(x):
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


def c12(x):
    return (1 + (x[0] + x[1] + 1) ** 2 * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
            30 + (2 * x[0] - 3 * x[1]) ** 2 * (
            18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


@njit
def f14(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + 1 / 3 * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


def c14(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + 1 / 3 * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


@njit
def f15(x):
    return (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


def c15(x):
    return (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
