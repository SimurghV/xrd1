import numpy as np


def f01(x):  # for simanneal find max or min  best_x
    value = np.sin(x ** 2) * (x ** 2 - 5 * x)
    return value


def f02(x):
    # value = np.sum(np.power(x, 2))
    value = np.power(x-1, 2)
    return value


def f03(x):

    value = -(200 - (x[0] ** 2 + x[1] - 11) ** 2 - (x[0] + x[1] ** 2 - 7) ** 2)
    return value
