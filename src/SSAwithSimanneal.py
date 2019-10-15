import numpy as np
from simanneal import Simanneal
from SSACopy import SSA


def fitness_function(x):
    xx = x
    return np.sum(x**2, axis=1)
    # a = x ** 2
    # return a

def fitness_function02(x):
    xx = x
    return np.sum(x**2)


range_x = [-100, 100]
best_ssa_sa_position = []
best_ssa_sa_fit = []
best_fit = 100000.0

ssa = SSA(fitness_function)
ssa.run()
best_ssa_position = ssa.best_position_list
best_ssa_fit = ssa.best_position_fitness_list

for i in range(np.array(best_ssa_position).shape[0]):
    sa = Simanneal(range_x, best_ssa_position[i], fitness_function02)
    sa.find_best_x()
    best_ssa_sa_position.append(sa.best_x)
    best_ssa_sa_fit.append(sa.best_fit)

    # if sa.best_fit < best_fit:
    #     best_fit = sa.best_fit
    #     best_ssa_sa_position =
print('position: before')
print(best_ssa_position)
print('after')
print(best_ssa_sa_position)
print('\n')
print('fitness: before')
print(best_ssa_fit)
print('after')
print(best_ssa_sa_fit)

