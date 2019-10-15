import numpy as np
import random
import matplotlib.pyplot as plt
# from finess_function import f01
from finess_function import f02
# from finess_function import f03

class Simanneal(object):

    def __init__(self, x_range, init_x, func):
        self.x_range = x_range
        self.max_t = 10000
        self.min_t = 1
        self.iter_num = 1000
        self.cool_rate = 0.95
        self.init_x = init_x
        self.best_x = self.init_x
        self.func = func
        self.best_hist_t = []
        self.best_fit = 0

    def find_best_x(self):
        x1 = self.init_x
        T = self.max_t
        while T >= self.min_t:
            for i in range(self.iter_num):
                f1 = self.func(x1)
                dx = np.random.random(x1.shape) * 0.1 - np.ones(x1.shape) * 0.05  # 随机数dx的大小与实际问题相关，此处简略取[-1， 1]之间的随机数
                for j in range(x1.size):
                    if self.x_range[0] <= (x1[j] + dx[j]) <= self.x_range[1]:
                        dx[j] *= -1
                x2 = x1 - dx

                f2 = self.func(x2)
                df = f2 - f1
                if df < 0:
                    x1 = x2
                else:
                    probability = np.exp(-df / T)
                    if probability > random.random():
                        x1 = x2
            T *= self.cool_rate
            self.best_hist_t.append(x1)
        self.best_x = x1
        self.best_fit = self.func(self.best_x)

    def display(self):
        print('init_x: {}\nbest_x: {}'.format(self.init_x, self.best_x))
        plt.figure(figsize=(6, 4))
        x = np.linspace(self.x_range[0], self.x_range[1], 300)
        y = self.func(x)
        plt.plot(x, y, 'g-', label='finess_function')
        plt.plot(self.init_x, self.func(self.init_x), 'bo', label='init_x')
        plt.plot(self.best_x, self.func(self.best_x), 'r*', label='best_x')
        plt.title('best_x = {}'.format(self.best_x))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('Simanneal.png', dpi=500)
        plt.show()
        plt.close()


if __name__ == '__main__':
    range_x = [-5, 5]
    dim = 1
    x0 = np.random.rand(dim) * (range_x[1] - range_x[0]) - range_x[1]  # 此处x0为测试用
    # x0 = np.array([4])
    sim = Simanneal(range_x, x0, f02)
    sim.find_best_x()
    sim.display()
