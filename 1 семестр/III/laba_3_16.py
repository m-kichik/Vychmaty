# 3 раздел, 1 вариант 16 задание

import numpy as np


def f(x, y):
    return (y ** 2 + x * (x - 2) * y) / (x ** 2 * (x - 1))


x_boards = [2., 3.]
epsilon = 1e-4
N = 10
h = (x_boards[1] - x_boards[0]) / N

x_0 = x_boards[0]
y_0 = 4.

x_pre = []
for i in range(N + 1):
    x_pre.append(x_0 + h * i)

y_pre = []
for x in x_pre:
    if x == x_0:
        y_pre.append(y_0)
    else:
        x_n = x - h
        y_n = y_pre[-1]
        f_1 = f(x_n, y_n)
        f_2 = f(x, y_n + h * f_1)
        y_pre.append(y_n + (h / 2) * (f_1 + f_2))

N_new = N * 2
h_new = (x_boards[1] - x_boards[0]) / N_new
x_new = []
for i in range(N_new + 1):
    x_new.append(x_0 + h_new * i)

y = []
for x in x_new:
    if x == x_0:
        y.append(y_0)
    else:
        x_n = x - h_new
        y_n = y[-1]
        f_1 = f(x_n, y_n)
        f_2 = f(x, y_n + h_new * f_1)
        y.append(y_n + (h_new / 2) * (f_1 + f_2))

while True:
    if abs(y_pre[-1] - y[-1]) < epsilon:
        break
    else:
        x_pre = x_new
        y_pre = y
        N_new = N_new * 2
        h_new = (x_boards[1] - x_boards[0]) / N_new
        x_new = []
        for i in range(N_new + 1):
            x_new.append(x_0 + h_new * i)

        y = []
        for x in x_new:
            if x == x_0:
                y.append(y_0)
            else:
                x_n = x - h_new
                y_n = y[-1]
                f_1 = f(x_n, y_n)
                f_2 = f(x, y_n + h_new * f_1)
                y.append(y_n + (h_new / 2) * (f_1 + f_2))

step = int(N_new / 10)
for i in range(11):
    print(f'{x_new[i * step]} {round(y[i * step], 6)}')