# 3 раздел, 1 вариант 16 задание

from matplotlib import pyplot as plt
import math
import numpy as np


# FUNCTIONS
def f(x, y):
    return (y ** 2 + x * (x - 2) * y) / (x ** 2 * (x - 1))


def make_x(x__0, N, h):
    res = []
    for i in range(N + 1):
        res.append(x__0 + h * i)
    return res


def make_y(x__0, y__0, x_s, h):
    res = []
    for x in x_s:
        if x == x__0:
            res.append(y__0)
        else:
            x_n = x - h
            y_n = res[-1]
            f_1 = f(x_n, y_n)
            f_2 = f(x, y_n + h * f_1)
            res.append(y_n + (h / 2) * (f_1 + f_2))
    return res


def make_h(x_b, N):
    return (x_b[1] - x_b[0]) / N


def select_for_grid_10(x, N):
    x_new = []
    step = int(N / 10)
    for i in range(11):
        x_new.append(x[i * step])

    return x_new


def norm(x_1, x_2):
    x_res = []
    for i in range(len(x_1)):
        x_res.append(abs(x_1[i] - x_2[i]))
    return math.sqrt(sum([x ** 2 for x in x_res]))


def err(y_1, y_2, N):
    return norm(y_1, y_2)# / (2 ** int(math.log2(N / 10)) - 1)


def difference_in_nodes(y_1, y_2):
    return [abs(el[1] - el[0]) for el in zip(y_1, y_2)]


def find_good_grid(x_0, y_0, x_b, y_pre, N_pre, eps):
    N_new = N_pre * 2
    h_new = make_h(x_b, N_new)
    x_new = make_x(x_0, N_new, h_new)
    y_new = make_y(x_0, y_0, x_new, h_new)
    error = err(select_for_grid_10(y_pre, N_new / 2), select_for_grid_10(y_new, N_new), N_new)
    errors = [error]

    while True:
        if error < eps:
            break
        else:
            y_pre = y_new
            N_new = N_new * 2
            h_new = make_h(x_b, N_new)
            x_new = make_x(x_0, N_new, h_new)
            y_new = make_y(x_0, y_0, x_new, h_new)
            error = err(select_for_grid_10(y_pre, N_new / 2), select_for_grid_10(y_new, N_new), N_new)
            errors.append(error)

    return x_new, y_new, N_new, errors


def print_grid(x, y, N):
    print(f'N = {N}')
    for i in range(len(x)):
        print('{:<05.6f} {:<05.6f}'.format(x[i], y[i]))
    print()


def print_diff(x_, y_1, y_2):
    diff = difference_in_nodes(y_1, y_2)
    print('x        y1        y2    difference')
    for i in range(len(x_)):
        print('{}  {:<05.6f}  {:<05.6f}  {:<05.6f}'.format(x[i], y_1[i], y_2[i], diff[i]))

# START CONDITIONS
x_boards = [2. , 3.]
epsilon = 1e-4
N = 10
h = make_h(x_boards, N)

x_0 = x_boards[0]
y_0 = 4.
x = make_x(x_0, N, h)
y = make_y(x_0, y_0, x, h)

# BEST GRID
N_sample = 10240
h_sample = make_h(x_boards, N_sample)
x_sample = make_x(x_0, N_sample, h_sample)
y_sample = make_y(x_0, y_0, x_sample, h_sample)

print('Эталонная сетка:')
print_grid(select_for_grid_10(x_sample, N_sample), select_for_grid_10(y_sample, N_sample), N_sample)

plt.plot(select_for_grid_10(x_sample, N_sample), select_for_grid_10(y_sample, N_sample), color='black')
plt.scatter(select_for_grid_10(x_sample, N_sample), select_for_grid_10(y_sample, N_sample), color='red')
plt.savefig('3_16_et')
plt.clf()

# FIND SOLUTION FOR START CONDITIONS
x_good, y_good, N_good, errs = find_good_grid(x_0, y_0, x_boards, y, N, epsilon)

print_grid(select_for_grid_10(x_good, N_good), select_for_grid_10(y_good, N_good), N_good)

plt.scatter(np.arange(0, len(errs), 1.), errs, color='red')
plt.savefig('3_16_errs')

# N_1 = 20
# h_1 = make_h(x_boards, N_1)
# x_1 = make_x(x_0, N_1, h_1)
# y_1 = make_y(x_0, y_0, x_1, h_1)
#
# N_2 = 2 * N_1
# h_2 = make_h(x_boards, N_2)
# x_2 = make_x(x_0, N_2, h_2)
# y_2 = make_y(x_0, y_0, x_2, h_2)

value = int(input('Если вы хотите получить сетку для произвольного N, введите 0\n'))
              #'если вы хотите получить значения функции в h и 2h, введите 1 : '))

if value == 0:
    N = int(input('Введите N : '))
    h = make_h(x_boards, N)
    x = make_x(x_0, N, h)
    y = make_y(x_0, y_0, x, h)
    print('Полученная сетка: ')
    print_grid(x, y, N)
else:
    N_1 = int(input('Введите N : '))
    h_1 = make_h(x_boards, N)
    x_1 = make_x(x_0, N, h)
    y_1 = make_y(x_0, y_0, x, h)

    N_2 = 2 * N_1
    h_2 = make_h(x_boards, N_2)
    x_2 = make_x(x_0, N_2, h_2)
    y_2 = make_y(x_0, y_0, x_2, h_2)

    print_diff(select_for_grid_10(x_1, N_1), select_for_grid_10(y_1, N_1), select_for_grid_10(y_2, N_2))
