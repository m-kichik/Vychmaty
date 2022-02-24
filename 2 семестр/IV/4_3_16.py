import math
import numpy as np

x_start = [0., 1.]
u_start = [2., 1.]
x_0 = 1 / math.sqrt(3)


def k_q_f(x):
    if x < x_0:
        return math.exp(- x), x ** 3, x ** 2 - 1.
    if x > x_0:
        return math.exp(- x), x, 1.


k_a = math.exp(- x_0)
k_b = math.exp(- x_0)
q_a = x_0 ** 3
q_b = x_0
f_a = x_0 ** 2 - 1.
f_b = 1.

# Аналитическое решение модельной задачи

lambda_a = math.sqrt(q_a / k_a)
lambda_b = math.sqrt(q_b / k_b)

mu_a = f_a / q_a
mu_b = f_b / q_b

A11 = math.exp(- lambda_a * x_0) - math.exp(lambda_a * x_0)
A12 = math.exp(lambda_b * (2 - x_0)) - math.exp(lambda_b * x_0)
A21 = k_a * lambda_a * (math.exp(lambda_a * x_0) + math.exp(- lambda_a * x_0))
A22 = k_b * lambda_b * (math.exp(lambda_b * (2 - x_0)) + math.exp(lambda_b * x_0))

B1 = mu_b - mu_a + (mu_a - u_start[0]) * math.exp(lambda_a * x_0) \
     - (mu_b - u_start[1]) * math.exp(lambda_b * (1 - x_0))
B2 = k_a * lambda_a * (u_start[0] - mu_a) * math.exp(lambda_a * x_0) \
     + k_b * lambda_b * (u_start[1] - mu_b) * math.exp(lambda_b * (1 - x_0))

C1 = (((u_start[0] - mu_a) * A11 - B1) * A22 - ((u_start[0] - mu_a) * A21 - B2) * A12) / (A11 * A22 - A12 * A21)
C2 = (B1 * A22 - B2 * A12) / (A11 * A22 - A12 * A21)
C3 = (B2 * A11 - B1 * A21) / (A11 * A22 - A12 * A21)
C4 = (u_start[1] - mu_b) * math.exp(lambda_b) - C3 * math.exp(2 * lambda_b)


def u_an(x):
    if 0 <= x < x_0:
        return C1 * math.exp(lambda_a * x) + C2 * math.exp(- lambda_a * x) + mu_a
    else:
        if x_0 < x <= 1:
            return C3 * math.exp(lambda_b * x) + C4 * math.exp(- lambda_b * x) + mu_b
        else:
            return 'out of boards'


print('\n\nАналитическое решение модельной задачи:\n\n',
      f'U = {C1:.3f} * exp({lambda_a:.3f} X) + {C2:.3f} * exp({- lambda_a:.3f} X) + {mu_a:.3f}, if 0 < X < {x_0:.3f}\n',
      'and\n',
      f'U = {C3:.3f} * exp({lambda_b:.3f} X) + {C4:.3f} * exp({- lambda_b:.3f} X) + {mu_b:.3f}, if {x_0:.3f} < X < 1\n\n')

# Численное решение модельной задачи с постоянными коэффициентами

N = 640
h = 1 / N
x_l = np.arange(0., 1. + h, h)


def find_l_a_l_b():
    for i in range(len(x_l) - 1):
        if x_l[i] < x_0 < x_l[i + 1]:
            return int(i), int(i) + 1
    return None, None


def find_grid_const():
    l_a, l_b = find_l_a_l_b()

    a_a, a_b = k_a, k_b
    b_a, b_b = - 2 * k_a - q_a * h ** 2, - 2 * k_b - q_b * h ** 2
    c_a, c_b = k_a, k_b
    d_a, d_b = - f_a * h ** 2, - f_b * h ** 2

    alpha_a = [- a_a / b_a, ]
    beta_a = [(d_a - c_a * u_start[0]) / b_a, ]
    alpha_b = [- c_b / b_b, ]
    beta_b = [(d_b - c_b * u_start[1]) / b_b, ]

    for i in range(1, l_a):
        alpha_a.append(- a_a / (b_a + c_a * alpha_a[i - 1]))
        beta_a.append((d_a - c_a * beta_a[i - 1]) / (b_a + c_a * alpha_a[i - 1]))

    for i in reversed(range(l_b + 1, N)):
        alpha_b.insert(0, - c_b / (b_b + a_b * alpha_b[0]))
        beta_b.insert(0, (d_b - c_b * beta_b[0]) / (b_b + a_b * alpha_b[1]))

    u = 2 * [(k_a * beta_a[-1] + k_b * beta_b[0]) / (k_a * (1 - alpha_a[-1]) + k_b * (1 - alpha_b[0]))]

    for i in reversed(range(0, l_a - 1)):
        u.insert(0, alpha_a[i] * u[0] + beta_a[i])

    for i in range(l_b + 1, N):
        u.append(alpha_b[i - l_b] * u[-1] + beta_b[i - l_b])

    u.insert(0, u_start[0]), u.append(u_start[1])
    return u


u_const = find_grid_const()


# Печать сетки

def print_grid_const():
    x_gr = np.arange(0., 1.1, 0.1)
    u1_res = [u_an(x) for x in x_gr]
    u2_res = []
    base = int(N / 10)
    for i in range(0, N + 1, base):
        u2_res.append(u_const[i])
    diff = []
    for i in range(0, 11):
        diff.append(abs(u2_res[i] - u1_res[i]))

    x_str = f'{x_gr[0]:.3e}'
    for x in x_gr[1:]:
        x_str += f' | {x:.3e}'
    u1_str = f'{u1_res[0]:.3e}'
    for u1 in u1_res[1:]:
        u1_str += f' | {u1:.3e}'
    u2_str = f'{u_const[0]:.3e}'
    for u2 in u2_res[1:]:
        u2_str += f' | {u2:.3e}'
    d_str = f'{diff[0]:.3e}'
    for d in diff[1:]:
        d_str += f' | {d:.3e}'

    print('Сетка: \n\n',
          f'X     | {x_str}\n',
          f'U_mod | {u1_str}\n',
          f'U     | {u2_str}\n',
          f'diff  | {d_str}\n', )


print_grid_const()


# Решение задачи с переменными коэффициентами

def find_grid_var():
    l_a, l_b = find_l_a_l_b()

    a_a, a_b = k_a, k_b
    b_a, b_b = - 2 * k_a - q_a * h ** 2, - 2 * k_b - q_b * h ** 2
    c_a, c_b = k_a, k_b
    d_a, d_b = - f_a * h ** 2, - f_b * h ** 2

    alpha_a = [- a_a / b_a, ]
    beta_a = [(d_a - c_a * u_start[0]) / b_a, ]
    alpha_b = [- c_b / b_b, ]
    beta_b = [(d_b - c_b * u_start[1]) / b_b, ]

    def a_b_c_d(x):
        k, q, f = k_q_f(x)
        a = k
        b = - 2 * k - q * h ** 2
        c = k
        d = - f * h ** 2
        return a, b, c, d

    for i in range(1, l_a):
        a, b, c, d = a_b_c_d(x_l[i])
        alpha_a.append(- a / (b + c * alpha_a[i - 1]))
        beta_a.append((d - c * beta_a[i - 1]) / (b + c * alpha_a[i - 1]))

    for i in reversed(range(l_b + 1, N)):
        a, b, c, d = a_b_c_d(x_l[i])
        alpha_b.insert(0, - c / (b + a * alpha_b[0]))
        beta_b.insert(0, (d - c * beta_b[0]) / (b + a * alpha_b[1]))

    u = 2 * [(k_a * beta_a[-1] + k_b * beta_b[0]) / (k_a * (1 - alpha_a[-1]) + k_b * (1 - alpha_b[0]))]

    for i in reversed(range(0, l_a - 1)):
        u.insert(0, alpha_a[i] * u[0] + beta_a[i])

    for i in range(l_b + 1, N):
        u.append(alpha_b[i - l_b] * u[-1] + beta_b[i - l_b])

    u.insert(0, u_start[0]), u.append(u_start[1])
    return u


u_var = find_grid_var()

from matplotlib import pyplot as plt

plt.plot(x_l, u_var, color='black')
plt.savefig('4_3_16')
plt.clf()


def print_grid_var():
    x_gr = np.arange(0., 1.1, 0.1)
    u_res = []
    base = int(N / 10)
    for i in range(0, N + 1, base):
        u_res.append(u_var[i])

    x_str = f'{x_gr[0]:.3e}'
    for x in x_gr[1:]:
        x_str += f' | {x:.3e}'
    u_str = f'{u_res[0]:.3e}'
    for u in u_res[1:]:
        u_str += f' | {u:.3e}'

    print('\nЗадача с переменными коэффициентами: \n\n',
          f'X     | {x_str}\n',
          f'U     | {u_str}\n',)

print_grid_var()
