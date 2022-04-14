# задание 12, нужно сделать В3 на отл, В2 на хор и В1 на уд
import math
import numpy as np
from matplotlib import pyplot as plt


def u_an(x, t):
    return math.exp(2 * t) * (x + 4) ** 2


def u_n1_l(h, t, x, u_n):
    return u_n[0] + \
           (t / (6 * h)) * (1 + t / 2 + (t ** 2) / 6) * (x + 4) * (
                   2 * u_n[3] - 9 * u_n[2] + 18 * u_n[1] - 11 * u_n[0]) + \
           ((t ** 2) / (2 * h ** 2)) * (1 + t) * ((x + 4) ** 2) * (- u_n[3] + 4 * u_n[2] - 5 * u_n[1] + 2 * u_n[0]) + \
           ((t ** 3) / (6 * h ** 3)) * ((x + 4) ** 3) * (u_n[3] - 3 * u_n[2] + 3 * u_n[1] - u_n[0])


def u_n_l(t):
    return 25 * math.exp(2 * t)


def u_n_l1(h, t):
    # return - 10 * h * math.exp(2 * t)  + (h ** 2) * math.exp(2 * t)
    return ((h - 5) ** 2) * math.exp(2 * t)


def u_n_l2(h, t):
    # return - 20 * h * math.exp(2 * t) + 4 * (h ** 2) * math.exp(2 * t)
    return ((2 * h - 5) ** 2) * math.exp(2 * t)


def u_num(h, tau, X, T):
    U_grid = [[(x + 4) ** 2 for x in X]]
    for t in T[1:]:
        u_grid = []
        for i in range(len(X[:-3])):
            u_grid.append(
                u_n1_l(h, tau, X[i], (U_grid[-1][i], U_grid[-1][i + 1], U_grid[-1][i + 2], U_grid[-1][i + 3])))
        u_grid.append(u_n_l2(h, t))
        u_grid.append(u_n_l1(h, t))
        u_grid.append(u_n_l(t))
        U_grid.append(u_grid)

    return U_grid[-1]


def print_results(X, U_a, U_n):
    l = len(X)
    h = (l - 1) / 10
    x_str = 'X    '
    u_a_str = 'U_a  '
    u_n_str = 'U_n  '
    diff_str = 'diff '
    diff = []
    for i in range(11):
        x_str += f'| {X[int(i * h)]:.3e} '
        u_a_str += f'| {U_a[int(i * h)]:.3e} '
        u_n_str += f'| {U_n[int(i * h)]:.3e} '
        diff_str += f'| {(abs(U_n[int(i * h)] - U_a[int(i * h)])):.3e} '
        diff.append((abs(U_n[int(i * h)] - U_a[int(i * h)])))

    # print(x_str, u_a_str, u_n_str, diff_str, f'max_diff: {max(diff):.3e}', sep='\n')
    print(x_str, u_a_str, u_n_str, diff_str, f'max_diff: {find_diff(U_a, U_n):.3e}', sep='\n')


def find_diff(U_a, U_n):
    h = (len(U_a) - 1) / 10
    # diff = [(abs(U_n[int(i * h)] - U_a[int(i * h)])) for i in range(11)]
    return max([(abs(U_n[int(i * h)] - U_a[int(i * h)])) for i in range(11)])


def draw_diff():
    print('\nStart calculating diffs...')

    # K = np.arange(0.2, 0.45, 0.05)
    # K = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.45, 0.55, 0.6, 0.65] # что не так с 0.5 и 0.15 ?????
    K = [0.25, 0.3, 0.35, 0.4]
    # K = [0.1]
    L = [640, 1280, 2560, 5120]
    # L = [10 * (2 ** n) for n in range(6)]
    for k in K:
        print(f'calculating for K = {k:.2f}...')
        diffs = []
        for l in L:
            h = 1 / l
            tau = k * h
            X = np.arange(0, 1 + h, h)
            T = np.arange(0, 1 + tau, tau)
            U_a = u_an(X, 1)
            U_n = u_num(h, tau, X, T)
            diffs.append(find_diff(U_a, U_n))
        log_diff = [math.log(d, 10) for d in diffs]
        log_L = [math.log(l / 10, 2) for l in L]
        # plt.plot(log_L, log_diff, label=f'K = {k:.2f}')
        plt.plot(log_L, diffs, label=f'K = {k:.2f}')
        plt.legend(loc='best')
        print(f'd: {diffs}')
    plt.savefig('5_12_diff_eval_tar')
    plt.clf()
    print('Plot successfully created')


if __name__ == '__main__':
    L = 320
    # K = 1 / 3
    K = 0.1
    h = 1 / L
    tau = K * h
    X = np.arange(0, 1 + h, h)
    T = np.arange(0, 1 + tau, tau)
    U_a = u_an(X, 1)
    U_n = u_num(h, tau, X, T)

    print_results(X, U_a, U_n)

    plt.plot(X, U_a, color='black')
    plt.plot(X, U_n, color='orange')
    plt.savefig('5_12')
    plt.clf()

    draw_diff()
