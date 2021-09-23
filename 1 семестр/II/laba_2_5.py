# 2 раздел 1.5 (5 задание)
from matplotlib import pyplot as plt
import numpy as np

verbose = 1

# ALGEBRAIC INTERPOLATION

x = [0.5236, 0.87267, 1.22173, 1.57080, 1.91986, 2.26893, 2.61799]
y = [0.00010, 0.00112, 0.00687, 0.03018, 0.10659, 0.3207, 0.85128]

F_2 = [['f(x0, x1)', 0],
       ['f(x1, x2)', 0],
       ['f(x2, x3)', 0],
       ['f(x3, x4)', 0],
       ['f(x4, x5)', 0],
       ['f(x5, x6)', 0]
       ]
for i in range(len(F_2)):
    F_2[i][1] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])

F_3 = [['f(x0, x1, x2)', 0],
       ['f(x1, x2, x3)', 0],
       ['f(x2, x3, x4)', 0],
       ['f(x3, x4, x5)', 0],
       ['f(x4, x5, x6)', 0]
       ]
for i in range(len(F_3)):
    F_3[i][1] = (F_2[i + 1][1] - F_2[i][1]) / (x[i + 2] - x[i])

F_4 = [['f(x0, x1, x2, x3)', 0],
       ['f(x1, x2, x3, x4)', 0],
       ['f(x2, x3, x4, x5)', 0],
       ['f(x3, x4, x5, x6)', 0]
       ]
for i in range(len(F_4)):
    F_4[i][1] = (F_3[i + 1][1] - F_3[i][1]) / (x[i + 3] - x[i])

F_5 = [['f(x0, x1, x2, x3, x4)', 0],
       ['f(x1, x2, x3, x4, x5)', 0],
       ['f(x2, x3, x4, x5, x6)', 0]
       ]
for i in range(len(F_5)):
    F_5[i][1] = (F_4[i + 1][1] - F_4[i][1]) / (x[i + 4] - x[i])

F_6 = [['f(x0, x1, x2, x3, x4, x5)', 0],
       ['f(x1, x2, x3, x4, x5, x6)', 0]
       ]
for i in range(len(F_6)):
    F_6[i][1] = (F_5[i + 1][1] - F_5[i][1]) / (x[i + 5] - x[i])

F_7 = [['f(x0, x1, x2, x3, x4, x5, x6)', (F_6[1][1] - F_6[0][1]) / (x[6] - x[0])]]

b_lib = [y[0], F_2[0][1], F_3[0][1], F_4[0][1], F_5[0][1], F_6[0][1], F_7[0][1]]

if verbose:
    print(f'b_0 = {b_lib[0]}\n'
          f'b_1 = {b_lib[1]}\n'
          f'b_2 = {b_lib[2]}\n'
          f'b_3 = {b_lib[3]}\n'
          f'b_4 = {b_lib[4]}\n'
          f'b_5 = {b_lib[5]}\n'
          f'b_6 = {b_lib[6]}\n')

poly_lib = [[1, -x[0]],
            [1, -x[1]],
            [1, -x[2]],
            [1, -x[3]],
            [1, -x[4]],
            [1, -x[5]]]
poly = np.poly1d([0.])
for i in range(len(b_lib)):
    tmp_poly = 1.
    for j in range(i):
        p = np.poly1d(poly_lib[j])
        tmp_poly *= p

    tmp_poly *= b_lib[i]
    poly += tmp_poly

print(f'poly_coeffs = {poly.coeffs}\n')

if verbose:
    for num in x:
        print(f'P({num}) = {poly(num)}')
    print('\n')

if verbose:
    plt.plot(x, y, 'o', np.arange(x[0], x[6], 0.01), poly(np.arange(x[0], x[6], 0.01)), '--')
    plt.grid(True)
    plt.savefig('2_5_alg_interpolation_test')
    plt.clf()


# SPLINES

poly_deriv = poly.deriv(1)
derivs_lib = [poly_deriv(num) for num in x]
if verbose:
    print(f'derivs_lib = {derivs_lib}\n')

def find_spl_coeffs(x, f, P):
    den = (x[1] - x[0]) ** 3

    a_3 = ((P[1] + P[0]) * (x[1] - x[0]) - 2 * (f[1] - f[0])) / den
    a_2 = (3 * (f[1] - f[0]) * (x[1] + x[0]) - (P[1] * (x[1] + 2 * x[0]) + P[0] * (x[0] + 2 * x[1])) * (
            x[1] - x[0])) / den
    a_1 = ((P[1] * x[0] * (2 * x[1] + x[0]) + P[0] * x[1] * (x[1] + 2 * x[0])) * (x[1] - x[0]) - 6 * (f[1] - f[0]) * x[
        0] * x[1]) / den
    a_0 = ((f[1] * x[0] ** 2 * (3 * x[1] - x[0]) + f[0] * x[1] ** 2 * (x[1] - 3 * x[0])) - (
            P[1] * x[0] ** 2 * x[1] + P[0] * x[0] * x[1] ** 2) * (x[1] - x[0])) / den

    return [a_3, a_2, a_1, a_0]


splines = []
for i in range(len(x) - 1):
    coeffs = find_spl_coeffs([x[i], x[i + 1]], [y[i], y[i + 1]], [derivs_lib[i], derivs_lib[i + 1]])
    splines.append(np.poly1d(coeffs))

for i in range(len(splines)):
    print(f'spline_{i} : {splines[i].coeffs}')

if verbose:
    plt.plot(x, y, 'o', color='black')
    for i in range(len(x) - 1):
        plt.plot(np.arange(x[i], x[i + 1], 0.01), splines[i](np.arange(x[i], x[i + 1], 0.01)), '--')
    plt.grid(True)
    plt.savefig('2_5_splines_test')
    plt.clf()

if verbose:
    plt.plot(x, y, 'o', color='black')
    for i in range(len(x) - 1):
        plt.plot(np.arange(x[i], x[i + 1], 0.01), splines[i](np.arange(x[i], x[i + 1], 0.01)), '--')
    plt.plot(x, y, 'o', np.arange(x[0], x[6], 0.01), poly(np.arange(x[0], x[6], 0.01)))
    plt.grid(True)
    plt.savefig('2_5_splines_and_alg_int_test')
    plt.clf()

spline_boards = []
for i in range(len(x) - 1):
    spline_boards.append([x[i], x[i + 1]])


def find_f(x, spl_b, spls):
    n_segment = len(spls)
    for i in range(len(spls)):
        if spl_b[i][0] <= x <= spl_b[i][1]:
            n_segment = i

    if n_segment != len(spls):
        return spls[n_segment](x)
    else:
        return 'out of boards'


print('\nCalculating the function value at a given point:')
while True:
    value = input('Enter value or \'exit\': ')

    if value != 'exit':
        print(f'spline : {find_f(float(value), spline_boards, splines)}')
        print(f'poly   : {poly(float(value))}')
        print(f'delta  : {find_f(float(value), spline_boards, splines) - poly(float(value))}')
    else:
        break
