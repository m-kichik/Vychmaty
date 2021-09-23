# нелинейные уравнения и системы уравнений - вариант 2 задание 1

gamma_0 = 5 / 3
rho_0 = 1.694e-4
P_0 = 1.013e6
U_0 = 0.
gamma_3 = 7 / 5
C_3 = 3.6537e4
P_3 = 1.6768e6
U_3 = 0.

rho_3 = gamma_3 * (P_3 / C_3 ** 2)

alpha_0 = (gamma_0 + 1) / (gamma_0 - 1)
n = 2 * gamma_3 / (gamma_3 - 1)
mu = (U_3 - U_0) * ((gamma_0 - 1) * rho_0 / (2 * P_0)) ** 0.5
nu = (2 / (gamma_3 - 1)) * ((gamma_3 * (gamma_0 - 1) / 2) * (P_3 / P_0) * (rho_0 / rho_3)) ** 0.5
X = P_3 / P_0

print(f'alpha_0 = {alpha_0}')
print(f'n = {n}')
print(f'mu = {mu}')
print(f'nu = {nu}')
print(f'X = {X}\n')

a_0 = X ** 2
a_1 = - alpha_0 * (nu ** 2) * X
a_2 = 2 * alpha_0 * nu * (mu + nu) * X
a_3 = -(2 + (mu + nu) ** 2 * alpha_0) * X
a_4 = - nu ** 2
a_5 = 2 * nu * (mu + nu)
a_6 = - (mu + nu) ** 2 + 1

print(f'a_0 = {a_0}')
print(f'a_1 = {a_1}')
print(f'a_2 = {a_2}')
print(f'a_3 = {a_3}')
print(f'a_4 = {a_4}')
print(f'a_5 = {a_5}')
print(f'a_6 = {a_6}\n')

coeffs = [a_0, 0., 0., 0., 0., a_1, a_2, a_3, 0., 0., 0., 0., a_4, a_5, a_6]
delta = 1 + max(coeffs) / a_0

print(f'delta = {delta}\n')


def f(coeffs, x):
    res = 0
    n = len(coeffs)
    for i in range(len(coeffs)):
        res += coeffs[i] * (x ** (n - 1))
        n -= 1

    return res


num_pieces = 100
small_delta = delta / num_pieces
x = 0.
x_list = []
prev_x = 0.
prev_f = f(coeffs, prev_x)
f_list = []
for _ in range(num_pieces):
    x = x + small_delta
    func = f(coeffs, x)

    if (prev_f >= 0 and func <= 0) or (prev_f <= 0 and func >= 0):
        x_list.append((prev_x, x))
        f_list.append((prev_f, func))

    prev_x = x
    prev_f = func

print(f'области локализации: {x_list}\n')

epsilon = 1e-8

my_roots = []
for interval in x_list:
    x_left = interval[0]
    x_right = interval[1]
    while x_right - x_left > epsilon:
        x_middle = (x_left + x_right) / 2
        if f(coeffs, x_middle) * f(coeffs, x_left) < 0.:
            x_right = x_middle
        if f(coeffs, x_middle) * f(coeffs, x_right) < 0.:
            x_left = x_middle
    my_roots.append((x_left + x_right) / 2)

print(f'roots : {my_roots}\n')

# небольшой чит

# import numpy as np

# roots = np.roots(coeffs)
# Z_list = [x.real for x in roots if x.imag == 0. and x.real > 0]

Z_list = my_roots
print(f'Z_list = {Z_list}\n')

P_1_list = P_2_list = [x ** n * P_3 for x in Z_list]
print(f'P_1_list = {P_1_list}\n')

C_2_list = [C_3 * (x / P_3) ** ((gamma_3 - 1) / (2 * gamma_3)) for x in P_2_list]
print(f'C_2_list = {C_2_list}\n')

U_1_list = U_2_list = [U_3 + 2 * ((C_3 - x) / (gamma_3 - 1)) for x in C_2_list]
print(f'U_1_list = {U_1_list}\n')

D_0_list = []
for i in range(len(Z_list)):
    D_0_list.append(U_0 - (1 / rho_0) * ((P_1_list[i] - P_0) / (U_0 - U_1_list[i])))

print(f'D_0_list = {D_0_list}')
