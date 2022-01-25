import statistics

import numpy as np
import matplotlib.pyplot as plt

# y_2[x]+np.cos[x]*y_1[x]+ np.sin(x)*y = x * np.sin(x)

def u(x):
    return x+np.cos(x)
def u1(x):
    return 1 - np.sin(x)

left_x = 0
right_x = 1
n= 21
h = (right_x-left_x)/(n-1)
xn = np.arange(left_x, right_x + h, h)
alpha = [3,2]
beta = [-1, 1]
gamma = [2, 3.239133626928383]






def p(x):
    return np.cos(x)


def q(x):
    return np.sin(x)

def func(x):
    return x * np.sin(x)


def difference_1(left_x, right_x, h, n, xn  ):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    f = np.zeros(n)

    a[0] = 0
    b[0] = alpha[0] - beta[0]/h
    c[0] = beta[0]/h

    a[-1] = -beta[1]/h
    b[-1] = alpha[1] + beta[1]/h
    c[-1] = 0

    f[0] = gamma[0]
    f[-1] = gamma[1]

    for i in range(1,n-1):
        a[i] = 1 / (h ** 2) - p(xn[i]) / (2 * h)
        b[i] = -2 / (h ** 2) + q(xn[i])
        c[i] = 1 / (h ** 2) + p(xn[i]) / (2 * h)
        f[i] = func(xn[i])

    return method_progonki(a,b,c,f, n)


def difference_2(left_x, right_x, h, n, xn  ):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    f = np.zeros(n)
    if beta[0] == 0:
        c[0] = 0
        b[0] = alpha[0]
        a[0] = 0
        f[0] = gamma[0]
    else:
        a[0] = 0
        b[0] = -2 + ((2 * alpha[0] * h) / beta[0]) - ((p(xn[0]) * alpha[0] * (h ** 2)) / (beta[0])) + q(xn[0]) * (h ** 2)
        c[0] = 2

    if beta[1] == 0:
        c[-1] = 0
        b[-1] = alpha[1]
        a[-1] = 0
        f[-1] = gamma[1]
    else:
        a[-1] = 2
        b[-1] = -2 - ((2 * alpha[1] * h) / beta[1]) - ((p(xn[-1]) * alpha[1] * (h ** 2)) / beta[1]) + (q(xn[-1]) * (h ** 2))
        c[-1] = 0

    f[0] = func(xn[0]) * (h ** 2) + ((gamma[0] * 2 * h) / beta[0]) - ((p(xn[0]) * gamma[0] * (h ** 2)) / beta[0])
    f[-1] = func(xn[-1]) * (h ** 2) - (((h ** 2) * p(xn[-1]) * gamma[1]) / beta[1]) - ((2 * h * gamma[1]) / beta[1])

    for i in range(1,n-1):
        a[i] = 1 / (h ** 2) - p(xn[i]) / (2 * h)
        b[i] = -2 / (h ** 2) + q(xn[i])
        c[i] = 1 / (h ** 2) + p(xn[i]) / (2 * h)
        f[i] = func(xn[i])

    return method_progonki(a, b, c, f, n)





def method_progonki(a,b,c,f, n):
    A = np.zeros(n)
    B = np.zeros(n)
    y = np.zeros(n)

    A[0] = -c[0] / b[0]
    B[0] = f[0] / b[0]

    for i in range(1, n - 1):
        A[i] = -c[i] / (b[i] + a[i] * A[i - 1])
    A[-1] = 0
    for i in range(1, n):
        B[i] = (f[i] - a[i] * B[i - 1]) / (b[i] + a[i] * A[i - 1])

    y[-1] = B[-1]
    for i in reversed(range(n-1)):
        y[i] = B[i] + A[i] * y[i + 1]
    return y



y_h1 = difference_1(left_x, right_x, h, n, xn)
y_h2  = difference_2(left_x, right_x, h, n, xn)
u_true = [u(x_) for x_ in xn]
plt.subplot(2, 2, 1)
plt.title("Разностный метод 1 порядка")
plt.grid()
plt.plot(xn, y_h1, color='blue')
plt.plot(xn, u(xn), color='green')

plt.subplot(2, 2, 2)
plt.title("Разностный метод 1 порядка")
plt.grid()
plt.plot(xn, y_h2, color='blue')
plt.plot(xn, u(xn), color='green')


n_ = range(20,400,20)
# hmin = 0.0005
# hmax = 0.005
# hstep = 0.0001
# hrange = np.arange(hmin, hmax+hstep, hstep)
hrange = []
error_1 = [0 for i in range(len(n_))]
error_2 = [0 for i in range(len(n_))]
for i in range(len(n_)):
    n = n_[i]
    h = (right_x-left_x)/(n-1)
    xn_ = np.arange(left_x, right_x + h, h)
    y_h1_arr = difference_1(left_x, right_x, h, n, xn_)
    y_h2_arr = difference_2(left_x, right_x, h, n, xn_)
    u_true_ = [u(x_) for x_ in xn_]
    error_1[i] = np.log10(max([abs(u_true_[i] - y_h1_arr[i]) for i in range(n)]))
    error_2[i] = np.log10(max([abs(u_true_[i] - y_h2_arr[i]) for i in range(n)]))
    hrange.append(np.log10(h))
plt.subplot(2, 2, 3)
plt.title("Ошибка 1 порядка")
plt.grid()
plt.plot(hrange, error_1, color='blue')

plt.subplot(2, 2, 4)
plt.title("Ошибка 2 порядка")
plt.grid()
plt.plot(hrange, error_2, color='blue')
plt.show()

