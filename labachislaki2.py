import math
import matplotlib.pyplot as plt
import numpy as np


def Function(x):
    return np.sinh(1 / (1 + x**2))


def Functiondiff1(x):
    return (-2) * x * np.cosh(1/(1 + x**2)) / (1 + x**2)**2


def Functiondiff2(x):
    result = 8*np.power(x, 2)*np.cosh(1/(1 + np.power(x, 2)))/ np.power(1 + np.power(x, 2), 3) + \
             4*np.power(x, 2)*np.sinh(1/(1 + np.power(x, 2)))/ np.power(1 + np.power(x, 2), 4) - \
             2*np.cosh(1/(1 + np.power(x, 2)))/ np.power(1 + np.power(x, 2), 2)
    return result

def centrdiff(left, right, step):
    return (right - left) / (2 * step)


def rightdiff(now, right, step):
    return (right - now) / step


def centrdiff22(left, now, right, step):
    return (left - 2 * now + right) / np.power(step, 2)


def centrdiff24(left2, left1, now, right1, right2, h):
    return (-left2 + 16 * left1 - 30 * now + 16 * right1 - right2) / (12 * np.power(h, 2))


a = -3
b = 3
x = np.linspace(a, b, 100)
y = Function(x)

number_of_knot = 51
h = (b - a) / (number_of_knot - 1)

x_knots = np.arange(a, b + h, h)
y_knots = []
for i in range(len(x_knots)):
    y_knots.append(Function(x_knots[i]))

print('Узлы сетки: ', x_knots)
print('Значения сетки: ', y_knots)
print("Шаг: ", h)

func_diff_knots = Functiondiff1(x)
diff_right_knots = [rightdiff(y_knots[i], y_knots[i+1], h) for i in range(len(y_knots) - 1)]
diff_central_knots = [centrdiff(y_knots[i], y_knots[i+2], h) for i in range(len(y_knots) - 2)]
err_right_knots = [abs(Functiondiff1(x_knots[i]) - diff_right_knots[i]) for i in range(len(x_knots)-1)]
err_central_knots = [abs(Functiondiff1(x_knots[i+1]) - diff_central_knots[i]) for i in range( len(x_knots)-2)]


diff2_func = Functiondiff2(x)
diff22_central_knots = [centrdiff22(y_knots[i], y_knots[i+1], y_knots[i+2], h) for i in range(len(y_knots) - 2)]
diff24_central_knots = [centrdiff24(y_knots[i], y_knots[i+1], y_knots[i+2], y_knots[i+3], y_knots[i+4], h) for i in range(len(y_knots) - 4)]
err_dif24 = [abs(Functiondiff2(x_knots[i+2]) - diff24_central_knots[i]) for i in range( len(x_knots)-4)]
err_dif22 = [abs(Functiondiff2(x_knots[i+1]) - diff22_central_knots[i]) for i in range( len(x_knots)-2)]


plt.figure(1)
plt.plot(x, y, color='red', label='func')
plt.plot(x_knots, y_knots, 'o', label='knots')
plt.title(r'Функция $y = sinh((1 + x^2)^{-1})}$', fontsize=16, y=1.05)
plt.grid(True)
plt.legend()

plt.figure(2)
plt.subplot(221)
plt.plot(x_knots[0:len(x_knots)-1], diff_right_knots, color='blue', label='right diff')
plt.plot(x, func_diff_knots, color='red', label='diff func')
# plt.plot(x_knots, y_dif_form, color='red', label='diff func')
plt.title(r'1 порядок right', fontsize=16, y=1.05)
plt.legend()

plt.subplot(222)
plt.plot(x_knots[0:len(x_knots)-1], err_right_knots, color='red', label='Погрешность')
plt.legend()

plt.subplot(223)
plt.plot(x_knots[1:len(x_knots)-1], diff_central_knots, color='blue', label='central diff')
plt.plot(x, func_diff_knots, color='red', label='diff func')
# plt.plot(x_knots, y_dif_form, color='red', label='diff func')
plt.title(r'1 порядок central', fontsize=16, y=1.05)
plt.legend()

plt.subplot(224)
plt.plot(x_knots[1:len(x_knots)-1], err_central_knots, color='red', label='Погрешность')
plt.legend()



plt.figure(3)
plt.subplot(221)
plt.plot(x_knots[1:len(x_knots)-1], diff22_central_knots, color='blue', label='centr 2 производная 2 порядка')
plt.plot(x, diff2_func, color='red', label='diff2 func')
plt.title(r'2 порядок точности', fontsize=16, y=1.05)
plt.legend()

plt.subplot(222)
plt.plot(x_knots[1:len(x_knots)-1], err_dif22, color='red', label='Погрешность 2 порядка')
plt.legend()

plt.subplot(223)
plt.plot(x_knots[2:len(x_knots)-2], diff24_central_knots, color='blue', label='centr 2 производная 4 порядка')
plt.plot(x, diff2_func, color='red', label='diff2 func')
plt.title(r'4 порядок точности', fontsize=16, y=1.05)
plt.legend()

plt.subplot(224)
plt.plot(x_knots[2:len(x_knots)-2], err_dif24, color='red', label='Погрешность 4 порядка')
plt.legend()


hmin = 0.01
hstep = 0.001
hmax = 0.1
hrange = np.arange(hmin, hmax, hstep)
a = -3
b = 3
max_abs_err_right = np.zeros(len(hrange))
max_abs_err_central = np.zeros(len(hrange))
max_abs_err_diff22 = np.zeros(len(hrange))
max_abs_err_diff24 = np.zeros(len(hrange))

for j in range(len(hrange)):
    h = hrange[j]
    x_knots = np.arange(a, b + h, h)
    y_knots = []
    for i in range(len(x_knots)):
        y_knots.append(Function(x_knots[i]))

    func_diff_knots = Functiondiff1(x)
    diff_right_knots = [rightdiff(y_knots[i], y_knots[i + 1], h) for i in range(len(y_knots) - 1)]
    diff_central_knots = [centrdiff(y_knots[i], y_knots[i + 2], h) for i in range(len(y_knots) - 2)]
    diff2_func = Functiondiff2(x)
    diff22_central_knots = [centrdiff22(y_knots[i], y_knots[i + 1], y_knots[i + 2], h) for i in range(len(y_knots) - 2)]
    diff24_central_knots = [centrdiff24(y_knots[i], y_knots[i + 1], y_knots[i + 2], y_knots[i + 3], y_knots[i + 4], h)
                            for i in range(len(y_knots) - 4)]
    err_right_knots = [Functiondiff1(x_knots[i]) - diff_right_knots[i] for i in range(len(x_knots) - 1)]
    err_central_knots = [Functiondiff1(x_knots[i + 1]) - diff_central_knots[i] for i in range(len(x_knots) - 2)]
    err_dif24 = [Functiondiff2(x_knots[i + 2]) - diff24_central_knots[i] for i in range(len(x_knots) - 4)]
    err_dif22 = [Functiondiff2(x_knots[i + 1]) - diff22_central_knots[i] for i in range(len(x_knots) - 2)]

    max_abs_err_right[j] = max([abs(elem) for elem in err_right_knots])
    max_abs_err_central[j] = max([abs(elem) for elem in err_central_knots])
    max_abs_err_diff22[j] = max([abs(elem) for elem in err_dif22])
    max_abs_err_diff24[j] = max([abs(elem) for elem in err_dif24])



# plt.figure(4)
# plt.grid()
# plt.plot(hrange, max_abs_err_right, color='red', label='right diff')
# plt.plot(hrange, max_abs_err_central, color='blue', label='central diff')
# plt.plot(hrange, max_abs_err_diff22, color='yellow', label='centr22')
# plt.plot(hrange, max_abs_err_diff24, color='k', label='centr24')
# plt.legend()

hrange = [np.log(elem) for elem in hrange]
max_abs_err_diff22 = [np.log(elem) for elem in max_abs_err_diff22]
max_abs_err_diff24 = [np.log(elem) for elem in max_abs_err_diff24]
max_abs_err_central = [np.log(elem) for elem in max_abs_err_central]
max_abs_err_right = [np.log(elem) for elem in max_abs_err_right]

plt.figure(5)
plt.grid()
plt.plot(hrange, max_abs_err_right, color='red', label='right diff')
plt.plot(hrange, max_abs_err_central, color='blue', label='central diff')
plt.plot(hrange, max_abs_err_diff22, color='yellow', label='centr22')
plt.plot(hrange, max_abs_err_diff24, color='k', label='centr24')
plt.legend()

plt.show()








# y_diff_right = [lagranz(x_knots[0:len(x_knots)-1], diff_right_knots, i) for i in x]
# y_diff_err = [lagranz(x_knots[0:len(x_knots)-1], err_y_knots, i) for i in x]