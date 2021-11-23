import math
import matplotlib.pyplot as plt
import numpy as np


def Function(x):
    result = np.power((x / 10), np.sin(x))
    return result


def lagranz(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i != j:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z

a = 0
b = 10
x = np.linspace(0, 10, 100)
y = np.power((x / 10), np.sin(x))

# number_of_knot = int(input('Количество узлов: '))
number_of_knot = 10
step = 10 / (number_of_knot-1)

x_knots = [x for x in np.arange(0, 10 + step, step) if x <= 10]
y_knots = []
for i in range(len(x_knots)):
    y_knots.append(Function(x_knots[i]))

print('Узлы интерполяции: ', x_knots)

ynew = [lagranz(x_knots, y_knots, i) for i in x]

plt.figure(1)
plt.subplot(211)
plt.plot(x, ynew, color='blue', label='lagranz')
plt.plot(x, y, color='red', label='func')
plt.plot(x_knots, y_knots, 'o', label='knots')
plt.title(r'Функция $y = (x / 10)^{sin(x)}$', fontsize=16, y=1.05)
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.title("Погрешность пункт а")
y_diffrence = [y[i]-ynew[i] for i in range(len(x))]
plt.plot(x,y_diffrence, color = 'purple', label = 'diffrence')
plt.legend()

max_diff = max(y_diffrence) if abs(max(y_diffrence)> abs(min(y_diffrence))) else min(y_diffrence)
print("Максимальная погрешность: ", abs(max_diff))

x_cheb = [(a+b)/2 + ((b-a)*math.cos((2*i+1)*np.pi/(2*number_of_knot)))/2 for i in range(number_of_knot)]
y_cheb = []
for i in range(len(x_cheb)):
    y_cheb.append(Function(x_cheb[i]))
y_cheb_knots = [lagranz(x_cheb, y_cheb, i) for i in x]

plt.figure(2)
plt.subplot(211)
plt.title("Узлы Чебышева")
plt.plot(x, y_cheb_knots, color='blue', label='Chebyshev')
plt.plot(x, y, color='red', label='func')
plt.plot(x_cheb, y_cheb, 'o', label='knots')
plt.legend()


plt.subplot(212)
plt.title("Погрешность пункт б")
y_diffrence_cheb = [y[i]-y_cheb_knots[i] for i in range(len(x))]
plt.plot(x,y_diffrence_cheb, color = 'purple', label = 'diffrence')
plt.legend()

max_diff_cheb = max(y_diffrence_cheb) if abs(max(y_diffrence_cheb)> abs(min(y_diffrence_cheb))) else min(y_diffrence_cheb)
print("Минимальная максимальная погрешность: ", abs(max_diff_cheb))

plt.show()
