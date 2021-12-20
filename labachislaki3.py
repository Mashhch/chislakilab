import numpy as np
import matplotlib.pyplot as plt


def Function(x):
    return  x/np.sqrt(5- x**2)


def Integral(x):
    return (-1)*np.sqrt(5-x**2)


def Rectangle(a, b, f):
    return (b - a) * f(a)


def Trapezoid(a, b, f):
    return (b - a) * (f(a) + f(b)) / 2


def Simpsons(a, b, f):
    return (b - a) * (f(a) + 4 * f((a + b) / 2) + f(b)) / 6


def Rule38(a, b, f):
    return (b - a) * (f(a) + 3 * f((2 * a + b) / 3) + 3 * f((a + 2 * b) / 3) + f(b)) / 8

def Gaussian2(a, b, f):
    return (b - a)/2 * (f((a + b) / 2 - (b - a) / (2 * np.sqrt(3))) + f((a + b) / 2 + (b - a) / (2 * np.sqrt(3))))

a = -2
b = 1
integral_true_value = -1
h = 0.1
integral_rectangle_value = 0
integral_trapezoid_value = 0
integral_simpson_value = 0
integral_38_value = 0
integral_gaussian_value = 0
x_knots = np.arange(a, b + h, h)
for i in range(len(x_knots)-1):
    integral_rectangle_value += Rectangle(x_knots[i], x_knots[i + 1], Function)
    integral_trapezoid_value += Trapezoid(x_knots[i], x_knots[i + 1], Function)
    integral_simpson_value += Simpsons(x_knots[i], x_knots[i + 1], Function)
    integral_38_value += Rule38(x_knots[i], x_knots[i + 1], Function)
    integral_gaussian_value += Gaussian2(x_knots[i], x_knots[i + 1], Function)

print("True value", integral_true_value, "\nRectangle", integral_rectangle_value, "\nTrapezoid", integral_trapezoid_value, "\nSimpson", integral_simpson_value)
print("Rule 3/8", integral_38_value, "\nGaussian", integral_gaussian_value)


hmin = 0.01
hmax = 0.1
hstep = 0.001
hrange = np.arange(hmin, hmax, hstep)
max_abs_err_rectangle = np.zeros(len(hrange))
max_abs_err_trapezoid = np.zeros(len(hrange))
max_abs_err_simpson = np.zeros(len(hrange))
max_abs_err_38 = np.zeros(len(hrange))
max_abs_err_gaussian = np.zeros(len(hrange))

Glob_err_rectangle = np.zeros(len(hrange))
Glob_err_trapezoid = np.zeros(len(hrange))
Glob_err_simpson = np.zeros(len(hrange))
Glob_err_38 = np.zeros(len(hrange))
Glob_err_gaussian = np.zeros(len(hrange))

for j in range(len(hrange)):
    h = hrange[j]
    x_knots = np.arange(a, b + h, h)
    true_integral = []
    integral_rectangle = np.zeros(len(x_knots)-1)
    integral_trapezoid = np.zeros(len(x_knots)-1)
    integral_simpson = np.zeros(len(x_knots)-1)
    integral_38 = np.zeros(len(x_knots)-1)
    integral_gaussian = np.zeros(len(x_knots) - 1)
    for i in range(len(x_knots)):
        true_integral.append(Integral(x_knots[i]))
    for i in range(len(x_knots) - 1):
        integral_rectangle[i] = Rectangle(x_knots[i], x_knots[i + 1], Function)
        integral_trapezoid[i] = Trapezoid(x_knots[i], x_knots[i + 1], Function)
        integral_simpson[i] = Simpsons(x_knots[i], x_knots[i + 1], Function)
        integral_38[i] = Rule38(x_knots[i], x_knots[i + 1], Function)
        integral_gaussian[i] = Gaussian2(x_knots[i], x_knots[i + 1], Function)

    err_rectangle = [true_integral[i+1]- true_integral[i] - integral_rectangle[i] for i in range(len(x_knots) - 1)]
    err_trapezoid = [true_integral[i + 1] - true_integral[i] - integral_trapezoid[i] for i in range(len(x_knots) - 1)]
    err_simson = [true_integral[i + 1] - true_integral[i] - integral_simpson[i] for i in range(len(x_knots) - 1)]
    err_38 = [true_integral[i + 1] - true_integral[i] - integral_38[i] for i in range(len(x_knots) - 1)]
    err_gaussian = [true_integral[i + 1] - true_integral[i] - integral_gaussian[i] for i in range(len(x_knots) - 1)]
    for elem1 in err_rectangle:
        Glob_err_rectangle[j] += elem1
    for elem2 in err_trapezoid:
        Glob_err_trapezoid[j] += elem2
    for elem3 in err_simson:
        Glob_err_simpson[j] += elem3
    for elem4 in err_gaussian:
        Glob_err_gaussian[j] += elem4
    for elem5 in err_38:
        Glob_err_38[j] += elem5

    max_abs_err_rectangle[j] = max(abs(elem) for elem in err_rectangle)
    max_abs_err_trapezoid[j] = max(abs(elem) for elem in err_trapezoid)
    max_abs_err_simpson[j] = max(abs(elem) for elem in err_simson)
    max_abs_err_38[j] = max(abs(elem) for elem in err_38)
    max_abs_err_gaussian[j] = max(abs(elem) for elem in err_gaussian)


print(Glob_err_gaussian)
hrange = [np.log(elem) for elem in hrange]
max_abs_err_rectangle = [np.log(elem) for elem in max_abs_err_rectangle]
max_abs_err_trapezoid = [np.log(elem) for elem in max_abs_err_trapezoid]
max_abs_err_simpson = [np.log(elem) for elem in max_abs_err_simpson]
max_abs_err_38 = [np.log(elem) for elem in max_abs_err_38]
max_abs_err_gaussian = [np.log(elem) for elem in max_abs_err_gaussian]

Glob_err_rectangle = [np.log(abs(elem)) for elem in Glob_err_rectangle]
Glob_err_trapezoid = [np.log(abs(elem)) for elem in Glob_err_trapezoid]
Glob_err_simpson = [np.log(abs(elem)) for elem in Glob_err_simpson]
Glob_err_38 = [np.log(abs(elem)) for elem in Glob_err_38]
Glob_err_gaussian = [np.log(abs(elem)) for elem in Glob_err_gaussian]


plt.figure(1)
plt.grid()
plt.plot(hrange, max_abs_err_rectangle, color='green', label='rectangle')
plt.plot(hrange, max_abs_err_trapezoid, color='red', label='trapezoid')
# plt.plot(hrange, max_abs_err_gaussian, color='blue', label='gaussian')
plt.plot(hrange, max_abs_err_simpson, color='yellow', label='simpson')
plt.plot(hrange, max_abs_err_38, color='black', label='3/8')
plt.legend()

plt.figure(2)
plt.grid()
plt.plot(hrange, Glob_err_rectangle, color='green', label='rectangle')
plt.plot(hrange, Glob_err_trapezoid, color='red', label='trapezoid')
plt.plot(hrange, Glob_err_gaussian, color='blue', label='gaussian')
plt.plot(hrange, Glob_err_simpson, color='yellow', label='simpson')
plt.plot(hrange, Glob_err_38, color='black', label='3/8')
plt.legend()


plt.show()


plt.figure(2)
