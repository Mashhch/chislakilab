import numpy as np
import matplotlib.pyplot as plt


def Function(x):
    return x ** 5 - 7 * x ** 3 - 3 * x - 2


def Functiondiff1(x):
    return 5 * x ** 4 - 21 * x ** 2 - 3


def Dichotomy(left, right, epsilon):
    root = (left + right) / 2
    Xn = [left, right]
    n = 1
    while right - left > epsilon:  # abs(Function(right)-Function(left)) > epsilon:
        root = (left + right) / 2
        n += 1
        Xn.append(root)
        if np.sign(Function(left)) != np.sign(Function(root)):
            right = root
        else:
            left = root
    # print("Elems Dichotomy", Xn)
    return [root, n, Rate(Xn)]


def Newton(x, eps):
    Xn = [x]
    n = 1
    while abs(Function(x)) > eps:
        n += 1
        x = x - (Function(x) / Functiondiff1(x))
        Xn.append(x)
    # print("Elems Newton", Xn)
    return [x, n, Rate(Xn)]


# def Newton(x, eps):
#     Xn = [x]
#     n =1
#     root = x - (Function(x) / Functiondiff1(x))
#     while abs(root - x) > eps: #abs(Function(root)) > eps:
#         x = root
#         n += 1
#         root = x - (Function(x) / Functiondiff1(x))
#         Xn.append(root)
#         print(x, root)
#     print("Elems ", Xn)
#     return [root, n, Rate(Xn)]

def Chord(left, right, epsilon):
    Xn = [right, left]
    n = 0
    while abs(Function(right)) > epsilon:
        n += 1
        left, right = right, right - Function(right) * (right - left) / (Function(right) - Function(left))
        Xn.append(right)
    # print("Elems chord", Xn)
    return [right, n, Rate(Xn)]


def Rate(Xn):
    R = []
    if len(Xn) > 3:
        for k in range(3, len(Xn)):
            R.append(np.log(abs(Xn[k] - Xn[k - 1]) / abs(Xn[k - 1] - Xn[k - 2])) / np.log(
                abs(Xn[k - 1] - Xn[k - 2]) / abs(Xn[k - 2] - Xn[k - 3])))
    else:
        return None
    return R


a = 0
b = 10
epsilon = [10 ** (-9), 10 ** (-7), 10 ** (-3)]
print("Значение корня:", 2.738165672423897)
for e in epsilon:
    dichotomy = Dichotomy(a, b, e)
    newton = Newton((a + b) / 2, e)
    chord = Chord((a+b)/2, b, e)
    print("Дихтомия " + str(e) + " : " + str(dichotomy[0]))
    print("Количество итераций: " + str(dichotomy[1]))
    print("Скорость сходимости: " + str(dichotomy[2]))

    print("Метод Ньютон: " + str(e) + " : " + str(newton[0]))
    print("Количество итераций: " + str(newton[1]))
    print("Скорость сходимости: " + str(newton[2]))

    print("Метод секущих: " + str(e) + " : " + str(chord[0]))
    print("Количество итераций: " + str(chord[1]))
    print("Скорость сходимости: " + str(chord[2]))

print("lol")
