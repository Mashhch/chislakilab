import numpy as np
import matplotlib.pyplot as plt

def u0(x):
    return x + np.cos(x)

def F2(x, y1, y2):
    return x * np.sin(x) - np.cos(x) * y2 - np.sin(x) * y1

def F(x, y1, y2):
    F_y1 = y2
    F_y2 = x * np.sin(x) - np.cos(x) * y2 - np.sin(x) * y1
    return [F_y1,F_y2]

def Euler(xn, y_y1, y_y2, h):
    for i in range(0,len(xn)-1):
        y_y1[i+1] = y_y1[i]+h*y_y2[i]
        y_y2[i+1] = y_y2[i] + h*F2(xn[i],y_y1[i], y_y2[i])
    return y_y1



def Gune(xn, y_y1, y_y2, h):
    for i in range(len(xn)-1):
        predictor_y1 = y_y1[i]+h*y_y2[i]
        predictor_y2 = y_y2[i] + h * F2(xn[i],y_y1[i], y_y2[i])
        y_y2[i + 1] = y_y2[i] + (h / 2) * (F2(xn[i], y_y1[i], y_y2[i]) + F2(xn[i + 1], predictor_y1, predictor_y2))
        y_y1[i+1]= y_y1[i]+(h/2)*(y_y2[i]+predictor_y2)

    return y_y1

def Runge(xn, y_y1, y_y2, h):
    y = [[0]*len(xn) for i in range(2)]
    y[0][0] = y_y1
    y[1][0] = y_y2
    for i in range(0, len(xn)-1):
        x = xn[i]
        k1 = F(x,y[0][i], y[1][i])
        k2 = F(x + h/2,y[0][i]+h/2*k1[0],y[1][i]+h/2*k1[1])
        k3 = F(x + h/2, y[0][i]+h/2*k2[0],y[1][i]+h/2*k2[1])
        k4 = F(x + h, y[0][i]+h*k3[0],y[1][i]+h*k3[1])
        y[0][i+1] = y[0][i] + (h / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        y[1][i+1] = y[1][i] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return y

def Adams(xn, y_y1, y_y2, h):
    y = [[0]*len(xn) for i in range(2)]
    z =  Runge([xn[0],xn[1], xn[2]], 1, 1, h)
    for j in range(3):
        y[0][j]= z[0][j]
        y[1][j] = z[1][j]
    for i in range(3, len(xn)):
        k3 = F(xn[i-3],y[0][i-3], y[1][i-3])
        k2 = F(xn[i-2],y[0][i-2], y[1][i-2])
        k1 = F(xn[i-1],y[0][i-1], y[1][i-1])
        y[0][i] = y[0][i-1] + h * ((23/12) * k1[0] - (16/12) * k2[0] + (5/12) * k3[0])
        y[1][i] = y[1][i-1] + h * ((23 / 12) * k1[1] - (16 / 12) * k2[1] + (5 / 12) * k3[1])
    return y

def adams_corr_Runge(a,b, y_y1, y_y2, h):
    xn_1 = np.arange(a, b + h, h)
    xn_2 = np.arange(a, b + h, h/2)
    corr = [[0] * len(xn_1) for i in range(2)]
    p = 3

    adams_i = Adams(xn_1, y_y1, y_y2, h)
    adams_i2 = Adams(xn_2, y_y1, y_y2, h/2)
    for i in range(len(xn_1)):
        corr[0][i] = adams_i2[0][2*i] + (adams_i2[0][2*i] - adams_i[0][i]) / (2**p - 1)
        corr[1][i] = adams_i2[1][i] + (adams_i2[1][2*i] - adams_i[1][i]) / (2**p - 1)
    y_true = [u0(x) for x in xn]
    return corr[0]


h=0.05
a = 0
b = 1
xn = np.arange(a, b + h, h)
length = len(xn)
y_y1= np.zeros(length)
y_y1[0] = 1
y_y2= np.zeros(length)
y_y2[0] = 1
# y_y2[0] = 0
xn_ = np.arange(a, b + h, h)
y_y1_= np.zeros(length)
y_y2_ = np.zeros(length)
y_y1_[0] = 1
y_y2_[0] = 1
y_Euler = Euler(xn, y_y1, y_y2,h)
y_Runge__ = Runge(xn, 1, 1, h)
y_Runge = y_Runge__[0]
y_Gune = Gune(xn_, y_y1_, y_y2_,h)
y_Adams___ = Adams(xn_, 1, 1,h)
y_Adams = y_Adams___[0]
y_cor =adams_corr_Runge(a,b, 1, 1,h)
y_true = [u0(x) for x in xn]
error_Euler = abs(y_true[-1] -y_Euler[-1])
print( y_true, "\n", y_Euler, "\n", error_Euler)


plt.figure(1)
plt.subplot(3,3,1)
plt.plot(xn, y_Euler, color='red', label='Euler')
plt.title('Метод Эйлера 1 порядка')
plt.grid(True)
plt.legend()
plt.subplot(3,3,2)
plt.plot(xn, y_Gune, color='red', label='Gune')
plt.title('Метод Гюна 2 порядка')
plt.grid(True)
plt.legend()

plt.subplot(3,3,3)
plt.plot(xn, y_Runge, color='red', label='Runge')
plt.title('Метод Рунге 4 порядка')
plt.grid(True)
plt.legend()

plt.subplot(3,3,4)
plt.plot(xn, y_Adams, color='red', label='Adams')
plt.title('Метод Рунге 4 порядка')
plt.grid(True)
plt.legend()

plt.subplot(3,3,5)
plt.plot(xn, y_cor, color='red', label='Adams')
plt.title('Метод Рунге 4 порядка')
plt.grid(True)
plt.legend()

plt.subplot(3,3,6)
plt.title('Функция')
plt.plot(xn, y_true, color='blue', label='true')
plt.grid(True)
plt.legend()

hmin = 0.01
hmax = 0.1
hstep = 0.001
hrange = np.arange(hmin, hmax, hstep)
error_Euler = np.zeros(len(hrange))
error_Gune = np.zeros(len(hrange))
error_Runge = np.zeros(len(hrange))
error_Adams = np.zeros(len(hrange))
error_cor_Adams = np.zeros(len(hrange))
for i in range(len(hrange)):
    h = hrange[i]
    xn__ = np.arange(a, b + h, h)
    y_y1 = np.zeros(len(xn__))
    y_y1[0] = 1
    y_y2 = np.zeros(len(xn__))
    y_y2[0] = 1
    # y_y2[0] = 0
    y_y1_ = np.zeros(len(xn__))
    y_y2_ = np.zeros(len(xn__))
    y_y1_[0] = 1
    y_y2_[0] = 1
    y_Euler_ = Euler(xn__, y_y1, y_y2, h)
    y_Gune_ = Gune(xn__, y_y1_, y_y2_, h)
    y_Adams_____ = Adams(xn__, 1, 1, h)
    y_Adams_= y_Adams_____[0]
    y_Runge___ = Runge(xn__, 1, 1, h)
    y_Runge_ = y_Runge___[0]
    y_cor_Adams_ = adams_corr_Runge(a,b, 1, 1, h)
    y_true = [u0(x) for x in xn__]
    # error_Euler[i] = abs(y_true[-1] - y_Euler[-1])
    error_Euler[i] = np.max([np.abs(y_true[j] - y_Euler_[j]) for j in range(len(xn__))])
    error_Gune[i] = np.max([np.abs(y_true[j] - y_Gune_[j]) for j in range(len(xn__))])
    error_Runge[i] = np.max([np.abs(y_true[j] - y_Runge_[j]) for j in range(len(xn__))])
    error_Adams[i] = np.max([np.abs(y_true[j] - y_Adams_[j]) for j in range(len(xn__))])
    error_cor_Adams[i] = np.max([np.abs(y_true[j] - y_cor_Adams_[j]) for j in range(len(xn__))])
    xx =0
error_Euler = [np.log10(elem) for elem in error_Euler]
error_Gune = [np.log10(elem) for elem in error_Gune]
error_Runge = [np.log10(elem) for elem in error_Runge]
error_Adams = [np.log10(elem) for elem in error_Adams]
error_cor_Adams = [np.log10(elem) for elem in error_cor_Adams]
hrange = [np.log10(elem) for elem in hrange]

plt.figure(2)
plt.plot(hrange, error_Euler, color='red', label='Euler')
plt.title('Метод Эйлера 1 порядка ошибка')
plt.plot(hrange, error_Gune, color='blue', label='Gune')
plt.title('Метод Гюна 2 порядка ошибка')
plt.plot(hrange, error_Runge, color='yellow', label='Runge')
plt.title('Метод Рунге 4 порядка ошибка')
plt.plot(hrange, error_Adams, color='green', label='Adams')
plt.title('Метод Рунге 4 порядка ошибка')
plt.plot(hrange, error_cor_Adams, color='pink', label='Adams_correction')
plt.title('Поправка Рунге для Адамса 3')
plt.grid(True)
plt.legend()

plt.show()