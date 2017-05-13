import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft, fftfreq
import copy
import numpy as np
from sympy import Heaviside, sin, cos, symbols

t = symbols("t", positive=True)
N=200
Tau = 20
A = 10
signal = A*(1 - 2*Heaviside(t-Tau/2) + Heaviside(t-Tau))

# Задаём шаг
T = Tau/N
x = np.linspace(0.0, N * T, N)
y = []
f2 = copy.deepcopy(signal)
# Заполняем массив значений сигнала
for i in x:
    if int(i) == i:
        f2 = f2.subs(Heaviside(t - int(i)), 1)
    else:
        f2 = f2.subs(Heaviside(t - i), 1)
    y.append(f2.subs(t, i))
# Заполняем значения амплитудного спектра
yf = fftshift(fft(y))
# Заполняем частоты
xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), N)
yf2 = copy.deepcopy(2.0 / N * yf)
# Фильтруем помехи
threshold = np.max(np.abs(yf2)) / 10000
for i in range(len(yf2)):
    if abs(yf2[i]) < threshold:
        yf2[i] = 0
xp = []
yp = []
# Заполняем новые массивы без помех
for i in range(len(yf2)):
    if yf2[i] != 0:
        xp.append(xf[i])
        yp.append(yf2[i])
yp = np.angle(yp)
yf = np.abs(yf)

plt.plot(xf,yf)
plt.grid()
plt.show()
