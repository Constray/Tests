import sympy as sp
from sympy.abc import t, s
import numpy as np
import matplotlib.pyplot as plt


f = sp.laplace_transform(5*sp.sin(t) + 2*t ,t,s)
x = np.arange(0, 3, 0.1)
y = []
for i in x:
    y.append(f[0].subs(s, i))
plt.plot(x,y)
plt.show()
pass