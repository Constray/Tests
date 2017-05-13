import numpy as np
from sympy.abc import symbols
from sympy import I, im, re

def talbot_inverse(f_s, t_z, M = 64):
    k = np.arange(M)
    delta = np.zeros(M, dtype=complex)
    for i in k:
        if i != 0:
            delta[i] = 2*np.pi/5 * i * (np.tan(np.pi/M*i)**(-1)+1.j)
    delta[0] = 2*M/5
    gamma = np.zeros(M, dtype=complex)
    for i in k:
        if i != 0:
            gamma[i] = (1 + 1.j*np.pi/M*i*(1+np.tan(np.pi/M*i)**(-2))-1.j*np.tan(np.pi/M*i)**(-1))*np.exp(delta[i])
    gamma[0] = 0.5*np.exp(delta[0])
    delta_mesh, t_mesh = np.meshgrid(delta, t_z)
    gamma_mesh = np.meshgrid(gamma,t_z)[0]
    points = delta_mesh/t_mesh
    fun_res_ch = np.zeros(np.shape(points), dtype=complex)
    fun_res_zn = np.zeros(np.shape(points), dtype=complex)
    for i in range(len(f_s[0])):
        fun_res_ch = fun_res_ch + f_s[0][i]*points**(len(f_s[0]) - i)
    for i in range(len(f_s[1])):
        fun_res_zn = fun_res_zn + f_s[1][i]*points**(len(f_s[1]) - i)
    fun_res = fun_res_ch/fun_res_zn
    sum_ar = np.real(gamma_mesh*fun_res)
    sum_ar = np.sum(sum_ar, axis = 1)
    ilt = 0.4/t_z * sum_ar
    return ilt



# h = (s**2+0.25*s+12)/(s**2 + 4.75*s + 1)
s = symbols("s")
h = [1.0, 0.25, 12], [1.0, 4.75, 1]
hs_zn = []
for i in h[1]:
    hs_zn.append(i)
hs_zn.append(0)
asd = np.exp(12 + 5.j)
hs = h[0], hs_zn
t_z = np.linspace(1,5,num=50)
res = talbot_inverse(h, t_z)
fun = 2.0*s/(1.0*s**2 + 3.0*s + 2.0)
pass