import numpy as np 
import matplotlib.pyplot as plt

R0 = 42.164e6
Rz = 6.378e6

G = 6.6743e-11
M = 5.972e24
t = 0

v0 = 0
a0 = 0
r0 = R0
delta_t = 1

r_list = []
v_list = []
t_list = []
a_list = []

t = 0


a = a0	
v = v0
r = r0

v_list.append(v)
a_list.append(a)
r_list.append(r)
t_list.append(t)




while r>Rz:
	t_list.append(t)
	
	
	a = G*M * (1/(r**2))

	v = v + a*t
	

	r = r - v*t - 0.5*a*(t**2)

	r_list.append(r)
	v_list.append(v)	
	a_list.append(a)

	



	t = t + delta_t
		
	









plt.plot(t_list, r_list)
plt.xlabel('t')
plt.ylabel('r')
plt.show()



plt.plot(t_list, v_list)
plt.xlabel('t')
plt.ylabel('v')
plt.show()



plt.plot(t_list, a_list)
plt.xlabel('t')
plt.ylabel('a')
plt.show()













