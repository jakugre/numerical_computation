import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


n = 3

G = const.G
mS = 1.989e30
mE = 5.972e24
mM = 6.4171e23


v_E = 29.78e3
v_M = 24e3

r_E_0 = 149.6e9 #m
r_M_0 = 228e9 #m


M = np.ones((n,1))
M[0,0] = mS
M[1,0] = mE
M[2,0] = mM

R = np.zeros((n, 3)) #m
V = np.zeros((n, 3)) #m/s
A = np.zeros((n, 3)) #m/s^2
F = np.zeros((n, 3))

R[1,0] = r_E_0
R[2,0] = -r_M_0
V[1,1] = v_E
V[2,1] = -v_M



lista = []



MM = np.zeros(n*(n-1))
r = np.zeros(n*(n-1))
t_final = 10. * r_E_0/v_E
dt = 0.001 * t_final
t = 0 # s
maxiterator = 10000
iterator = 0


R_Values = np.zeros((maxiterator*n,3))

while iterator < maxiterator:
	F = F*0.0
	k = 0
	for i in range(n):

	    for j in range(n):

	        if j != i:            

	            lista.append([i,j])
	            MM[k] = M[i]*M[j]

	            odl = np.sum((R[i,:] - R[j,:])**2)**0.5

	            er = -(R[i,:] - R[j,:])/odl

	            F[i,:] = F[i,:] + (M[i,:]*M[j,:]*er*G)/(odl**2)


	A = F/M 
	V = V + A*dt
	R = R + V*dt 
	
	R_Values[iterator*n:(iterator+1)*n] = R
	iterator += 1





def func(num, dataSet, line):
    line.set_data(dataSet[0:2, :num])    
    line.set_3d_properties(dataSet[2, :num])    
    return line

data_sun = np.array([R_Values[0::3,0], R_Values[0::3,1], R_Values[0::3,2]])
data_earth = np.array([R_Values[1::3,0], R_Values[1::3,1], R_Values[1::3,2]])
data_mars = np.array([R_Values[2::3,0], R_Values[2::3,1], R_Values[2::3,2]])

length = len(R_Values)


fig = plt.figure()
ax = Axes3D(fig)

Sun = plt.plot(data_sun[0], data_sun[1], data_sun[2], lw=10, c='yellow')[0]
Earth = plt.plot(data_earth[0], data_earth[1], data_earth[2], lw=3, c='blue')[0]
Mars = plt.plot(data_mars[0], data_mars[1], data_mars[2], lw=2, c='red')[0]
animation_Earth = animation.FuncAnimation(fig, func, length, fargs=(data_earth,Earth), interval=1, blit=False)
animation_Mars = animation.FuncAnimation(fig, func, length, fargs=(data_mars,Mars), interval=1, blit=False)

plt.show()