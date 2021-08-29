import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import differential_evolution as de 

def f_2(X):
	return - 20*np.exp(-0.2*((X[0]**2 + X[1]**2)/2)**0.5) - np.exp(0.5*(np.cos(2*np.pi*X[0])+np.cos(2*np.pi*X[1]))) + np.exp(1) + 20

def f_4(X):
	return - 20*np.exp(-0.2*((X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2)/2)**0.5) - np.exp(0.5*(np.cos(2*np.pi*X[0])+np.cos(2*np.pi*X[1]) + np.cos(2*np.pi*X[2]) + np.cos(2*np.pi*X[3]))) + np.exp(1) + 20

def f_8(X):
	return - 20*np.exp(-0.2*((X[0]**2 + X[1]**2 + X[2]**2 + X[3]**2 + X[4]**2 + X[5]**2 + X[6]**2 + X[7]**2)/2)**0.5) - np.exp(0.5*(np.cos(2*np.pi*X[0])+np.cos(2*np.pi*X[1]) + np.cos(2*np.pi*X[2]) + np.cos(2*np.pi*X[3]) + np.cos(2*np.pi*X[4])+np.cos(2*np.pi*X[5]) + np.cos(2*np.pi*X[6]) + np.cos(2*np.pi*X[7]))) + np.exp(1) + 20


bounds_2 = [(-32,32),(-32,32)]
bounds_4 = [(-32,32),(-32,32),(-32,32),(-32,32)]
bounds_8 = [(-32,32),(-32,32),(-32,32),(-32,32),(-32,32),(-32,32),(-32,32),(-32,32)]

x_2 = de(f_2,bounds_2)




print(x_2)
print('Minimum 2D: ',x_2.x[0],x_2.x[1])

x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
X = [x,y]


fig = plt.figure()

ax = fig.gca(projection='3d')

x,y = np.meshgrid(x,y)

surf = ax.plot_surface(x,y, f_2([x,y]), cmap=plt.cm.get_cmap('Reds'))
ax.scatter(x_2.x[0],x_2.x[1],x_2.fun,c='g')

ax.set(xlabel='x', ylabel='y', zlabel='z')

plt.show()

x_4 = de(f_4,bounds_4)
print(x_4)

x_8 = de(f_8,bounds_8)
print('Minimum 8D: ',x_8.x[0],x_8.x[1],x_8.x[2],x_8.x[3],x_8.x[4],x_8.x[5],x_8.x[6],x_8.x[7])