import numpy as np 
import matplotlib.pyplot as plt 


x1 = 0
x2 = 100

e = 0.5

def f(x):
	return x**3 - 1500000*x**2 + 750000000002*x - 125000000000999990 

k = 2

x_list = np.ones(100)


while abs(f(x_list[k])) > e:
	f1 = f(x1)
	f2 = f(x2)
	
	try:
		x = (f2*x1 - f1*x2)/(f2-f1)
	except:
		break
	x_list[k] = x 
	x1 = x2
	x2 = x 
	k += 1
	


x = np.arange(0, 1E6, 10)

fig, ax = plt.subplots()
ax.plot(x, f(x))
ax.plot(x_list, f(x_list), '.')

plt.show()

print('Miejsce zerowe funkcji: ',x_list[k-1])



