import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize

def sign_b(a,b):
	if (b >= 0.0):
		return abs(a)
	else:
		return -abs(a)

def f(r):


	r_0 = 0.33*10**-10 # m
	e = 1.60217662 * 10**-19 # C
	V_0 = 1.09*10**3 # eV

	e_0 = 55.26349406 * 10**-9 * 10**15 # e^2⋅eV^−1⋅m^−1
	v = -1/(4*np.pi*e_0*r) + V_0*np.exp(-r/r_0)
	return v


A = 10**-10 # m

ax = 1.5*A
bx = 2.5*A
cx = 3*A
tol = 1.0e-6 #precyzja ograniczenia minimum
i = 0 #krok iteracji 
iter_max = 100 #maksymalna liczba iteracji
zeps = 1.0e-12 #zabezpieczenie przed minimum w zerze
c_gold = 0.3819660 #współczynnik złotego podziału

if ax < cx: # a i b ustawiamy w kolejności rosnącej
        a = ax
else:
    a = cx

if ax > cx:
    b = ax
else:
    b = cx


v = b
w = v
x = v
e = 0.0 #przesunięcie minimum w przedostatnim kroku
fx = f(x) #obliczenie wartości w punkcie środkowym
fv = fx
fw = fx

while i < iter_max:
	xm = 0.5*(a+b)
	tol1 = tol*abs(x) + zeps
	tol2 = 2.0*tol1
	

	if (abs(x - xm) <= (tol2 - 0.5*(b-a))): #warunek zakończenia
		break
	if (abs(e) > tol1): #znajdujemy probną parabolę
		r = (x - w)*(fx - fv)
		q = (x - v)*(fx - fw)
		p = (x - v)*q - (x - w)*r
		q = 2*(q - r)
		if (q > 0.0):
			p = -p
		q = abs(q)
		etemp = e
		if ((abs(p) >= abs(0.5*q*etemp)) or (p <= q*(a -x)) or (p >= q*(b -x))): #warunek decyduje czy robimy złoty podział czy interpolacje parabol
			if (x >= xm):
				e = a - x
			else:
				e = b - x
			d = c_gold*e
		else: #interpolacja paraboliczna
			d = p/q
			u = x + d
			if ((u - a < tol2) or (b - u < tol2)):
				d = sign_b(tol1, xm - x)
				
	else:
		if (x >= xm):
			e = a - x
		else:
			e = b - x
		d = c_gold*e
	if (abs(d) >= tol1):
		u = x + d
	else:
		u = x + sign_b(tol1, d)
    
    	
	fu = f(u) #tylko raz w pętli obliczamy wartość funkcji i decydujemy co dalej przygotowując następną pętlę
        

	if (fu <= fx):
		if (u >= x):
			a = x
		else:
			b = x
		fv = fw 
		w = x
		fw = fx
		x = u
		fx = fu
	else:
		if (u < x):
			a = u
		else:
			b = u
		if ((fu <= fw) or (w == x)):
			v = w
			fv = fw
			w = u
			fw = fu

		else:
			if ((fu <= fv) or (v == x) or (v == w)):
				v = u
				fv = fu



	i += 1

xmin = x
fmin = fx
X = np.arange(1.*A, 5.*A, 0.001*A)

plt.plot(X,f(X))
plt.plot(xmin,fmin,'ro')
plt.show()


brent = optimize.brent(f,brack=(a,b))
print(brent)
print(xmin)