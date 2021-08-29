import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.array([-1.0077903311937846,
	-0.9082374375674063,
	-0.6878422667350763,
	-0.5806037506675183,
	-0.47237259562526823,
	-0.37928232620964786,
	-0.27181345088635955,
	-0.16289959477712723,
	-0.04985602546516854,
	0.046727327937342356,
	0.1621791985592076,
	0.2711851983707321,
	0.3740720186800417,
	0.47607928546747225,
	0.5778980765002146,
	0.6725757306053213,
	0.7718354397244069,
	0.9011591258913332,
	1.0083557584578497])

Y = np.array([0.0786572149565985,
	0.1889364731997949,
	0.06242735830288071,
	0.07446362941478313,
	-0.10330041988209748,
	-0.012591227500706292,
	0.07143231102687908,
	0.6070123451619325,
	0.9331277551490533,
	1.1154256934337146,
	2.194135262766615,
	2.2585102038679414,
	1.9106415505272092,
	1.287912421599322,
	0.6062846193313307,
	0.19305152717715712,
	0.21171062689130205,
	0.12536255405589403,
	0.12431023109221773])

uY = [0.08050928,
	0.19537596,
	0.06266496,
	0.07172729,
	-0.11159463,
	-0.01384578,
	0.07550053,
	0.57858563,
	0.92955094,
	1.19755522,
	2.15650817,
	2.38802197,
	1.99521974,
	1.40446746,
	0.5542059,
	0.19005834,
	0.21625005,
	0.12690156,
	0.11575943]


a1 = 1
a2 = 1
a3 = 1
a = np.array([1,1,1])

def f(x,a1,a2,a3):
	return a1*np.exp((-(x-a2)**2)/(2*(a3**2)))


popt,pcov = curve_fit(f,X,Y,sigma=uY)
perr = (np.diag(pcov))**0.5


print('perr',perr)
print(popt,pcov)


fig, ax = plt.subplots()
ax.plot(X,Y,'.')
ax.plot(np.linspace(-1,1,100),f(np.linspace(-1,1,100),popt[0],popt[1],popt[2]),'.')
plt.show()