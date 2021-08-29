import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation 
from numpy import sin, cos



L1 = 1 #długość pierwszego wahadła
L2 = 1 #długość "drugiego" wahadła
M1 = 1 #masa pierwszego wahadła
M2 = 1 #masa "drugiego" wahadła

g = 9.81 #przyspieszenie grawitacyjne

n = 4  #liczba równań różniczkowych

tmin = 0 #czas początkowy
tmax = 10 #czas trwania symulacji
dt = 0.005 #orzyrost czasu

n_of_steps = int(tmax/dt) #ilość kroków obliczeń

dydx = np.zeros(n) #wektor przechowujący wartości pochodnych 

#wartości kątów oraz prędkości kątowych oraz wektory przechowujące te wartości 
th1 = 90.0
th2 = -20.0
w1 = 0.0
w2 = 0.0

th1_list = np.zeros(n_of_steps)
w1_list = np.zeros(n_of_steps)
th2_list = np.zeros(n_of_steps)
w2_list = np.zeros(n_of_steps)

th1_list[0] = np.radians(th1)
w1_list[0] = np.radians(w1)
th2_list[0] = np.radians(th2)
w2_list[0] = np.radians(w2)

#wektor przechowujący kolejne wartości czasu 
t = np.arange(tmin,tmax,dt)

#Wektor pomocniczy
y = np.zeros(n)
#Wektor używany do metody RK jako wektor do którego przekazywane są wartości kolejnych kroków
y_output = np.zeros(n)


def derivatives(y_input, dydt):
	#dydx jest postaci th1',w1',th2',w2', gdzie th1' = w1 a th2' = w2
	dydt[0] = y_input[1]

	delta = y_input[2] - y_input[0]

	den1 = (M1+M2)*L1 - M2*L1*cos(delta)*cos(delta)

	dydt[1] = (M2*L1*y_input[1]*y_input[1]*sin(delta)*cos(delta) + M2*g*sin(y_input[2])*cos(delta) + M2*L2*y_input[3]*y_input[3]*sin(delta) - (M1+M2)*g*sin(y_input[0]))/den1

	dydt[2] = y_input[3]

	den2 = (L2/L1)*den1

	dydt[3] = (-M2*L2*y_input[3]*y_input[3]*sin(delta)*cos(delta) + (M1+M2)*g*sin(y_input[0])*cos(delta)  - (M1+M2)*L1*y_input[1]*y_input[1]*sin(delta) - (M1+M2)*g*sin(y_input[2]))/den2

	return dydx


def RK(x_input, y_input, y_output, h):
	

	xh = x_input + h/2
	dydt = np.zeros(n)
	dydt1 = np.zeros(n)
	dydt2 = np.zeros(n)
	dydt3 = np.zeros(n)
	dydt4 = np.zeros(n)

	dydt = derivatives(y_input,dydx)
	s1 = np.zeros(n)
	s2 = np.zeros(n)
	s3 = np.zeros(n)
	s4 = np.zeros(n)

	yt = np.zeros(n)

	s1 = h*dydt
	yt = y_input + 0.5*s1

	dydt1 = derivatives(yt,dydt1)

	s2 = h*dydt1
	yt = y_input + 0.5*s2

	dydt2 = derivatives(yt,dydt2)

	s3 = h*dydt2
	yt = y_input + s3

	dydt3 = derivatives(yt,dydt3)

	s4 = h*dydt3
	y_output = y_input + s1/6 + s2/3 + s3/3 + s4/6

	return y_output




for i in range(n_of_steps-1):
	y[0] = th1_list[i]
	y[1] = w1_list[i]
	y[2] = th2_list[i]
	y[3] = w2_list[i]

	y_output = RK(t[i],y,y_output,dt)

	th1_list[i+1] = y_output[0]
	w1_list[i+1] = y_output[1]
	th2_list[i+1] = y_output[2]
	w2_list[i+1] = y_output[3]


#tworzenie listy współrzędnych wahadeł na podstawie kątów i prędkości kątowych 
x1 = L1*sin(th1_list)
y1 = -L1*cos(th1_list)
x2 = x1 + L2*sin(th2_list)
y2 = y1 - L2*cos(th2_list)

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-(L1+L2+1), (L1+L2+1)), ylim=(-(L1+L2+1), (L1+L2+1)))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
	#Kolejne współrzędne punktów: początek, wahadło 1, wahadło 2
    X = [0, x1[i], x2[i]]
    Y = [0, y1[i], y2[i]]

    #Ustawianie danych współrzędnych
    line.set_data(X, Y)
    #Czas trwania ruchu
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, n_of_steps),interval=1, blit=True, init_func=init)


#Ewentualne wyświetlenie samych torów ruchu wahadeł
# plt.plot(x1,y1,c='r')
# plt.plot(x2,y2,c='g')


plt.show()