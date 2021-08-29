import numpy as np
from scipy import constants as const
from scipy.spatial.distance import euclidean as dist
import matplotlib.pyplot as plt
n = 3
e = const.elementary_charge # C
mp = const.proton_mass # kg
me = const.electron_mass # kg
eps0 = const.epsilon_0
k = 1/(4*np.pi*eps0)
# v_e_0 = 2180*10**3 #m/s
v_e_0 = 64.0 #m/s
r_e_0 = 31*10**-12 #m

R = np.zeros((n, 3)) #m
V = np.zeros((n, 3)) #m/s
A = np.zeros((n, 3)) #m/s^2

R[1,0] = r_e_0
R[2,0] = -r_e_0
V[1,1] = v_e_0
V[2,1] = -v_e_0
Q = -e*np.ones((n,1))

Q[0] = e
lista = []

qq = np.zeros(n*(n-1))
r = np.zeros(n*(n-1))
m = np.zeros((n*(n-1),1))
e = np.zeros((n*(n-1),3))

t_f = 10 * r_e_0/v_e_0 # s
dt = t_f * 10**-4

maxiters = 8000
iters = 0

AllT = np.zeros((maxiters*n,3))
AllR = np.zeros((maxiters*n,3))

while iters < maxiters:
    k = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                lista.append([i, j])

                qq[k] = Q[i]*Q[j]

                r[k] = dist(R[i], R[j])

                e[k] = (R[i]-R[j])/r[k]
                
                if Q[i]>0:
                    m[k] = mp
                else:
                    m[k] = me
                    
                k+=1

    F = k* qq/r**2
    F = F.reshape((n*(n-1),1))
    F = F*e
        
    A = F/m
    
    f = np.zeros((n,3))
    a = np.zeros((n,3))
    k = 0
    
    for i in range(0, n*(n-1), n-1):
        f[k] = sum(F[i:i+(n-1)])
        a[k] = sum(A[i:i+(n-1)])
        # print("%d: F = %ei + %ej + %ek; a = %ei + %ej + %ek" %(lista[i][0], f[k][0],f[k][1],f[k][2], a[k][0],a[k][1],a[k][2]))
        k+=1
    # print("r = ", r)
    # print("v = ", V)
    # print(a)
    V += a*dt
    # print(a*dt)
    R += V*dt + 1/2 *a*dt**2
    
    AllR[iters*n:(iters+1)*n] = dt*iters
    AllR[iters*n:(iters+1)*n] = R
    iters += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")


ax.scatter(AllR[0::3,0], AllR[0::3,1], AllR[0::3,2])
ax.scatter(AllR[1::3,0], AllR[1::3,1], AllR[1::3,2])
ax.scatter(AllR[2::3,0], AllR[2::3,1], AllR[2::3,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()