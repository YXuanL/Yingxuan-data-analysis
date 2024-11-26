from scipy.sparse import spdiags
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.fft import fft2, ifft2

## Problem a
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]

X, Y = np.meshgrid(x, y)
w = np.exp(-X**2 - Y**2 / 20).flatten()
w2 = w.reshape(N)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

m=64; n=m*m; L=10; dx= (2*L)/m
e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)
for j in range(1, m+1):
  e2[m*j-1] = 0
  e4[m*j-1] = 1
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
matA = spdiags(diagonals, offsets, n, n).toarray()
A = matA / dx**2

diagonal_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B =  [-(n-m), -m, m, (n-m)]
matB = spdiags(diagonal_B, offsets_B, n, n).toarray()
B = matB / (2*dx)

for i in range(1, n):
    e1[i] = e4[i - 1]

diagonal_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,  m - 1]
matC = spdiags(diagonal_C, offsets_C, n, n).toarray()
C = matC / (2*dx)

import time
start_time = time.time()

def spc_rhs(t, w2, nx, ny, nu, A, B, C, K, N):
    w = w2.reshape((nx,ny))
    wt = fft2(w)
    psit = -wt/K
    psi = np.real(ifft2(psit)).reshape(N)
    rhs = (nu*A.dot(w2)-((C.dot(w2))*(B.dot(psi)))+((C.dot(psi))*(B.dot(w2))))
    return rhs

wtsol = solve_ivp(spc_rhs, [0, 4], w2, method='RK45', t_eval=tspan, args=(nx, ny, nu, A, B, C, K, N))
A1 = wtsol.y
print(A1)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for FFT routine: {elapsed_time:.2f} seconds")

## Problem b1
A[0,0]=2/ (dx**2)

start_time = time.time()

def spc_rhs(t, w, nu, A, B, C):
    psi = np.linalg.solve(A, w)
    rhs = (nu*A.dot(w)-((C.dot(w))*(B.dot(psi)))+((C.dot(psi))*(B.dot(w))))
    return rhs

w_sol = solve_ivp(spc_rhs, [0, 4], w, method='RK45', t_eval=tspan, args=(nu, A, B, C))
A2 = w_sol.y
print(A2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for Ab routine: {elapsed_time:.2f} seconds")

## Problem b2
from scipy.linalg import lu, solve_triangular
A[0,0]=2
P, L, U = lu(A)

start_time = time.time()

def spc_rhs(t, w2, nx, ny, N, K, nu, A, B, C):
    Pb = np.dot(P, w2)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    rhs = (nu*A.dot(w2)-((C.dot(w2))*(B.dot(psi)))+((C.dot(psi))*(B.dot(w2))))
    return rhs

w_sol = solve_ivp(spc_rhs, [0, 4], w2, method='RK45', t_eval=tspan, args=(nx, ny, N, K, nu, A, B, C))
A3 = w_sol.y
print(A3)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for LU routine: {elapsed_time:.2f} seconds")

## Problem b3
#from scipy.sparse.linalg import bicgstab

#residuals_bi = []

#start_time = time.time()

#def bi_callback(residual_norm):
    #residuals_bi.append(residual_norm)

#def bicg_rhs(t, w, nu, A, B, C):
    #psi, exitcode = bicgstab(A, w, tol = 1e-4, callback = bi_callback)
    #rhs = (nu*A.dot(w)-((C.dot(w))*(B.dot(psi)))+((C.dot(psi))*(B.dot(w))))
    #return rhs

#w_sol = solve_ivp(bicg_rhs, [0, 4], w, method='RK45', t_eval=tspan, args=(nu, A, B, C))
#bic_sol = w_sol.y
#print(bic_sol)
#print(len(residuals_bi))

#end_time = time.time()
#elapsed_time = end_time - start_time
#print(f"Elapsed time for BICGSTAB routine: {elapsed_time:.2f} seconds")

## Problem b4
#from scipy.sparse.linalg import gmres

#residuals_gm = []

#start_time = time.time()

#def gm_callback(residual_norm):
    #residuals_gm.append(residual_norm)

#def gm_rhs(t, w, nu, A, B, C):
    #psi, exitcode = gmres(A, w, tol = 1e-4, callback=gm_callback)
    #rhs = (nu*A.dot(w)-((C.dot(w))*(B.dot(psi)))+((C.dot(psi))*(B.dot(w))))
    #return rhs

#w_sol = solve_ivp(gm_rhs, [0, 4], w, method='RK45', t_eval=tspan, args=(nu, A, B, C))
#gm_sol = w_sol.y
#print(gm_sol)
#print(len(residuals_gm))

#end_time = time.time()
#elapsed_time = end_time - start_time
#print(f"Elapsed time for GMRES routine: {elapsed_time:.2f} seconds")

## Problem c1
#omega2 = np.exp(-((X + 5)**2 + Y**2) / 2) - np.exp(-((X - 5)**2 + Y**2) / 2)
#def spc_rhs(t, omega2, nx, ny, nu, A, B, C, K, N):
    #omega2 = omega2.reshape((nx, ny))
    #wt = fft2(omega2)
    #psit = -wt/K
    #psi = np.real(ifft2(psit)).reshape(N)
    #rhs = (nu*A.dot(omega2.flatten())-((C.dot(omega2.flatten()))*(B.dot(psi)))+((C.dot(psi))*(B.dot(omega2.flatten()))))
    #return rhs

#omega2_flat = omega2.flatten()
#w_sol = solve_ivp(spc_rhs, [0, 4], omega2_flat, method='RK45', t_eval=tspan, args=(nx, ny, nu, A, B, C, K, N))
#gm_sol = w_sol.y
#print(gm_sol)

#for j, t in enumerate(tspan):
    #omega_t = gm_sol[:, j].reshape((nx, ny))
    #plt.subplot(3, 3, j + 1)
    #plt.pcolor(x, y, omega_t, shading='interp')
    #plt.title(f'Time: {t}')
    #plt.colorbar()

#plt.tight_layout()
#plt.show()

## Problem c2
#omega2 = np.exp(-((X + 5)**2 + Y**2) / 2) + np.exp(-((X - 5)**2 + Y**2) / 2)
#def spc_rhs(t, omega2, nx, ny, nu, A, B, C, K, N):
    #omega2 = omega2.reshape((nx, ny))
    #wt = fft2(omega2)
    #psit = -wt/K
    #psi = np.real(ifft2(psit)).reshape(N)
    #rhs = (nu*A.dot(omega2.flatten())-((C.dot(omega2.flatten()))*(B.dot(psi)))+((C.dot(psi))*(B.dot(omega2.flatten()))))
    #return rhs

#omega2_flat = omega2.flatten()
#w_sol = solve_ivp(spc_rhs, [0, 4], omega2_flat, method='RK45', t_eval=tspan, args=(nx, ny, nu, A, B, C, K, N))
#gm_sol = w_sol.y
#print(gm_sol)

#for j, t in enumerate(tspan):
    #omega_t = gm_sol[:, j].reshape((nx, ny))
    #plt.subplot(3, 3, j + 1)
    #plt.pcolor(x, y, omega_t, shading='interp')
    #plt.title(f'Time: {t}')
    #plt.colorbar()

#plt.tight_layout()
#plt.show()

## Problem c3
#omega2 = (np.exp(-((X + 5)**2 + (Y + 5)**2) / 2) - np.exp(-((X - 5)**2 + (Y + 5)**2) / 2) +
          #np.exp(-((X + 5)**2 + (Y - 5)**2) / 2) - np.exp(-((X - 5)**2 + (Y - 5)**2) / 2))
#def spc_rhs(t, omega2, nx, ny, nu, A, B, C, K, N):
    #omega2 = omega2.reshape((nx, ny))
    #wt = fft2(omega2)
    #psit = -wt/K
    #psi = np.real(ifft2(psit)).reshape(N)
    #rhs = (nu*A.dot(omega2.flatten())-((C.dot(omega2.flatten()))*(B.dot(psi)))+((C.dot(psi))*(B.dot(omega2.flatten()))))
    #return rhs

#omega2_flat = omega2.flatten()
#w_sol = solve_ivp(spc_rhs, [0, 4], omega2_flat, method='RK45', t_eval=tspan, args=(nx, ny, nu, A, B, C, K, N))
#gm_sol = w_sol.y
#print(gm_sol)

#for j, t in enumerate(tspan):
    #omega_t = gm_sol[:, j].reshape((nx, ny))
    #plt.subplot(3, 3, j + 1)
    #plt.pcolor(x, y, omega_t, shading='interp')
    #plt.title(f'Time: {t}')
    #plt.colorbar()

#plt.tight_layout()
#plt.show()

## Problem c4
#omega2 = np.random.normal(0, 1, (nx, ny))
#def spc_rhs(t, omega2, nx, ny, nu, A, B, C, K, N):
    #omega2 = omega2.reshape((nx, ny))
    #wt = fft2(omega2)
    #psit = -wt/K
    #psi = np.real(ifft2(psit)).reshape(N)
    #rhs = (nu*A.dot(omega2.flatten())-((C.dot(omega2.flatten()))*(B.dot(psi)))+((C.dot(psi))*(B.dot(omega2.flatten()))))
    #return rhs

#omega2_flat = omega2.flatten()
#w_sol = solve_ivp(spc_rhs, [0, 4], omega2_flat, method='RK45', t_eval=tspan, args=(nx, ny, nu, A, B, C, K, N))
#gm_sol = w_sol.y
#print(gm_sol)

#for j, t in enumerate(tspan):
    #omega_t = gm_sol[:, j].reshape((nx, ny))
    #plt.subplot(3, 3, j + 1)
    #plt.pcolor(x, y, omega_t, shading='interp')
    #plt.title(f'Time: {t}')
    #plt.colorbar()

#plt.tight_layout()
#plt.show()

#from matplotlib.animation import FuncAnimation, PillowWriter
#from IPython.display import HTML

#fig, ax = plt.subplots()
#c = ax.pcolor(X, Y, gm_sol[:, 0].reshape((nx, ny)), shading='auto', cmap='viridis')
#fig.colorbar(c, ax=ax)
#time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

## Problem d
#def update(frame):
    #c.set_array(gm_sol[:,frame].flatten())
    #time_text.set_text(f'Time: {tspan[frame]:.2f}')
    #return c, time_text

# Create the animation
#ani = FuncAnimation(fig, update, frames=len(tspan), interval=100, blit=False)

#plt.close(fig)
#HTML(ani.to_jshtml())
#ani.save('fft_random_vortices.gif', writer = 'pillow', fps = 10)