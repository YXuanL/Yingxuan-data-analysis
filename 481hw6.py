import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft2, ifft2, fftfreq

## Problem 1
def lambda_A(U, V):
    A2 = U ** 2 + V ** 2
    return 1 - A2


def omega_A(U, V):
    A2 = U ** 2 + V ** 2
    return - beta * A2

L = 20 
n = 64 
beta = 1
D1 = 0.1
D2 = 0.1
t_span = np.arange(0, 4.5, 0.5) 

x = np.linspace(-L / 2, L / 2, n, endpoint=False)
y = np.linspace(-L / 2, L / 2, n, endpoint=False)
dx = dy = L / n 
 
X, Y = np.meshgrid(x, y)

m = 1 
U0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - (np.sqrt(X**2 + Y**2)))
V0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - (np.sqrt(X**2 + Y**2)))

def rhs(t, UV):
    U_hat = UV[:n ** 2].reshape((n, n))
    V_hat = UV[n ** 2:].reshape((n, n))

    U = ifft2(U_hat)
    V = ifft2(V_hat)

    dUdt_reaction = lambda_A(U, V) * U - omega_A(U, V) * V
    dVdt_reaction = omega_A(U, V) * U + lambda_A(U, V) * V

    RU_hat = fft2(dUdt_reaction)
    RV_hat = fft2(dVdt_reaction)

    kx = 2 * np.pi * np.fft.fftfreq(U.shape[1], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(U.shape[0], d=dy)
    KX, KY = np.meshgrid(kx, ky)
    lap = - (KX ** 2 + KY ** 2)

    dUdt_diffusion_hat = D1 * lap * U_hat
    dVdt_diffusion_hat = D2 * lap * V_hat

    dUdt = RU_hat + dUdt_diffusion_hat
    dVdt = RV_hat + dVdt_diffusion_hat

    rhs = np.concatenate([dUdt.flatten(), dVdt.flatten()])
    return rhs

UV_0 = np.concatenate([fft2(U0).flatten(), fft2(V0).flatten()])

sol = solve_ivp(rhs,
                (t_span[0], t_span[-1]),
                UV_0,
                t_eval=t_span,
                method='RK45')

A1 = sol.y
print(A1.shape)
print(A1)

## Problem 2
def cheb(N):
    if N == 0:
        D = 0.
        x = 1.
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)
        c = (np.hstack(([2.], np.ones(N - 1), [2.])) * (-1) ** n).reshape(N + 1, 1)
        X = np.tile(x, (1, N + 1))
        dX = X - X.T
        D = np.dot(c, 1. / c.T) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N + 1)

n = 30 
L = 20
beta = 1
n_new = (n + 1) ** 2
D, x = cheb(n)
D[n, :] = 0 
D[0, :] = 0
Dx = np.dot(D, D) / ((L / 2) ** 2) 
y = x  

I = np.eye(len(Dx))
L = np.kron(I, Dx) + np.kron(Dx, I) 

X, Y = np.meshgrid(x, y)
X = X * 10
Y = Y * 10

U0 = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))
V0 = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X ** 2 + Y ** 2))

def rhs(t, uv_t):
    n_rhs = n + 1

    u, v = uv_t[:n_rhs ** 2], uv_t[n_rhs ** 2:]

    dUdt = (lambda_A(u, v) * u - omega_A(u, v) * v) + D1 * (L @ u)
    dVdt = (omega_A(u, v) * u + lambda_A(u, v) * v) + D2 * (L @ v)

    return np.concatenate([dUdt, dVdt])

UV0 = np.concatenate([U0.reshape(n_new), V0.reshape(n_new)])

solution = solve_ivp(rhs, (t_span[0], t_span[-1]), UV0, t_eval=t_span, method='RK45')

A2 = solution.y
print(A2.shape)
print(A2)
