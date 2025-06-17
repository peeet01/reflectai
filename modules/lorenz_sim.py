import numpy as np
from scipy.integrate import solve_ivp

def generate_lorenz_data(sigma=10.0, rho=28.0, beta=8/3, t_max=25, dt=0.01):
    def lorenz(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz, t_span, [1.0, 1.0, 1.0], t_eval=t_eval, method="RK45")
    return sol.t, sol.y.T  # (N, 3)