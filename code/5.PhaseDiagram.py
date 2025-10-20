# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Define Lotka-Volterra ODE system ---
def lotka_volterra_ode(y, t, lambda_rate, mu_rate, gamma_p, gamma_c):
    a, b = y
    da_dt = mu_rate * a - gamma_p * a * b
    db_dt = -lambda_rate * b + gamma_p * a * b - gamma_c * b
    return [da_dt, db_dt]

# --- Set parameters ---
lambda_rate = 1.0
mu_rate = 1.0
gamma_p = 1.0
gamma_c = 0.0

# --- Plot phase diagrams ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# --- Case 1: gamma_c = 0 (neutral center) ---
gamma_c = 0.0
ax = axes[0]
a_max, b_max = 2.0, 2.0
a_grid, b_grid = np.meshgrid(np.linspace(0, a_max, 20), np.linspace(0, b_max, 20))

u, v = np.zeros_like(a_grid), np.zeros_like(b_grid)
NI, NJ = a_grid.shape
for i in range(NI):
    for j in range(NJ):
        y = [a_grid[i, j], b_grid[i, j]]
        dydt = lotka_volterra_ode(y, 0, lambda_rate, mu_rate, gamma_p, gamma_c)
        u[i, j], v[i, j] = dydt[0], dydt[1]

# Plot flow field
ax.streamplot(a_grid, b_grid, u, v, density=1.2, color='darkslategrey')

# Plot fixed point
a_star = mu_rate / gamma_p
b_star = lambda_rate / gamma_p
ax.plot(a_star, b_star, 'o', markersize=8, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)

# Plot sample trajectories
initial_conditions = [[a_star*1.5, b_star*1.5], [a_star*0.5, b_star*0.5]]
t_span = np.linspace(0, 50, 500)
colors = ['red', 'orange']
for i, y0 in enumerate(initial_conditions):
    sol = odeint(lotka_volterra_ode, y0, t_span, args=(lambda_rate, mu_rate, gamma_p, gamma_c))
    ax.plot(sol[:, 0], sol[:, 1], color=colors[i], lw=2, alpha=0.8)

ax.set_xlabel('a', fontsize=14)
ax.set_ylabel('b', fontsize=14)
ax.set_xlim(0, a_max)
ax.set_ylim(0, b_max)

# --- Case 2: gamma_c > 0 (stable spiral point) ---
gamma_c = 0.001
ax = axes[1]
a_max, b_max = 2.0, 2.0
a_grid, b_grid = np.meshgrid(np.linspace(0, a_max, 20), np.linspace(0, b_max, 20))

u, v = np.zeros_like(a_grid), np.zeros_like(b_grid)
NI, NJ = a_grid.shape
for i in range(NI):
    for j in range(NJ):
        y = [a_grid[i, j], b_grid[i, j]]
        dydt = lotka_volterra_ode(y, 0, lambda_rate, mu_rate, gamma_p, gamma_c)
        u[i, j], v[i, j] = dydt[0], dydt[1]

ax.streamplot(a_grid, b_grid, u, v, density=1.2, color='darkslategrey')

# Plot fixed point
b_star_damped = lambda_rate / gamma_p
a_star_damped = (mu_rate - gamma_c * b_star_damped) / gamma_p
ax.plot(a_star_damped, b_star_damped, 'o', markersize=8, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5)

# Plot sample trajectories
initial_conditions = [[a_star_damped*1.5, b_star_damped*1.5], [a_star_damped*0.5, b_star_damped*0.5]]
t_span = np.linspace(0, 50, 500)
colors = ['red', 'orange']
for i, y0 in enumerate(initial_conditions):
    sol = odeint(lotka_volterra_ode, y0, t_span, args=(lambda_rate, mu_rate, gamma_p, gamma_c))
    ax.plot(sol[:, 0], sol[:, 1], color=colors[i], lw=2, alpha=0.8)

ax.set_xlabel('a', fontsize=14)
ax.set_ylabel('b', fontsize=14)
ax.set_xlim(0, a_max)
ax.set_ylim(0, b_max)

fig.suptitle('Phase Portraits', fontsize=20, color='darkgreen', y=1.0)
plt.tight_layout()
plt.show()