import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Solve Chinese display issue
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # Used to properly display minus sign

# --- Parameter settings ---
D = 1.0
gamma = 0.5
T = 10.0
dt = 0.01
num_steps = int(T / dt)
num_particles = 10000
x0 = 5.0

# --- Simulation process ---
t = np.linspace(0, T, num_steps + 1)
x = np.full((num_particles, num_steps + 1), x0)

# Record mean and variance during simulation
mean_sim = np.zeros(num_steps + 1)
var_sim = np.zeros(num_steps + 1)
mean_sim[0] = x0
var_sim[0] = 0

for i in range(num_steps):
    random_increment = np.random.randn(num_particles)
    x[:, i+1] = x[:, i] - gamma * x[:, i] * dt + np.sqrt(2 * D * dt) * random_increment
    mean_sim[i+1] = np.mean(x[:, i+1])
    var_sim[i+1] = np.var(x[:, i+1])

# --- Theoretical solution ---
mean_theory = x0 * np.exp(-gamma * t)
var_theory = (D / gamma) * (1 - np.exp(-2 * gamma * t))

# --- Results visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Mean comparison
ax1.plot(t, mean_sim, 'o', markersize=4, label='Simulated Mean')
ax1.plot(t, mean_theory, 'r-', lw=3, label='Theoretical Mean')
ax1.set_ylabel(r'Average Position $\langle x \rangle$', fontsize=14)
ax1.set_title('Evolution of Mean and Variance in O-U Process', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Variance comparison
ax2.plot(t, var_sim, 'o', markersize=4, label='Simulated Variance')
ax2.plot(t, var_theory, 'r-', lw=3, label='Theoretical Variance')
ax2.axhline(D / gamma, color='k', linestyle='--', label=f'Steady-state Variance D/γ = {D/gamma:.2f}')
ax2.set_xlabel('Time (s)', fontsize=14)
ax2.set_ylabel('Variance Var[x]', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('confined_particle_ensemble_evolution.png', dpi=300, bbox_inches='tight')
plt.show()