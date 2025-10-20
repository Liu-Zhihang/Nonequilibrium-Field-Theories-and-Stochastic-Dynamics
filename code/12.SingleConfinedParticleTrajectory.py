import numpy as np
import matplotlib.pyplot as plt
# Solve Chinese display issue
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # Used to properly display minus sign

# --- Parameter settings ---
D = 1.0          # Diffusion coefficient (m^2/s)
gamma = 0.5      # Relaxation rate (1/s)
T = 50.0         # Total simulation time (s)
dt = 0.01        # Time step (s)
num_steps = int(T / dt) # Total steps
x0 = 10.0        # Initial position

# --- Simulation process ---
t = np.linspace(0, T, num_steps + 1)
x = np.zeros(num_steps + 1)
x[0] = x0

for i in range(num_steps):
    random_increment = np.random.randn()
    # Update with drift and diffusion terms
    x[i+1] = x[i] - gamma * x[i] * dt + np.sqrt(2 * D * dt) * random_increment

# --- Results visualization ---
plt.figure(figsize=(12, 7))
plt.plot(t, x, label='Particle Trajectory')
plt.axhline(0, color='r', linestyle='--', label='Equilibrium Position (x=0)')
plt.title('Trajectory of a Single Particle in a Harmonic Potential Well (O-U Process)', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Position (m)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('single_confined_particle_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()