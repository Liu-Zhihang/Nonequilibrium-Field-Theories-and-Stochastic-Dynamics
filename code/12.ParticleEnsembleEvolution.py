import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Solve Chinese display issue
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # Used to properly display minus sign

# --- Parameter settings ---
D = 1.0          # Diffusion coefficient (m^2/s)
T = 5.0          # Total simulation time (s)
dt = 0.01        # Time step (s)
num_steps = int(T / dt) # Total steps
num_particles = 10000 # Number of particles to simulate
x0 = 0.0         # Initial position

# --- Simulation process ---
# Initialize particle positions, all particles start at x0
x = np.full(num_particles, x0)
# Record particle positions at specific times
snapshots = {}
snapshot_times = [0.1, 1.0, 5.0]

for i in range(num_steps):
    t_current = (i + 1) * dt
    # Update positions of all particles
    random_increment = np.random.randn(num_particles)
    x = x + np.sqrt(2 * D * dt) * random_increment
    
    # Check if we've reached a snapshot time
    for t_snap in snapshot_times:
        if np.isclose(t_current, t_snap):
            snapshots[t_snap] = x.copy()

# --- Results visualization ---
plt.figure(figsize=(14, 8))
bins = np.linspace(-15, 15, 101) # Histogram bins

for t_snap, positions in snapshots.items():
    # Plot histogram of simulated data
    plt.hist(positions, bins=bins, density=True, alpha=0.7, label=f'Simulation t={t_snap}s')
    
    # Calculate and plot theoretical Gaussian distribution
    mean_theory = x0
    variance_theory = 2 * D * t_snap
    std_dev_theory = np.sqrt(variance_theory)
    x_theory = np.linspace(-15, 15, 200)
    pdf_theory = norm.pdf(x_theory, loc=mean_theory, scale=std_dev_theory)
    plt.plot(x_theory, pdf_theory, lw=3, linestyle='--', label=f'Theory t={t_snap}s')

plt.title('Evolution of Position Distribution for Brownian Motion Particle Ensemble', fontsize=16)
plt.xlabel('Position (m)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('particle_ensemble_evolution.png', dpi=300, bbox_inches='tight')
plt.show()