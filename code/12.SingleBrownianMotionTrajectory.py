import numpy as np
import matplotlib.pyplot as plt

# Solve Chinese display issue
plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # Used to properly display minus sign

# --- Parameter settings ---
D = 1.0          # Diffusion coefficient (m^2/s)
T = 10.0         # Total simulation time (s)
dt = 0.01        # Time step (s)
num_steps = int(T / dt) # Total steps
num_particles = 5 # Number of particles to simulate

# --- Simulation process ---
# Create time axis
t = np.linspace(0, T, num_steps + 1)
# Initialize particle position array, all particles start at x=0
x = np.zeros((num_particles, num_steps + 1))

# Euler-Maruyama method for iteration
for i in range(num_steps):
    # Generate random increment
    random_increment = np.random.randn(num_particles)
    # Update positions of all particles
    x[:, i+1] = x[:, i] + np.sqrt(2 * D * dt) * random_increment

# --- Results visualization ---
plt.figure(figsize=(12, 7))
for i in range(num_particles):
    plt.plot(t, x[i, :], label=f'Particle {i+1}')

plt.title('Trajectories of 5 Independent Brownian Motion Particles (Wiener Process)', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Position (m)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('single_brownian_motion_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()