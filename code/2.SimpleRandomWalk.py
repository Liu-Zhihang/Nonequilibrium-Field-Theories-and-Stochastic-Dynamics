import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# --- Simulation parameters (inferred from PPT) ---
p = 0.75  # Probability of moving right
q = 1 - p # Probability of moving left
a = 1     # Step size
tau = 1   # Time interval

# --- Figure 1: Probability distribution (left plot) ---
num_steps_dist = 50       # Total number of steps
num_walkers_dist = 50000  # Number of simulated particles for statistical distribution

# Simulation
# Generate displacement for each step (+a or -a)
steps = np.random.choice([a, -a], size=(num_walkers_dist, num_steps_dist), p=[p, q])
# Calculate final position of each particle
final_positions = np.sum(steps, axis=1)

# Plot histogram of simulation results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Use bins to center on even or odd positions
bins = np.arange(final_positions.min(), final_positions.max() + 2) - 0.5
plt.hist(final_positions, bins=bins, density=True, color='blue', alpha=0.7, label=f'Simulation (N={num_walkers_dist})')

# Calculate and plot theoretical binomial distribution
n_values = np.arange(-num_steps_dist, num_steps_dist + 1, 2) # Possible final positions n = k+ - k-
k_plus = (num_steps_dist + n_values) / 2
prob_theory = comb(num_steps_dist, k_plus) * (p**k_plus) * (q**(num_steps_dist - k_plus))
plt.plot(n_values * a, prob_theory, 'ko', label='Theoretical (Binomial)')

plt.title(f'Probability Distribution at k={num_steps_dist}')
plt.xlabel('Position x')
plt.ylabel('Probability P(x, t)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)


# --- Figure 2: Trajectories and statistical moments (right plot) ---
num_steps_traj = 80       # Total number of steps
num_walkers_traj_show = 10 # Number of displayed trajectories
num_walkers_traj_stat = 2000 # Number of particles for calculating mean and variance statistics

# Simulation
steps_stat = np.random.choice([a, -a], size=(num_walkers_traj_stat, num_steps_traj), p=[p, q])
# Calculate position of each particle at each step (cumulative sum)
trajectories = np.cumsum(steps_stat, axis=1)
# Add initial position 0 before trajectories
trajectories = np.insert(trajectories, 0, 0, axis=1)

# Plotting
plt.subplot(1, 2, 2)
time_points = np.arange(num_steps_traj + 1) * tau

# Plot sample trajectories
for i in range(num_walkers_traj_show):
    plt.plot(time_points, trajectories[i, :], alpha=0.5)

# Calculate and plot statistical mean (from simulation)
mean_sim = np.mean(trajectories, axis=0)
# Calculate and plot theoretical mean
mean_theory = time_points * a * (p - q) / tau
plt.plot(time_points, mean_theory, 'k:', linewidth=2, label=r'$\langle x \rangle$')

# Calculate and plot theoretical standard deviation range
variance_theory = 4 * (time_points/tau) * (a**2) * p * q
std_theory = np.sqrt(variance_theory)
plt.plot(time_points, mean_theory + std_theory, 'k--', linewidth=2, label=r'$\langle x \rangle \pm \sigma$')
plt.plot(time_points, mean_theory - std_theory, 'k--', linewidth=2)


plt.title('Random Walk Trajectories and Moments')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()