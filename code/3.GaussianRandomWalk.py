import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parameter settings ---
num_walkers = 5000  # Number of simulated particles
num_steps = 100     # Total number of steps
sigma_xi = 1.0      # Standard deviation of single-step displacement (σ_ξ)
plot_times = [10, 30, 100]  # Time points at which to plot the distribution

# --- Simulation process ---
# Generate random displacements for all steps, shape is (num_walkers, num_steps)
# Each step is sampled from N(0, sigma_xi^2)
steps = np.random.normal(loc=0.0, scale=sigma_xi, size=(num_walkers, num_steps))

# Calculate the position of each walker at each step (cumulative sum)
# positions has the same shape as (num_walkers, num_steps)
positions = np.cumsum(steps, axis=1)

# --- Visualization of results ---
plt.figure(figsize=(12, 6))
plt.suptitle('Gaussian Random Walk Simulation', fontsize=16)

# Plot some sample trajectories
plt.subplot(1, 2, 1)
for i in range(5): # Only plot 5 trajectories as examples
    plt.plot(range(1, num_steps + 1), positions[i, :], alpha=0.7)
plt.title('Sample Trajectories')
plt.xlabel('Time Step (t)')
plt.ylabel('Position (X_t)')
plt.grid(True)

# Plot position distributions at specified time points
plt.subplot(1, 2, 2)
for t in plot_times:
    # Theoretical variance
    variance_t = t * sigma_xi**2
    std_dev_t = np.sqrt(variance_t)
    
    # Plot histogram of simulation data
    plt.hist(positions[:, t-1], bins=50, density=True, alpha=0.6, label=f't = {t} (Sim)')
    
    # Plot theoretical Gaussian distribution curve
    x = np.linspace(-4 * std_dev_t, 4 * std_dev_t, 200)
    pdf = norm.pdf(x, loc=0, scale=std_dev_t)
    plt.plot(x, pdf, label=f't = {t} (Theory)')

plt.title('Position Distribution at Different Times')
plt.xlabel('Position (x)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()